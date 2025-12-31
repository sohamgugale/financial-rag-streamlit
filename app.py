import streamlit as st
import anthropic
import pypdf
from rank_bm25 import BM25Okapi
from io import BytesIO
import os

st.set_page_config(page_title="Financial RAG Assistant", page_icon="📊", layout="wide")

# Initialize Anthropic client
@st.cache_resource
def get_anthropic_client():
    api_key = os.getenv("ANTHROPIC_API_KEY") or st.secrets.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        st.error("Please set ANTHROPIC_API_KEY in secrets!")
        st.stop()
    return anthropic.Anthropic(api_key=api_key)

class Document:
    def __init__(self, content, metadata):
        self.page_content = content
        self.metadata = metadata

def extract_pdf_text(pdf_file):
    """Extract text from PDF with smaller chunks"""
    pdf_reader = pypdf.PdfReader(BytesIO(pdf_file.read()))
    documents = []
    
    for page_num, page in enumerate(pdf_reader.pages, 1):
        text = page.extract_text()
        if text.strip():
            # Smaller chunks - 3 sentences instead of 5
            sentences = text.split('. ')
            for i in range(0, len(sentences), 3):
                chunk = '. '.join(sentences[i:i+3])
                if chunk.strip() and len(chunk) > 50:  # Skip tiny chunks
                    documents.append(Document(
                        content=chunk[:500],  # Limit chunk size to 500 chars
                        metadata={"filename": pdf_file.name, "page": page_num}
                    ))
    
    return documents, len(pdf_reader.pages)

def search_documents(query, documents, k=3):
    """Search documents using BM25 - only top 3 results"""
    corpus = [doc.page_content.lower().split() for doc in documents]
    bm25 = BM25Okapi(corpus)
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [documents[i] for i in top_indices]

def answer_question(query, documents):
    """Answer question using RAG with limited context"""
    client = get_anthropic_client()
    
    # Get only top 3 most relevant documents
    relevant_docs = search_documents(query, documents, k=3)
    
    # Build concise context
    context_parts = []
    for doc in relevant_docs:
        context_parts.append(
            f"[{doc.metadata['filename']}, p.{doc.metadata['page']}]: {doc.page_content[:300]}"
        )
    
    context = "\n\n".join(context_parts)
    
    # Limit total context to 3000 characters
    if len(context) > 3000:
        context = context[:3000] + "..."
    
    try:
        # Query Claude with limited context
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": f"Answer based on this context. Cite sources.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
            }]
        )
        
        return message.content[0].text, relevant_docs
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return f"Error querying Claude: {str(e)}", []

# UI
st.title("📊 Financial Research Assistant")
st.markdown("Upload financial documents and ask questions")

# Initialize session state
if "documents" not in st.session_state:
    st.session_state.documents = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar for file upload
with st.sidebar:
    st.header("📁 Upload Documents")
    uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    
    if uploaded_files:
        if st.button("Process Documents"):
            with st.spinner("Processing documents..."):
                st.session_state.documents = []
                for pdf_file in uploaded_files:
                    docs, pages = extract_pdf_text(pdf_file)
                    st.session_state.documents.extend(docs)
                    st.success(f"✅ {pdf_file.name} ({pages} pages, {len(docs)} chunks)")
    
    st.divider()
    st.markdown(f"**Total:** {len(st.session_state.documents)} chunks")
    
    if st.button("Clear All"):
        st.session_state.documents = []
        st.session_state.chat_history = []
        st.rerun()

# Main chat interface
if st.session_state.documents:
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                with st.expander("📄 Sources"):
                    for doc in message["sources"]:
                        st.markdown(f"**{doc.metadata['filename']}** - Page {doc.metadata['page']}")
                        st.text(doc.page_content[:200] + "...")
                        st.divider()
    
    # Chat input
    if prompt := st.chat_input("Ask a question..."):
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, sources = answer_question(prompt, st.session_state.documents)
                st.markdown(answer)
                
                if sources:
                    with st.expander("📄 Sources"):
                        for doc in sources:
                            st.markdown(f"**{doc.metadata['filename']}** - Page {doc.metadata['page']}")
                            st.text(doc.page_content[:200] + "...")
                            st.divider()
        
        # Add assistant message
        st.session_state.chat_history.append({
            "role": "assistant", 
            "content": answer,
            "sources": sources
        })

else:
    st.info("👈 Upload PDF documents to get started!")
    
    st.markdown("### Example Questions:")
    st.markdown("- What is the total revenue?")
    st.markdown("- Summarize the main risks")
    st.markdown("- What are the business segments?")

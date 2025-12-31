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
    return anthropic.Client(api_key=api_key)

class Document:
    def __init__(self, content, metadata):
        self.page_content = content
        self.metadata = metadata

def extract_pdf_text(pdf_file):
    """Extract text from PDF"""
    pdf_reader = pypdf.PdfReader(BytesIO(pdf_file.read()))
    documents = []
    
    for page_num, page in enumerate(pdf_reader.pages, 1):
        text = page.extract_text()
        if text.strip():
            sentences = text.split('. ')
            for i in range(0, len(sentences), 5):
                chunk = '. '.join(sentences[i:i+5])
                if chunk.strip():
                    documents.append(Document(
                        content=chunk,
                        metadata={"filename": pdf_file.name, "page": page_num}
                    ))
    
    return documents, len(pdf_reader.pages)

def search_documents(query, documents, k=5):
    """Search documents using BM25"""
    corpus = [doc.page_content.lower().split() for doc in documents]
    bm25 = BM25Okapi(corpus)
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [documents[i] for i in top_indices]

def answer_question(query, documents):
    """Answer question using RAG"""
    client = get_anthropic_client()
    
    # Get relevant documents
    relevant_docs = search_documents(query, documents)
    
    # Build context
    context = "\n\n".join([
        f"Source: {doc.metadata['filename']} (Page {doc.metadata['page']})\n{doc.page_content}"
        for doc in relevant_docs
    ])
    
    # Query Claude
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1500,
        messages=[{
            "role": "user",
            "content": f"Based on the following context, answer the question. Always cite the source document and page.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        }]
    )
    
    return message.content[0].text, relevant_docs

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
    st.markdown(f"**Total Documents:** {len(st.session_state.documents)} chunks")

# Main chat interface
if st.session_state.documents:
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                with st.expander("📄 View Sources"):
                    for doc in message["sources"]:
                        st.markdown(f"**{doc.metadata['filename']}** - Page {doc.metadata['page']}")
                        st.text(doc.page_content[:200] + "...")
                        st.divider()
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, sources = answer_question(prompt, st.session_state.documents)
                st.markdown(answer)
                
                with st.expander("📄 View Sources"):
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
    
    # Example
    st.markdown("### Example Questions:")
    st.markdown("- What are the key financial metrics?")
    st.markdown("- Summarize the main risks")
    st.markdown("- What revenue growth is reported?")

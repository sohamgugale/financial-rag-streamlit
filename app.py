import streamlit as st
import google.generativeai as genai
import pypdf
from rank_bm25 import BM25Okapi
from io import BytesIO
import os

st.set_page_config(page_title="Financial RAG Assistant", page_icon="📊", layout="wide")

# Initialize Gemini
@st.cache_resource
def get_gemini_model():
    api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", "")
    if not api_key:
        st.error("Please set GEMINI_API_KEY in secrets!")
        st.stop()
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-1.5-flash')

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
            for i in range(0, len(sentences), 3):
                chunk = '. '.join(sentences[i:i+3])
                if chunk.strip() and len(chunk) > 50:
                    documents.append(Document(
                        content=chunk[:500],
                        metadata={"filename": pdf_file.name, "page": page_num}
                    ))
    
    return documents, len(pdf_reader.pages)

def search_documents(query, documents, k=3):
    """Search documents using BM25"""
    corpus = [doc.page_content.lower().split() for doc in documents]
    bm25 = BM25Okapi(corpus)
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [documents[i] for i in top_indices]

def answer_question(query, documents):
    """Answer question using RAG with Gemini"""
    model = get_gemini_model()
    
    # Get top 3 relevant documents
    relevant_docs = search_documents(query, documents, k=3)
    
    # Build context
    context_parts = []
    for doc in relevant_docs:
        context_parts.append(
            f"[{doc.metadata['filename']}, p.{doc.metadata['page']}]: {doc.page_content[:300]}"
        )
    
    context = "\n\n".join(context_parts)
    
    if len(context) > 3000:
        context = context[:3000] + "..."
    
    try:
        prompt = f"""Answer based on this context. Always cite the source document and page number.

Context:
{context}

Question: {query}

Answer:"""
        
        response = model.generate_content(prompt)
        return response.text, relevant_docs
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return f"Error: {str(e)}", []

# UI
st.title("📊 Financial Research Assistant")
st.markdown("Upload financial documents and ask questions • Powered by Google Gemini (FREE)")

# Initialize session state
if "documents" not in st.session_state:
    st.session_state.documents = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar
with st.sidebar:
    st.header("📁 Upload Documents")
    uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    
    if uploaded_files:
        if st.button("Process Documents"):
            with st.spinner("Processing..."):
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

# Chat interface
if st.session_state.documents:
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                with st.expander("📄 Sources"):
                    for doc in message["sources"]:
                        st.markdown(f"**{doc.metadata['filename']}** - Page {doc.metadata['page']}")
                        st.text(doc.page_content[:200] + "...")
                        st.divider()
    
    if prompt := st.chat_input("Ask a question..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
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

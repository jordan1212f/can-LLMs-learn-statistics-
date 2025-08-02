"""
Multi-Document RAG System for Educational Content
================================================

This script creates a Retrieval-Augmented Generation (RAG) system that can handle
multiple PDF documents, providing source-aware responses with proper citations.

Features:
- Multi-PDF loading and processing
- Enhanced tutoring prompt template
- Source-aware retrieval and responses
- Configurable chunking parameters
- Built-in search and utility functions

Author: Jordan Fernandes
Date: January 2025
Purpose: Dissertation research on educational RAG systems
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import OnlinePDFLoader
from pathlib import Path
from typing import List, Dict, Any
import json

# Additional imports for RAG functionality
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import os
import sys

# Configuration
PDF_DIRECTORY = "/Users/jordanfernandes/Desktop/Dissertation workspace/datasets_pdfs"
model = "llama3.2"

def load_multiple_pdfs(pdf_configs: List[Dict[str, Any]]) -> List:
    """
    Load and process multiple PDFs with specific page ranges
    
    Args:
        pdf_configs: List of dicts with 'path', 'name', optional 'pages' (tuple or None)
    
    Returns:
        List of Document objects with metadata
    """
    all_documents = []
    
    for config in pdf_configs:
        try:
            pdf_path = config['path']
            pdf_name = config['name']
            page_range = config.get('pages', None)  # (start, end) or None for all pages
            
            print(f"Loading {pdf_name}...")
            
            if pdf_path.startswith('http'):
                loader = OnlinePDFLoader(pdf_path)
            else:
                loader = PyPDFLoader(pdf_path)
            
            # Load all pages first
            documents = loader.load()
            
            # Filter by page range if specified
            if page_range:
                start_page, end_page = page_range
                documents = documents[start_page:end_page + 1]
                print(f"  Using pages {start_page}-{end_page} ({len(documents)} pages)")
            else:
                print(f"  Using all pages ({len(documents)} pages)")
            
            # Add source metadata
            for doc in documents:
                doc.metadata.update({
                    'source_document': pdf_name,
                    'file_path': pdf_path
                })
            
            all_documents.extend(documents)
            
        except Exception as e:
            print(f"Error loading {config.get('name', 'unknown')}: {e}")
            continue
    
    print(f"\nTotal documents loaded: {len(all_documents)}")
    return all_documents

def get_pdf_configs():
    """
    Configure which PDFs to load. Easily extendable for 8-10+ PDFs.
    
    Returns:
        List of PDF configurations
    """
    configs = [
        {
            'path': os.path.join(PDF_DIRECTORY, "os.pdf"),
            'name': "OpenStax Statistics",
            'pages': None  # Load all pages
        },
        # Add more PDFs here as needed:
        # {
        #     'path': os.path.join(PDF_DIRECTORY, "think_stats.pdf"),
        #     'name': "Think Stats",
        #     'pages': (0, 200)  # First 200 pages only
        # },
        # {
        #     'path': "https://example.com/statistics_book.pdf",
        #     'name': "Online Stats Book",
        #     'pages': None
        # }
    ]
    
    return configs

def create_vector_store(documents: List, chunk_size: int = 1200, chunk_overlap: int = 300):
    """
    Create vector store from documents with configurable chunking.
    """
    print("Creating vector store...")
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    
    splits = text_splitter.split_documents(documents)
    print(f"Created {len(splits)} chunks")
    
    # Create embeddings and vector store
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_db = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    
    print("Vector store created successfully")
    return vector_db

def check_ollama_model(model_name: str):
    """Check if Ollama model is available."""
    try:
        import ollama
        models = ollama.list()
        model_names = [model['name'] for model in models['models']]
        
        if any(model_name in name for name in model_names):
            print(f"✓ Model {model_name} is available")
            return True
        else:
            print(f"✗ Model {model_name} not found")
            print("Available models:", model_names)
            return False
    except Exception as e:
        print(f"Error checking Ollama models: {e}")
        return False

def create_rag_chain(vector_db, model_name: str = "llama3.2"):
    """
    Create RAG chain with enhanced prompt template for tutoring.
    """
    from langchain.prompts import ChatPromptTemplate, PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_ollama import ChatOllama
    from langchain_core.runnables import RunnablePassthrough
    from langchain.retrievers import MultiQueryRetriever
    
    # Check model availability
    if not check_ollama_model(model_name):
        return None, None
    
    # Setup retriever
    retriever = vector_db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    
    # Setup LLM
    llm = ChatOllama(
        model=model_name,
        temperature=0.1
    )
    
    # Enhanced multi-query retriever for better results
    query_prompt = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate five 
        different versions of the given user question to retrieve relevant documents from a vector 
        database. By generating multiple perspectives on the user question, your goal is to help 
        the user overcome some of the limitations of the distance-based similarity search. 
        Provide these alternative questions separated by newlines.
        Original question: {question}"""
    )
    
    retriever = MultiQueryRetriever.from_llm(
        llm,
        prompt=query_prompt,   # your sub-query prompt
    )   
    
    # Enhanced prompt template for multi-document tutoring context
    tutor_template = """You are a patient, step-by-step statistics tutor. Use the following context excerpts from multiple textbooks to answer the student's question.

Context from textbooks:
{context}

When you answer, please:

1. **Restate the question** to confirm understanding.
2. **List what is given** (data, formulas, definitions) and any assumptions.
3. **Outline your solution strategy** ("We will do X, then Y...").
4. **Work through the solution in numbered steps**, showing intermediate calculations, code snippets (in Python or R) or formula applications as needed.
5. **Cite each fact or formula** inline (e.g. "[OpenStax Ch. 7]", "[Think Stats Sec 4.2]").
6. **Summarize the final answer** in plain language at the end.
7. If the answer is not fully contained in the provided context, say "I don't know" rather than guessing.

**Question:** {question}

**Answer as a tutor:**"""
    
    prompt = ChatPromptTemplate.from_template(tutor_template)

    def format_docs_with_sources(docs):
        """Format documents with source information."""
        formatted = []
        for doc in docs:
            source = doc.metadata.get('source_document', 'Unknown Source')
            page = doc.metadata.get('page', 'Unknown Page')
            content = doc.page_content
            formatted.append(f"[Source: {source}, Page: {page}]\n{content}")
        return "\n\n".join(formatted)
    
    # Create RAG chain
    rag_chain = (
        {"context": retriever | format_docs_with_sources, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain, retriever

def search_by_source(vector_db, query: str, source_filter: str = None, k: int = 3):
    """
    Search with optional source filtering.
    
    Args:
        vector_db: The Chroma vector database
        query: Search query
        source_filter: Filter by source document name (optional)
        k: Number of results to return
    
    Returns:
        List of relevant documents
    """
    if source_filter:
        # Search with metadata filter
        results = vector_db.similarity_search(
            query, 
            k=k, 
            filter={"source_document": source_filter}
        )
        print(f"Found {len(results)} results from '{source_filter}'")
    else:
        results = vector_db.similarity_search(query, k=k)
        print(f"Found {len(results)} results from all sources")
    
    return results

def get_available_sources(vector_db):
    """Get list of available source documents."""
    # This is a simple way - in production you might want to store this metadata separately
    sample_docs = vector_db.similarity_search("statistics", k=50)  # Get diverse sample
    sources = set()
    for doc in sample_docs:
        source = doc.metadata.get('source_document', 'Unknown')
        if source != 'Unknown':
            sources.add(source)
    return sorted(list(sources))

def ask_question(rag_chain, question: str):
    """Ask a question and get a tutored response."""
    print(f"\n🎓 Question: {question}")
    print("=" * 60)
    
    try:
        response = rag_chain.invoke(question)
        print(response)
        return response
    except Exception as e:
        print(f"Error: {e}")
        return None

def main():
    """
    Main function to set up and run the multi-document RAG system.
    """
    print("🚀 Setting up Multi-Document RAG System")
    print("=" * 50)
    
    # Step 1: Load PDF configurations
    pdf_configs = get_pdf_configs()
    
    # Step 2: Load documents
    documents = load_multiple_pdfs(pdf_configs)
    
    if not documents:
        print("❌ No documents loaded. Please check your PDF configurations.")
        return
    
    # Step 3: Create vector store
    vector_db = create_vector_store(documents)
    
    # Step 4: Create RAG chain
    rag_chain, retriever = create_rag_chain(vector_db, model)
    
    if not rag_chain:
        print("❌ Failed to create RAG chain. Please check your Ollama installation.")
        return
    
    print("\n✅ RAG system ready!")
    print(f"📚 Loaded {len(documents)} documents")
    
    # Show available sources
    sources = get_available_sources(vector_db)
    print(f"📖 Available sources: {', '.join(sources)}")
    
    # Example usage
    print("\n" + "=" * 60)
    print("🎯 EXAMPLE USAGE")
    print("=" * 60)
    
    # Example question
    sample_question = "What is the central limit theorem and how does it work?"
    ask_question(rag_chain, sample_question)
    
    print("\n" + "=" * 60)
    print("🔍 Interactive Mode")
    print("=" * 60)
    print("You can now ask questions! Type 'quit' to exit.")
    
    while True:
        try:
            question = input("\n❓ Your question: ").strip()
            if question.lower() in ['quit', 'exit', 'q']:
                break
            if question:
                ask_question(rag_chain, question)
        except KeyboardInterrupt:
            break
    
    print("\n👋 Thanks for using the Multi-Document RAG System!")

if __name__ == "__main__":
    main()

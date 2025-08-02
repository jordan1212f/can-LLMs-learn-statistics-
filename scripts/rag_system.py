from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import OnlinePDFLoader
from pathlib import Path
from typing import List, Dict, Any
import json

# Configuration
PDF_DIRECTORY = "/Users/jordanfernandes/Desktop/Dissertation workspace/datasets_pdfs"
model = "llama3.2"

def load_multiple_pdfs(pdf_configs: List[Dict[str, Any]]) -> List:
    """
    Load and process multiple PDFs with specific page ranges
    Args: pdf_configs: List of dictionaries with 'path', 'start_page', 'end_page', 'name'
    Returns: List of processed documents with metadata
    """
    all_documents = []
    
    for config in pdf_configs:
        pdf_path = config['path']
        start_page = config.get('start_page', None)
        end_page = config.get('end_page', None)
        doc_name = config.get('name', Path(pdf_path).stem)
        
        print(f"\nProcessing: {doc_name}")
        print(f"File: {pdf_path}")
        
        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load_and_split()
            print(f"Loaded {len(docs)} total pages")
            
            # relevant pages as we are focused on confidence intervals and hypothesis testing
            if start_page is not None and end_page is not None:
                selected_docs = docs[start_page:end_page + 1]
                print(f"Selected pages {start_page}-{end_page} ({len(selected_docs)} pages)")
            else:
                selected_docs = docs
                print(f"Using all {len(selected_docs)} pages")
            
            #individual document metadata
            for doc in selected_docs:
                doc.metadata.update({
                    'source_document': doc_name,
                    'file_path': pdf_path,
                    'selected_range': f"{start_page}-{end_page}" if start_page and end_page else "all"
                })
            
            all_documents.extend(selected_docs)
            print(f"Added {len(selected_docs)} pages from {doc_name}")
            
        except Exception as e:
            print(f"Error processing {pdf_path}: {str(e)}")
            continue
    
    print(f"\nTotal documents loaded: {len(all_documents)}")
    return all_documents

#PDF corpus this is where i can append multiple pdfs
pdf_corpus = [
    {
        'path': f"{PDF_DIRECTORY}/os.pdf",
        'start_page': 180,
        'end_page': 313,
        'name': "OpenIntro Statistics - Chapters 5-7"
    },
    # {
    #     'path': f"{PDF_DIRECTORY}/another_stats_book.pdf",
    #     'start_page': 0,
    #     'end_page': 100,
    #     'name': "Statistics Textbook 2"
    # },
    # {
    #     'path': f"{PDF_DIRECTORY}/probability_book.pdf",
    #     'start_page': None,  # Use all pages
    #     'end_page': None,
    #     'name': "Probability Theory"
    # }
]
#load all documents from the corupus aboe
print("Loading PDF corpus...")
all_docs = load_multiple_pdfs(pdf_corpus)

#extract and chunk      
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

def check_ollama_connection():
    """Check if Ollama is installed and running."""
    try:
        import ollama
        # Try to list models to check connection
        models = ollama.list()
        print("✅ Ollama is running")
        return True
    except ImportError:
        print("❌ Ollama Python package not installed. Run: pip install ollama")
        return False
    except Exception as e:
        if "Failed to connect" in str(e):
            print("❌ Ollama is not running. Please start Ollama first.")
            print("   - If not installed: Download from https://ollama.com/download")
            print("   - If installed: Run 'ollama serve' in terminal or start Ollama app")
        else:
            print(f"❌ Ollama connection error: {e}")
        return False

def ensure_ollama_model(model_name: str = "nomic-embed-text"):
    """Ensure the embedding model is available."""
    try:
        import ollama
        models = ollama.list()
        model_names = [model['name'].split(':')[0] for model in models['models']]
        
        if model_name in model_names:
            print(f"✅ Model {model_name} is available")
            return True
        else:
            print(f"📥 Pulling model {model_name}...")
            ollama.pull(model_name)
            print(f"✅ Model {model_name} downloaded successfully")
            return True
    except Exception as e:
        print(f"❌ Error with model {model_name}: {e}")
        return False

def create_vector_store(documents: List, chunk_size: int = 1200, chunk_overlap: int = 300):
    """
    Create vector store from multiple documents.
    Args: documents: List of loaded documents, chunk_size: Size of text chunks, chunk_overlap: Overlap between chunks
    Returns: Vector store and chunks
    """
    print(f"\nCreating chunks with size={chunk_size}, overlap={chunk_overlap}")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(documents)
    
    print(f"Created {len(chunks)} chunks from {len(documents)} documents")
    
    #display chunk distribution by source document
    chunk_sources = {}
    for chunk in chunks:
        source = chunk.metadata.get('source_document', 'unknown')
        chunk_sources[source] = chunk_sources.get(source, 0) + 1
    
    print("\nChunk distribution by source:")
    for source, count in chunk_sources.items():
        print(f"  {source}: {count} chunks")
    
    # Check Ollama connection before proceeding
    if not check_ollama_connection():
        print("\n🛑 Cannot proceed without Ollama. Please install and start Ollama first.")
        return None, None
    
    # Ensure embedding model is available
    if not ensure_ollama_model("nomic-embed-text"):
        print("\n🛑 Cannot proceed without embedding model.")
        return None, None
    
    #create vector store
    print("Creating vector store...")
    try:
        vector_db = Chroma.from_documents(
            documents=chunks,
            embeddings=OllamaEmbeddings(model="nomic-embed-text"),
            collection_name="multi_document_rag_system",
        )
        
        print(f"✅ Vector store created with {len(chunks)} chunks from {len(set(chunk_sources.keys()))} documents")
        return vector_db, chunks
    except Exception as e:
        print(f"❌ Error creating vector store: {e}")
        return None, None

#create vector store from all loaded documents
print("\n" + "="*50)
print("SETTING UP VECTOR STORE")
print("="*50)

vector_db, all_chunks = create_vector_store(all_docs)

# Check if vector store creation was successful
if vector_db is None or all_chunks is None:
    print("\n❌ RAG system setup failed. Please fix the issues above and try again.")
    print("\n📋 Setup checklist:")
    print("1. Install Ollama: https://ollama.com/download")
    print("2. Start Ollama (run 'ollama serve' or start the app)")
    print("3. Install required packages: pip install ollama langchain langchain-community langchain-ollama")
    print("4. Re-run this script")
    exit(1)

# Retrieval and QA Setup
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_retrievers import MultiQueryRetriever

QUERY_PROMPT = """You are given a student’s complex statistics question. 
Rewrite it as 3–5 concise search queries that will help retrieve every relevant textbook passage.
For example:
Question: “How do I compute a 95% CI for the difference of two means with unequal variances?”
Sub‑queries:
1. “confidence interval two sample means unequal variances”
2. “Welch’s t test confidence interval formula”
3. “degrees of freedom unequal variance t test”
---
Now rewrite: {question}
"""

def setup_rag_chain(vector_db, model_name: str = "llama3.2"):
    """
    Set up RAG chain for multi-document retrieval.
    Args: vector_db: Vector store containing document chunks, model_name: LLM model name
    Returns: Configured RAG chain and retriever, or (None, None) if failed
    """
    # Check Ollama connection again
    if not check_ollama_connection():
        print("❌ Cannot setup RAG chain without Ollama connection")
        return None, None
    
    # Ensure the LLM model is available
    if not ensure_ollama_model(model_name):
        print(f"❌ Cannot setup RAG chain without model {model_name}")
        return None, None
    
    try:
        llm = ChatOllama(model=model_name)
        
        # Create basic retriever first
        basic_retriever = vector_db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 6}
        )

        # Wrap with MultiQueryRetriever
        retriever = MultiQueryRetriever.from_llm(
            basic_retriever,
            llm,
            prompt=PromptTemplate(
                input_variables=["question"],
                template=QUERY_PROMPT
            )
        )
        
        # Enhanced prompt template for multi-document context
        template = """You are a patient, step-by-step statistics tutor. Use the following context excerpts from multiple textbooks to answer the student's question.

{context}

When you answer, please:

1. **Restate the question.**
2. **List what is given** (data, formulas, definitions) and any assumptions.
3. **Outline your solution strategy** ("We will do X, then Y...").
4. **Work through the solution in numbered steps**, showing intermediate calculations, code snippets (in Python or R) or formula applications as needed.
5. **Cite each fact or formula** inline (e.g. "[OpenStax Ch. 7]", "[Think Stats Sec 4.2]").
6. **Summarize the final answer** in plain language at the end.
7. If the answer is not fully contained in the provided context, just say "I don't know" rather than guessing.

Question: {question}

**Answer as a tutor:**"""

        prompt = ChatPromptTemplate.from_template(template)

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
        
        print(f"✅ RAG chain setup successful with model {model_name}")
        return rag_chain, retriever
        
    except Exception as e:
        print(f"❌ Error setting up RAG chain: {e}")
        return None, None

def search_by_source(vector_db, query: str, source_filter: str = None, k: int = 3):
    """
    Search with optional source filtering.
    
    Args:
        vector_db: Vector store
        query: Search query
        source_filter: Optional source document name to filter by
        k: Number of results to return
    
    Returns:
        Retrieved documents
    """
    if source_filter:
        # Filter by source document
        docs = vector_db.similarity_search(
            query, 
            k=k,
            filter={"source_document": source_filter}
        )
    else:
        docs = vector_db.similarity_search(query, k=k)
    
    return docs

# Setup RAG chain
print("\nSetting up RAG chain...")
rag_chain, retriever = setup_rag_chain(vector_db, model)

# Check if RAG chain setup was successful
if rag_chain is None or retriever is None:
    print("\n❌ RAG chain setup failed. Cannot proceed with Q&A functionality.")
    print("The vector store was created successfully, but Ollama connection issues prevent RAG setup.")
    exit(1)

# Example usage functions
def ask_question(question: str):
    """Ask a question using the RAG system."""
    print(f"\n Question: {question}")
    response = rag_chain.invoke(question)
    print(f"\n Answer: {response}")
    return response

def search_specific_source(query: str, source: str):
    """Search within a specific source document."""
    docs = search_by_source(vector_db, query, source)
    print(f"\n Search results from '{source}' for: {query}")
    for i, doc in enumerate(docs, 1):
        page = doc.metadata.get('page', 'Unknown')
        print(f"\nResult {i} (Page {page}):")
        print(doc.page_content[:200] + "...")
    return docs

# Print available sources
print("\n Available sources in vector store:")
source_docs = set()
for chunk in all_chunks:
    source_docs.add(chunk.metadata.get('source_document', 'Unknown'))

for i, source in enumerate(sorted(source_docs), 1):
    print(f"  {i}. {source}")

print(f"\n RAG system ready! You can now:")
print("  - ask_question('Your question here')")
print("  - search_specific_source('query', 'source_name')")
print("  - Use the retriever directly for custom searches")

# Example Usage and Configuration
if __name__ == "__main__":
    print("\n" + "="*60)
    print("Multi-Doc RAG system ready.")
    print("="*60)
    
    # Example questions you can ask
    example_questions = [
        "What is the Central Limit Theorem?",
        "How do you calculate a confidence interval?",
        "What is the difference between Type I and Type II errors?",
        "Explain hypothesis testing procedures",
        "What are the conditions for using a t-distribution?"
    ]
    
    print("\n Example questions to try:")
    for i, q in enumerate(example_questions, 1):
        print(f"  {i}. {q}")


    print(f"\n🔧 To add more PDFs, modify the 'pdf_corpus' list above with:")
    print("""
    pdf_corpus.append({
        'path': '/path/to/your/pdf.pdf',
        'start_page': 0,     # Optional: start page (0-indexed)
        'end_page': 100,     # Optional: end page (inclusive)
        'name': 'Your PDF Name'
    })
    """)
    
    print("\n💡 Usage examples:")
    print("  # Ask a general question")
    print("  response = ask_question('What is a p-value?')")
    print()
    print("  # Search specific source")
    print("  results = search_specific_source('confidence interval', 'OpenIntro Statistics - Chapters 5-7')")
    print()
    print("  # Direct retrieval")
    print("  docs = retriever.invoke('your query here')")
    
    # Uncomment to test with a sample question
    # print("\n🧪 Testing with sample question...")
    # ask_question("What is the Central Limit Theorem and when does it apply?")

# Configuration for different document types
"""
EXAMPLE CONFIGURATIONS FOR DIFFERENT PDF TYPES:

# Full textbook (all chapters)
{
    'path': '/path/to/full_textbook.pdf',
    'start_page': None,
    'end_page': None,
    'name': 'Complete Statistics Textbook'
}

# Specific chapters
{
    'path': '/path/to/textbook.pdf',
    'start_page': 120,
    'end_page': 200,
    'name': 'Statistics Textbook - Inference Chapters'
}

# Multiple textbooks example:
pdf_corpus = [
    {
        'path': f"{PDF_DIRECTORY}/intro_stats.pdf",
        'start_page': 0,
        'end_page': 150,
        'name': "Intro Statistics - Descriptive Stats"
    },
    {
        'path': f"{PDF_DIRECTORY}/advanced_stats.pdf",
        'start_page': 50,
        'end_page': 200,
        'name': "Advanced Statistics - Inference"
    },
    {
        'path': f"{PDF_DIRECTORY}/probability.pdf",
        'start_page': None,
        'end_page': None,
        'name': "Probability Theory"
    }
]
"""

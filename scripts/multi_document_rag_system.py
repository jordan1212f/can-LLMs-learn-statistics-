

from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json
import logging
import ollama
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MultiDocumentRAGSystem:
    """
    A comprehensive RAG system for handling multiple educational PDFs.
    """
    
    def __init__(self, 
                 pdf_directory: str,
                 embedding_model: str = "nomic-embed-text",
                 llm_model: str = "llama3.2",
                 chunk_size: int = 1200,
                 chunk_overlap: int = 300,
                 collection_name: str = "educational_corpus"):
        """
        Initialize the Multi-Document RAG System.
        
        Args:
            pdf_directory: Base directory containing PDF files
            embedding_model: Model for creating embeddings
            llm_model: Language model for response generation
            chunk_size: Size of text chunks for processing
            chunk_overlap: Overlap between consecutive chunks
            collection_name: Name for the vector store collection
        """
        self.pdf_directory = Path(pdf_directory)
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.collection_name = collection_name
        
        # Initialize components
        self.text_splitter = None
        self.vector_db = None
        self.rag_chain = None
        self.retriever = None
        self.all_chunks = []
        self.loaded_documents = []
        
        logger.info(f"Initialized MultiDocumentRAGSystem with:")
        logger.info(f"  PDF Directory: {self.pdf_directory}")
        logger.info(f"  Embedding Model: {self.embedding_model}")
        logger.info(f"  LLM Model: {self.llm_model}")
        logger.info(f"  Chunk Size: {self.chunk_size}")
        logger.info(f"  Chunk Overlap: {self.chunk_overlap}")
    
    def load_multiple_pdfs(self, pdf_configs: List[Dict[str, Any]]) -> List:
        """
        Load and process multiple PDFs with specific page ranges.
        
        Args:
            pdf_configs: List of dictionaries with 'path', 'start_page', 'end_page', 'name'
        
        Returns:
            List of processed documents with metadata
        """
        all_documents = []
        successful_loads = 0
        
        logger.info(f"Starting to load {len(pdf_configs)} PDF documents...")
        
        for i, config in enumerate(pdf_configs, 1):
            pdf_path = config['path']
            start_page = config.get('start_page', None)
            end_page = config.get('end_page', None)
            doc_name = config.get('name', Path(pdf_path).stem)
            
            logger.info(f"\n[{i}/{len(pdf_configs)}] Processing: {doc_name}")
            logger.info(f"File: {pdf_path}")
            
            try:
                # Validate file exists
                if not Path(pdf_path).exists():
                    logger.error(f"File not found: {pdf_path}")
                    continue
                
                loader = PyPDFLoader(pdf_path)
                docs = loader.load_and_split()
                logger.info(f"Loaded {len(docs)} total pages")
                
                # Select relevant pages if specified
                if start_page is not None and end_page is not None:
                    selected_docs = docs[start_page:end_page + 1]
                    logger.info(f"Selected pages {start_page}-{end_page} ({len(selected_docs)} pages)")
                elif start_page is not None:
                    selected_docs = docs[start_page:]
                    logger.info(f"Selected pages {start_page}-end ({len(selected_docs)} pages)")
                elif end_page is not None:
                    selected_docs = docs[:end_page + 1]
                    logger.info(f"Selected pages 0-{end_page} ({len(selected_docs)} pages)")
                else:
                    selected_docs = docs
                    logger.info(f"Using all {len(selected_docs)} pages")
                
                # Add enhanced metadata
                for doc in selected_docs:
                    doc.metadata.update({
                        'source_document': doc_name,
                        'file_path': pdf_path,
                        'selected_range': f"{start_page}-{end_page}" if start_page and end_page else "all",
                        'load_timestamp': datetime.now().isoformat(),
                        'document_index': i
                    })
                
                all_documents.extend(selected_docs)
                successful_loads += 1
                logger.info(f"✅ Successfully added {len(selected_docs)} pages from {doc_name}")
                
            except Exception as e:
                logger.error(f"❌ Error processing {pdf_path}: {str(e)}")
                continue
        
        logger.info(f"\n📊 Loading Summary:")
        logger.info(f"  Total PDFs attempted: {len(pdf_configs)}")
        logger.info(f"  Successfully loaded: {successful_loads}")
        logger.info(f"  Total documents/pages: {len(all_documents)}")
        
        self.loaded_documents = all_documents
        return all_documents
    
    def create_vector_store(self, documents: List) -> Tuple[Any, List]:
        """
        Create vector store from multiple documents.
        
        Args:
            documents: List of loaded documents
        
        Returns:
            Tuple of (vector_store, chunks)
        """
        if not documents:
            raise ValueError("No documents provided for vector store creation")
        
        logger.info(f"\nCreating vector store...")
        logger.info(f"Chunk parameters: size={self.chunk_size}, overlap={self.chunk_overlap}")
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Split documents into chunks
        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
        
        # Analyze chunk distribution
        chunk_sources = {}
        for chunk in chunks:
            source = chunk.metadata.get('source_document', 'unknown')
            chunk_sources[source] = chunk_sources.get(source, 0) + 1
        
        logger.info("\n📊 Chunk distribution by source:")
        for source, count in sorted(chunk_sources.items()):
            logger.info(f"  {source}: {count} chunks")
        
        # Pull embedding model
        logger.info(f"\nPulling embedding model: {self.embedding_model}")
        try:
            ollama.pull(self.embedding_model)
            logger.info(f"✅ Embedding model {self.embedding_model} ready")
        except Exception as e:
            logger.error(f"❌ Error pulling embedding model: {e}")
            raise
        
        # Create vector store
        logger.info("Creating Chroma vector store...")
        try:
            self.vector_db = Chroma.from_documents(
                documents=chunks,
                embedding=OllamaEmbeddings(model=self.embedding_model),
                collection_name=self.collection_name,
            )
            logger.info(f"✅ Vector store created successfully")
            logger.info(f"  Collection: {self.collection_name}")
            logger.info(f"  Chunks: {len(chunks)}")
            logger.info(f"  Sources: {len(chunk_sources)}")
            
        except Exception as e:
            logger.error(f"❌ Error creating vector store: {e}")
            raise
        
        self.all_chunks = chunks
        return self.vector_db, chunks
    
    def setup_rag_chain(self) -> Tuple[Any, Any]:
        """
        Set up RAG chain for multi-document retrieval.
        
        Returns:
            Tuple of (rag_chain, retriever)
        """
        if not self.vector_db:
            raise ValueError("Vector store not created. Call create_vector_store first.")
        
        logger.info("\nSetting up RAG chain...")
        
        # Initialize LLM
        try:
            llm = ChatOllama(model=self.llm_model)
            logger.info(f"✅ LLM initialized: {self.llm_model}")
        except Exception as e:
            logger.error(f"❌ Error initializing LLM: {e}")
            raise
        
        # Create retriever with enhanced parameters
        self.retriever = self.vector_db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 8}  # Retrieve more chunks for multi-document context
        )
        
        # Enhanced prompt template for educational content
        template = """You are an expert statistics tutor with access to multiple educational textbooks. 
        Use the following context from various statistics textbooks to provide a comprehensive and accurate answer.
        
        When answering:
        1. Synthesize information from multiple sources when available
        2. Clearly indicate which textbook/source you're referencing
        3. Provide step-by-step explanations for complex concepts
        4. Include relevant formulas or examples when helpful
        5. If information is unclear or missing, state this explicitly
        
        Context from textbooks:
        {context}
        
        Student Question: {question}
        
        Comprehensive Answer:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        def format_docs_with_sources(docs):
            """Format documents with enhanced source information."""
            formatted = []
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get('source_document', 'Unknown Source')
                page = doc.metadata.get('page', 'Unknown Page')
                content = doc.page_content.strip()
                
                # Add source header
                header = f"=== Source {i}: {source} (Page {page}) ==="
                formatted.append(f"{header}\n{content}")
            
            return "\n\n".join(formatted)
        
        # Create RAG chain
        self.rag_chain = (
            {"context": self.retriever | format_docs_with_sources, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        logger.info("✅ RAG chain setup complete")
        return self.rag_chain, self.retriever
    
    def ask_question(self, question: str, verbose: bool = True) -> str:
        """
        Ask a question using the RAG system.
        
        Args:
            question: The question to ask
            verbose: Whether to print the Q&A to console
        
        Returns:
            The generated answer
        """
        if not self.rag_chain:
            raise ValueError("RAG chain not setup. Call setup_rag_chain first.")
        
        if verbose:
            logger.info(f"\n🤔 Question: {question}")
        
        try:
            response = self.rag_chain.invoke(question)
            if verbose:
                logger.info(f"\n💡 Answer:\n{response}")
            return response
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return error_msg
    
    def search_by_source(self, query: str, source_filter: str = None, k: int = 5) -> List:
        """
        Search with optional source filtering.
        
        Args:
            query: Search query
            source_filter: Optional source document name to filter by
            k: Number of results to return
        
        Returns:
            Retrieved documents
        """
        if not self.vector_db:
            raise ValueError("Vector store not created.")
        
        try:
            if source_filter:
                docs = self.vector_db.similarity_search(
                    query,
                    k=k,
                    filter={"source_document": source_filter}
                )
                logger.info(f"Found {len(docs)} results from '{source_filter}' for: {query}")
            else:
                docs = self.vector_db.similarity_search(query, k=k)
                logger.info(f"Found {len(docs)} results for: {query}")
            
            return docs
        except Exception as e:
            logger.error(f"❌ Search error: {e}")
            return []
    
    def get_available_sources(self) -> List[str]:
        """Get list of available source documents in the vector store."""
        if not self.all_chunks:
            return []
        
        sources = set()
        for chunk in self.all_chunks:
            sources.add(chunk.metadata.get('source_document', 'Unknown'))
        
        return sorted(list(sources))
    
    def display_sources(self):
        """Display available sources in a formatted way."""
        sources = self.get_available_sources()
        logger.info("\n📚 Available sources in vector store:")
        for i, source in enumerate(sources, 1):
            chunk_count = sum(1 for chunk in self.all_chunks 
                            if chunk.metadata.get('source_document') == source)
            logger.info(f"  {i}. {source} ({chunk_count} chunks)")
    
    def save_corpus_info(self, output_path: str):
        """Save information about the loaded corpus to a JSON file."""
        corpus_info = {
            'creation_date': datetime.now().isoformat(),
            'configuration': {
                'embedding_model': self.embedding_model,
                'llm_model': self.llm_model,
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap,
                'collection_name': self.collection_name
            },
            'statistics': {
                'total_documents': len(self.loaded_documents),
                'total_chunks': len(self.all_chunks),
                'sources': self.get_available_sources()
            },
            'source_distribution': {}
        }
        
        # Calculate source distribution
        for chunk in self.all_chunks:
            source = chunk.metadata.get('source_document', 'Unknown')
            corpus_info['source_distribution'][source] = corpus_info['source_distribution'].get(source, 0) + 1
        
        with open(output_path, 'w') as f:
            json.dump(corpus_info, f, indent=2)
        
        logger.info(f"📄 Corpus information saved to: {output_path}")

def main():
    """
    Main function demonstrating the Multi-Document RAG System.
    """
    # Configuration
    PDF_DIRECTORY = "/Users/jordanfernandes/Desktop/Dissertation workspace/datasets_pdfs"
    
    # Define your PDF corpus - customize this for your documents
    pdf_corpus = [
        {
            'path': f"{PDF_DIRECTORY}/os.pdf",
            'start_page': 180,
            'end_page': 313,
            'name': "OpenIntro Statistics - Chapters 5-7"
        },
        # Add more PDFs here:
        # {
        #     'path': f"{PDF_DIRECTORY}/statistics_textbook_2.pdf",
        #     'start_page': 0,
        #     'end_page': 150,
        #     'name': "Statistics Textbook 2 - Descriptive Statistics"
        # },
        # {
        #     'path': f"{PDF_DIRECTORY}/probability_theory.pdf",
        #     'start_page': None,  # Use all pages
        #     'end_page': None,
        #     'name': "Probability Theory Textbook"
        # },
        # {
        #     'path': f"{PDF_DIRECTORY}/advanced_stats.pdf",
        #     'start_page': 50,
        #     'end_page': 200,
        #     'name': "Advanced Statistics - Inference Methods"
        # }
    ]
    
    # Initialize RAG system
    print("🚀 Initializing Multi-Document RAG System...")
    rag_system = MultiDocumentRAGSystem(
        pdf_directory=PDF_DIRECTORY,
        embedding_model="nomic-embed-text",
        llm_model="llama3.2",
        chunk_size=1200,
        chunk_overlap=300,
        collection_name="educational_statistics_corpus"
    )
    
    try:
        # Load documents
        print("\n📚 Loading PDF corpus...")
        documents = rag_system.load_multiple_pdfs(pdf_corpus)
        
        if not documents:
            print("❌ No documents loaded successfully. Please check your PDF paths and try again.")
            return
        
        # Create vector store
        print("\n🔧 Creating vector store...")
        vector_db, chunks = rag_system.create_vector_store(documents)
        
        # Setup RAG chain
        print("\n⚙️ Setting up RAG chain...")
        rag_chain, retriever = rag_system.setup_rag_chain()
        
        # Display available sources
        rag_system.display_sources()
        
        # Save corpus information
        rag_system.save_corpus_info("data/corpus_info.json")
        
        print("\n" + "="*80)
        print("🎉 MULTI-DOCUMENT RAG SYSTEM READY!")
        print("="*80)
        
        # Example questions
        example_questions = [
            "What is the Central Limit Theorem and when does it apply?",
            "How do you calculate a 95% confidence interval for a population mean?",
            "What is the difference between Type I and Type II errors?",
            "Explain the conditions for using a t-distribution vs normal distribution",
            "What is a p-value and how do you interpret it?"
        ]
        
        print("\n📝 Example questions to try:")
        for i, q in enumerate(example_questions, 1):
            print(f"  {i}. {q}")
        
        print("\n💡 Usage examples:")
        print("  # Ask a question")
        print("  answer = rag_system.ask_question('What is hypothesis testing?')")
        print()
        print("  # Search specific source")
        print("  docs = rag_system.search_by_source('confidence interval', 'OpenIntro Statistics - Chapters 5-7')")
        print()
        print("  # Get available sources")
        print("  sources = rag_system.get_available_sources()")
        
        # Optional: Test with a sample question (uncomment to run)
        # print("\n🧪 Testing with sample question...")
        # rag_system.ask_question("What is the Central Limit Theorem and when does it apply?")
        
        print(f"\n✅ System ready for {len(rag_system.get_available_sources())} document sources!")
        
    except Exception as e:
        logger.error(f"❌ Error during setup: {e}")
        print(f"\n❌ Setup failed: {e}")
        print("Please check your configuration and try again.")

if __name__ == "__main__":
    main()

# Additional Configuration Examples
"""
EXAMPLE PDF CORPUS CONFIGURATIONS:

# For a comprehensive statistics course corpus:
pdf_corpus = [
    {
        'path': f"{PDF_DIRECTORY}/intro_statistics.pdf",
        'start_page': 0,
        'end_page': 200,
        'name': "Introduction to Statistics"
    },
    {
        'path': f"{PDF_DIRECTORY}/probability_theory.pdf",
        'start_page': None,
        'end_page': None,
        'name': "Probability Theory Fundamentals"
    },
    {
        'path': f"{PDF_DIRECTORY}/statistical_inference.pdf",
        'start_page': 50,
        'end_page': 300,
        'name': "Statistical Inference Methods"
    },
    {
        'path': f"{PDF_DIRECTORY}/regression_analysis.pdf",
        'start_page': 0,
        'end_page': 150,
        'name': "Regression Analysis Textbook"
    },
    {
        'path': f"{PDF_DIRECTORY}/experimental_design.pdf",
        'start_page': 25,
        'end_page': 175,
        'name': "Experimental Design and ANOVA"
    }
]

# For focused topic areas:
pdf_corpus = [
    {
        'path': f"{PDF_DIRECTORY}/hypothesis_testing.pdf",
        'start_page': None,
        'end_page': None,
        'name': "Hypothesis Testing Complete Guide"
    },
    {
        'path': f"{PDF_DIRECTORY}/confidence_intervals.pdf", 
        'start_page': None,
        'end_page': None,
        'name': "Confidence Intervals Theory and Practice"
    }
]
"""

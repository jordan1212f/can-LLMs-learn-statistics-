import fitz  # PyMuPDF
import json
import re
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
from datetime import datetime

# LangChain imports
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.schema import Document

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class QAPair:
    """Structured Q&A pair with metadata."""
    instruction: str
    response: str
    source: str
    page_number: int
    extraction_method: str
    topic: str = ""
    
@dataclass
class SemanticChunk:
    """Semantic chunk with metadata for RAG."""
    content: str
    chunk_id: str
    page_number: int
    chunk_index: int
    chapter: str
    section: str
    metadata: Dict[str, Any]

class HybridPDFExtractor:
    """
    Hybrid PDF extraction system combining direct Q&A extraction with semantic chunking.
    
    This class implements a two-pronged approach:
    1. Direct extraction: Uses regex patterns to find specific Q&A pairs
    2. Semantic chunking: Uses LangChain to create context-aware chunks for RAG
    """
    
    def __init__(self, pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the hybrid extractor.
        
        Args:
            pdf_path: Path to the PDF file
            chunk_size: Size of semantic chunks (characters)
            chunk_overlap: Overlap between chunks (characters)
        """
        self.pdf_path = pdf_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize text splitter for semantic chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Chapter boundaries (same as existing scripts)
        self.chapter_positions = {
            'chapter_1': (26, 73),
            'chapter_2': (74, 119),
            'chapter_3': (120, 165),
            'chapter_4': (166, 215),
            'chapter_5': (216, 287),
            'chapter_6': (288, 339),
            'chapter_7': (340, 413),
            'chapter_8': (414, 467),
            'appendix': (468, 544)
        }
        
        # Regex patterns for Q&A extraction
        self.example_pattern = r'EXAMPLE\s+(\d+(?:\.\d+)?)\s+(.*?)(?=EXAMPLE\s+\d+(?:\.\d+)?|$)'
        self.exercise_pattern = r'^\s*(\d+(?:\.\d+)?)\.\s+(.*?)(?=^\s*\d+(?:\.\d+)?\.|$)'
        
        logger.info(f"Initialized HybridPDFExtractor for {pdf_path}")
    
    def extract_direct_qa_pairs(self) -> Tuple[List[QAPair], List[QAPair]]:
        """
        Extract Q&A pairs using direct pattern matching (existing method).
        
        Returns:
            Tuple of (examples, exercises) as QAPair lists
        """
        logger.info("Starting direct Q&A extraction...")
        
        examples = []
        exercises = []
        
        # Open PDF with PyMuPDF
        doc = fitz.open(self.pdf_path)
        
        # Extract examples
        examples = self._extract_examples(doc)
        logger.info(f"Extracted {len(examples)} examples")
        
        # Extract exercises
        exercises = self._extract_exercises(doc)
        logger.info(f"Extracted {len(exercises)} exercises")
        
        doc.close()
        
        return examples, exercises
    
    def _extract_examples(self, doc: fitz.Document) -> List[QAPair]:
        """Extract examples using existing logic."""
        examples = []
        
        # Process each chapter
        for chapter_name, (start_page, end_page) in self.chapter_positions.items():
            if chapter_name == 'appendix':
                continue
                
            chapter_text = ""
            for page_num in range(start_page, min(end_page + 1, doc.page_count)):
                page = doc[page_num]
                page_text = page.get_text()
                chapter_text += page_text + "\n"
            
            # Find examples in chapter
            matches = re.finditer(self.example_pattern, chapter_text, re.DOTALL | re.IGNORECASE)
            
            for match in matches:
                example_num = match.group(1)
                content = match.group(2).strip()
                
                # Split into instruction and response
                instruction, response = self._split_example_content(content)
                
                if instruction and response:
                    qa_pair = QAPair(
                        instruction=f"EXAMPLE {example_num} {instruction}",
                        response=response,
                        source=self.pdf_path,
                        page_number=start_page,  # Approximate
                        extraction_method="direct_regex",
                        topic=chapter_name
                    )
                    examples.append(qa_pair)
        
        return examples
    
    def _extract_exercises(self, doc: fitz.Document) -> List[QAPair]:
        """Extract exercises using existing logic."""
        exercises = []
        
        # Process each chapter
        for chapter_name, (start_page, end_page) in self.chapter_positions.items():
            if chapter_name == 'appendix':
                continue
                
            chapter_text = ""
            for page_num in range(start_page, min(end_page + 1, doc.page_count)):
                page = doc[page_num]
                page_text = page.get_text()
                chapter_text += page_text + "\n"
            
            # Find exercises in chapter
            lines = chapter_text.split('\n')
            current_exercise = None
            current_content = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check for exercise number
                exercise_match = re.match(r'^(\d+(?:\.\d+)?)\.\s+(.*)', line)
                if exercise_match:
                    # Save previous exercise
                    if current_exercise and current_content:
                        content = ' '.join(current_content)
                        instruction, response = self._split_exercise_content(content)
                        
                        if instruction and response:
                            qa_pair = QAPair(
                                instruction=f"Exercise {current_exercise} {instruction}",
                                response=response,
                                source=self.pdf_path,
                                page_number=start_page,
                                extraction_method="direct_regex",
                                topic=chapter_name
                            )
                            exercises.append(qa_pair)
                    
                    # Start new exercise
                    current_exercise = exercise_match.group(1)
                    current_content = [exercise_match.group(2)]
                else:
                    # Continue current exercise
                    if current_exercise:
                        current_content.append(line)
        
        return exercises
    
    def _split_example_content(self, content: str) -> Tuple[str, str]:
        """Split example content into instruction and response."""
        # Look for common answer indicators
        answer_indicators = [
            r'\bSolution[:\s]',
            r'\bAnswer[:\s]',
            r'\bExplanation[:\s]',
            r'\bTo find',
            r'\bWe (can|will|need to)',
            r'\bFirst,',
            r'\bUsing'
        ]
        
        for indicator in answer_indicators:
            match = re.search(indicator, content, re.IGNORECASE)
            if match:
                split_pos = match.start()
                instruction = content[:split_pos].strip()
                response = content[split_pos:].strip()
                return instruction, response
        
        # If no clear split, use sentence-based approach
        sentences = content.split('. ')
        if len(sentences) >= 2:
            # Take first sentence as instruction, rest as response
            instruction = sentences[0] + '.'
            response = '. '.join(sentences[1:])
            return instruction, response
        
        return content, ""
    
    def _split_exercise_content(self, content: str) -> Tuple[str, str]:
        """Split exercise content into instruction and response."""
        # Similar logic to examples
        return self._split_example_content(content)
    
    def create_semantic_chunks(self) -> List[SemanticChunk]:
        """
        Create semantic chunks using LangChain's document processing.
        
        Returns:
            List of semantic chunks with metadata
        """
        logger.info("Starting semantic chunking with LangChain...")
        
        # Load document with LangChain
        loader = PyPDFLoader(self.pdf_path)
        documents = loader.load()
        
        logger.info(f"Loaded {len(documents)} pages with LangChain")
        
        # Split documents into chunks
        chunks = self.text_splitter.split_documents(documents)
        
        logger.info(f"Created {len(chunks)} semantic chunks")
        
        # Convert to SemanticChunk objects with metadata
        semantic_chunks = []
        for i, chunk in enumerate(chunks):
            # Extract page number from metadata
            page_num = chunk.metadata.get('page', 0)
            
            # Determine chapter based on page number
            chapter = self._get_chapter_from_page(page_num)
            
            # Create semantic chunk object
            semantic_chunk = SemanticChunk(
                content=chunk.page_content,
                chunk_id=f"chunk_{i:04d}",
                page_number=page_num,
                chunk_index=i,
                chapter=chapter,
                section=self._extract_section_from_content(chunk.page_content),
                metadata={
                    'source': self.pdf_path,
                    'extraction_method': 'langchain_semantic',
                    'chunk_size': len(chunk.page_content),
                    'timestamp': datetime.now().isoformat()
                }
            )
            semantic_chunks.append(semantic_chunk)
        
        return semantic_chunks
    
    def _get_chapter_from_page(self, page_num: int) -> str:
        """Determine chapter based on page number."""
        for chapter_name, (start_page, end_page) in self.chapter_positions.items():
            if start_page <= page_num <= end_page:
                return chapter_name
        return "unknown"
    
    def _extract_section_from_content(self, content: str) -> str:
        """Extract section information from content."""
        # Look for section headers
        section_pattern = r'^\s*(\d+(?:\.\d+)*)\s+([A-Z][A-Za-z\s]+)\s*$'
        lines = content.split('\n')[:5]  # Check first few lines
        
        for line in lines:
            match = re.match(section_pattern, line.strip())
            if match:
                return f"{match.group(1)} {match.group(2)}"
        
        return "unknown"
    
    def run_hybrid_extraction(self) -> Dict[str, Any]:
        """
        Run the complete hybrid extraction pipeline.
        
        Returns:
            Dictionary containing both direct Q&A pairs and semantic chunks
        """
        logger.info("Starting hybrid extraction pipeline...")
        
        # Direct Q&A extraction
        examples, exercises = self.extract_direct_qa_pairs()
        
        # Semantic chunking
        semantic_chunks = self.create_semantic_chunks()
        
        # Combine results
        results = {
            'extraction_metadata': {
                'pdf_path': self.pdf_path,
                'extraction_date': datetime.now().isoformat(),
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap,
                'total_examples': len(examples),
                'total_exercises': len(exercises),
                'total_semantic_chunks': len(semantic_chunks)
            },
            'direct_qa_pairs': {
                'examples': [asdict(ex) for ex in examples],
                'exercises': [asdict(ex) for ex in exercises]
            },
            'semantic_chunks': [asdict(chunk) for chunk in semantic_chunks]
        }
        
        logger.info("Hybrid extraction completed successfully")
        return results
    
    def save_results(self, results: Dict[str, Any], output_dir: str = "data"):
        """
        Save extraction results to files.
        
        Args:
            results: Results from hybrid extraction
            output_dir: Directory to save files
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save direct Q&A pairs (maintain compatibility)
        examples_path = output_path / "examples_qa_hybrid.jsonl"
        exercises_path = output_path / "exercises_qa_hybrid.jsonl"
        
        with open(examples_path, 'w', encoding='utf-8') as f:
            for example in results['direct_qa_pairs']['examples']:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        with open(exercises_path, 'w', encoding='utf-8') as f:
            for exercise in results['direct_qa_pairs']['exercises']:
                f.write(json.dumps(exercise, ensure_ascii=False) + '\n')
        
        # Save semantic chunks
        chunks_path = output_path / "semantic_chunks.jsonl"
        with open(chunks_path, 'w', encoding='utf-8') as f:
            for chunk in results['semantic_chunks']:
                f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
        
        # Save complete results
        complete_path = output_path / "hybrid_extraction_results.json"
        with open(complete_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {output_path}")
        logger.info(f"- Examples: {examples_path} ({len(results['direct_qa_pairs']['examples'])} items)")
        logger.info(f"- Exercises: {exercises_path} ({len(results['direct_qa_pairs']['exercises'])} items)")
        logger.info(f"- Semantic chunks: {chunks_path} ({len(results['semantic_chunks'])} items)")
        logger.info(f"- Complete results: {complete_path}")

def main():
    """Main execution function."""
    # Configuration
    PDF_PATH = "datasets_pdfs/os.pdf"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    OUTPUT_DIR = "data"
    
    # Initialize extractor
    extractor = HybridPDFExtractor(
        pdf_path=PDF_PATH,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    
    # Run hybrid extraction
    results = extractor.run_hybrid_extraction()
    
    # Save results
    extractor.save_results(results, OUTPUT_DIR)
    
    # Print summary
    print("\n" + "="*60)
    print("HYBRID EXTRACTION SUMMARY")
    print("="*60)
    print(f"PDF processed: {PDF_PATH}")
    print(f"Examples extracted: {results['extraction_metadata']['total_examples']}")
    print(f"Exercises extracted: {results['extraction_metadata']['total_exercises']}")
    print(f"Semantic chunks created: {results['extraction_metadata']['total_semantic_chunks']}")
    print(f"Chunk size: {CHUNK_SIZE} characters")
    print(f"Chunk overlap: {CHUNK_OVERLAP} characters")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*60)

if __name__ == "__main__":
    main()

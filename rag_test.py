from itertools import chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import OnlinePDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers import MultiQueryRetriever
from langchain.chains import ChatVectorDBChain
from langchain.chains import RetrievalQA
import streamlit as st
import logging
import ollama

#? what is this script for??? - this script is part of building the rag system 
#!loading pdfs starts here
MODEL_NAME = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_STORE_NAME = "simple_rag_db"


def load_and_select_pages(pdf_path: str, start_pg: int, end_pg: int):
    """loads a PDF and return only the specified Acrobat page range as Document objects."""
    loader = PyPDFLoader(file_path=pdf_path)
    pages = loader.load_and_split()
    selected = pages[start_pg-1 : end_pg]
    logging.info(f"{pdf_path}: loaded {len(pages)} pages, selected {len(selected)} pages")
    return selected

def chunk_documents(docs, chunk_size=1200, chunk_overlap=300):
    """splits a list of Documents into text chunks for vector embeddings"""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)
    logging.info(f"Split {len(docs)} docs into {len(chunks)} chunks")
    return chunks

def build_vector_store(chunks, persist_dir="simple_rag_db"):
    """Embed chunks and persist them into a Chroma vector store."""
    ollama.pull(EMBEDDING_MODEL)
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    vector_db.persist()
    logging.info(f"Persisted vector store with {len(chunks)} chunks to '{persist_dir}'")
    return vector_db

def create_retriver(vector_db, llm):
    """creates a multi-query retriver for vector db"""
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""
        You are an expert statistics tutor.  Your job is to take the student’s question and rewrite it as 3–5 concise search queries that will pull in the exact textbook passages needed on:

    • Confidence intervals  
    • Statistical inference  
    • Hypothesis testing (one‑tailed and two‑tailed)  
    • Z‑tests and T‑tests  

    Each sub‑query should be a short keyword phrase, e.g. “two‑tailed t‑test critical value formula” or “confidence interval interpretation.”  

    Example:  
    Student question: “How do I calculate and interpret a 95% confidence interval for a two‑tailed t‑test?”  
    Sub‑queries:  
    1. “95% confidence interval t‑test formula”  
    2. “interpret confidence interval results t‑test”  
    3. “two‑tailed t‑test critical t‑value”

    Now generate sub‑queries for:  
    “{question}”

    Sub‑queries:
    """
    )
    return MultiQueryRetriever.from_llm(
         vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
    )
    logging.info(f"Created MultiQueryRetriever with {len(vector_db)} documents")
    return retriever

def create_rag_chain(vector_db, retriever, llm):
    """creates a rag chain for answering user questions"""
    tutor_template = """You are a patient, step‑by‑step statistics tutor. Use only the following retrieved context excerpts to answer the student’s question. 

    Context excerpts:
    {context}

    When you answer, please:

    1. **Restate the question.**  
    2. **List what is given** (data, formulas, definitions) and any assumptions.  
    3. **Outline your solution strategy** (“We will do X, then Y…”).  
    4. **Work through the solution in numbered steps**, showing intermediate calculations, code snippets (in Python or R), or formula applications as needed.  
    5. **Cite each fact or formula** inline (e.g. “[OpenStax Ch. 7]”, “[Think Stats Sec 4.2]”).  
    6. **Summarise the final answer** in plain language at the end.  
    7. If the answer is not fully contained in the provided context, say “I don’t know” rather than guessing.

    *Question:**  
    {question}

    **Answer as a tutor:**"""

    tutor_prompt = ChatPromptTemplate.from_template(tutor_template)

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt" : tutor_prompt}
    )
    logging.info("Created RAG chain successfully for answering questions")
    return chain

def main():

    logging.basicConfig(level=logging.INFO)

    pdf_configs = [{
        'path' : 'datasets_pdfs/os.pdf',
        'start': 180,
        'end': 313,
        'name': "OpenIntro Statistics - Chapters 5-7"
    },]

    all_chunks = []
    for pdf in pdf_configs:
        pages = load_and_select_pages(pdf["path"], pdf["start"], pdf["end"])
        chunks = chunk_documents(pages)
        logging.info(f"From {pdf['name']}: produced {len(chunks)} chunks")
        all_chunks.extend(chunks)
    
    vector_db = build_vector_store(all_chunks, persist_dir="simple_rag_db")

    llm = ChatOllama(model=MODEL_NAME, temperature = 0.2) #prio on consistency for now

    retriever = create_retriver(vector_db, llm)
    rag_chain = create_rag_chain(vector_db, retriever, llm)

    print("\n📚 Welcome to your step-by-step Stats Tutor! (type 'exit' to quit)\n")
    while True:
        question = input("Your question> ").strip()
        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        qa = rag_chain.invoke({"query": question})
        answer = qa["result"]

        print("\n--- Tutor's Answer ---")
        print(answer.strip())

        # optional: show which docs were used
        docs = qa.get("source_documents", [])
        if docs:
            print("\n--- Sources Used ---")
            for doc in docs:
                src = doc.metadata.get("source", "unknown")
                snippet = doc.page_content[:200].replace("\n", " ")
                print(f"- {src}: {snippet}…")
        print("\n" + "="*40 + "\n")

if __name__ == "__main__":
    main()
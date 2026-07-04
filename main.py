# 1. IMPORTS

import os
import hashlib
from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


# 2. CONFIGURATION

PDF_PATH = os.environ.get("PDF_PATH", "python_handbook.pdf")


INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", "faiss_index")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.1-8b-instant"
TOP_K = 5
MAX_DISTANCE = 1.5

OUT_OF_SCOPE_MESSAGE = (
    "Sorry, this question is outside the knowledge available."
)

LLM_ERROR_MESSAGE = (
    "Sorry, I'm having trouble reaching the AI service right now. "
    "Please try again in a moment."
)



# 3. STARTUP VALIDATION


def validate_config():
    

    if not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to a .env file "
            "(see .env.example) or your environment before running the app."
        )

    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(
            f"PDF not found at path: {PDF_PATH}\n"
            f"Set the PDF_PATH environment variable, or place your PDF "
            f"at that location."
        )


# 4. LOAD DOCUMENTS


def load_documents(pdf_path: str):

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(
            f"PDF not found at path: {pdf_path}"
        )

    loader = PyPDFLoader(pdf_path)

    return loader.load()



# 5. SPLIT DOCUMENTS

def split_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    return splitter.split_documents(documents)



# 6. VECTOR STORE 


def _pdf_fingerprint(pdf_path: str) -> str:
    

    with open(pdf_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def create_vector_store(chunks, pdf_path: str = PDF_PATH):
    

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    fingerprint_file = os.path.join(INDEX_PATH, "source.fingerprint")
    current_fingerprint = _pdf_fingerprint(pdf_path)

    if os.path.exists(INDEX_PATH) and os.path.exists(fingerprint_file):

        with open(fingerprint_file, "r") as f:
            saved_fingerprint = f.read().strip()

        if saved_fingerprint == current_fingerprint:
            try:
                return FAISS.load_local(
                    INDEX_PATH,
                    embeddings,
                    allow_dangerous_deserialization=True
                )
            except Exception as e:
                print(f"Failed to load cached index ({e}), rebuilding...")
        else:
            print("Source PDF has changed since the cache was built — rebuilding index...")

   

    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    os.makedirs(INDEX_PATH, exist_ok=True)

    vector_store.save_local(INDEX_PATH)

    with open(fingerprint_file, "w") as f:
        f.write(current_fingerprint)

    return vector_store



# 7. CREATE RAG COMPONENTS

def create_rag_components():

    llm = ChatGroq(
        groq_api_key=os.environ.get(
            "GROQ_API_KEY"
        ),
        model_name=GROQ_MODEL,
        temperature=0.2
    )

    prompt = ChatPromptTemplate.from_template(
        """
You are an expert educational tutor.

IMPORTANT RULES:

1. Use the provided context as the source of truth.
2. Explain concepts in a detailed educational manner.
3. Provide examples whenever appropriate.
4. Do not invent facts not supported by the context.
5. If the answer cannot be determined from the context, respond exactly:

I cannot answer this because the information is not present in the provided document.

Student Level: {student_level}

Weak Area: {weak_area}

Context:
{context}

Question:
{question}
"""
    )

    return llm, prompt



# 8. ASK QUESTION 


def ask_question(
    llm,
    prompt,
    vector_store,
    question: str,
    student_level: str,
    weak_area: str
):

    

    try:
        results = vector_store.similarity_search_with_score(
            question,
            k=TOP_K
        )
    except Exception as e:
        print(f"Retrieval failed: {e}")
        return LLM_ERROR_MESSAGE

    if not results:

        return (
            "Sorry, no relevant information was found "
            "in the uploaded document."
        )

    

    print("\nRetrieved Chunks:")

    for i, (doc, distance) in enumerate(
        results,
        start=1
    ):
        print(
            f"Chunk {i}: Distance = {distance:.4f}"
        )

    
    best_distance = results[0][1]

    print(
        f"\nBest Distance: {best_distance:.4f}"
    )

    if best_distance > MAX_DISTANCE:

        return OUT_OF_SCOPE_MESSAGE

    
    docs = [
        doc
        for doc, distance in results
    ]

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    

    coverage_prompt = f"""
You are a retrieval validator.

Determine whether the context contains enough
information to answer the user's question.

Respond ONLY with:

YES

or

NO

Context:
{context}

Question:
{question}
"""

    try:
        coverage_response = (
            llm.invoke(
                coverage_prompt
            )
            .content
            .strip()
            .upper()
        )
    except Exception as e:
        print(f"Coverage check LLM call failed: {e}")
        return LLM_ERROR_MESSAGE

    print(
        f"Coverage Check: {coverage_response}"
    )

    if coverage_response != "YES":

        return OUT_OF_SCOPE_MESSAGE

    # STEP 6: Generate Final Answer

    formatted_prompt = prompt.format(
        context=context,
        question=question,
        student_level=student_level,
        weak_area=weak_area
    )

    try:
        response = llm.invoke(
            formatted_prompt
        )
    except Exception as e:
        print(f"Generation LLM call failed: {e}")
        return LLM_ERROR_MESSAGE

    return response.content
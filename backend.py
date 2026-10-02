"""Document ingestion and question-answering helpers for the meeting chat app."""

from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Keep all data beside this Python file, regardless of the folder used to start Streamlit.
BASE_DIR = Path(__file__).resolve().parent
# Uploaded PDFs are kept here. FAISS writes its two index files in the second folder.
PDF_FOLDER = BASE_DIR / "doc"
FAISS_INDEX_DIR = BASE_DIR / "faiss_meeting_index"
# This model turns text into searchable number vectors and supports several languages.
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# The interface intentionally uses one model so users only need to provide a token.
DEFAULT_MODEL_ID = "deepseek-ai/DeepSeek-V3"

# Streamlit reruns this file often. Keeping one model in memory avoids loading it repeatedly.
_embeddings: HuggingFaceEmbeddings | None = None


class DocumentIndexError(RuntimeError):
    """Raised when the meeting PDFs or the persisted index cannot be used.

    The Streamlit app catches this error and shows the message to the user instead of
    exposing a long technical traceback.
    """


class ModelConfigurationError(RuntimeError):
    """Raised when the text-generation model has not been configured correctly.

    It covers missing tokens, token formatting, and model-wrapper setup problems.
    """


def get_embeddings() -> HuggingFaceEmbeddings:
    """Return one shared multilingual embedding model for this process."""
    # ``global`` lets every function in this running Python process share one model object.
    # This matters because loading a transformer model is much slower than using one already in memory.
    global _embeddings
    if _embeddings is None:
        try:
            # The first run may download the model. Later runs reuse the Hugging Face cache.
            _embeddings = HuggingFaceEmbeddings(
                model=EMBEDDING_MODEL,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        except Exception as exc:
            raise DocumentIndexError(
                "Could not load the embedding model. Check your internet connection on "
                "the first run, then restart the app after the model download completes."
            ) from exc
    return _embeddings


def list_pdf_files(pdf_folder: Path | str = PDF_FOLDER) -> list[str]:
    """Return PDF names in a document directory, sorted for stable display."""
    folder = Path(pdf_folder)
    if not folder.is_dir():
        raise DocumentIndexError(f"PDF folder not found: {folder}")
    # Only use regular PDF files. Sorting keeps the displayed order predictable.
    return sorted(path.name for path in folder.iterdir() if path.is_file() and path.suffix.lower() == ".pdf")


def save_uploaded_documents(uploaded_files) -> list[str]:
    """Save PDF files uploaded through Streamlit into the document directory."""
    if not uploaded_files:
        return []

    # Create the storage folder on the first upload if it does not already exist.
    PDF_FOLDER.mkdir(parents=True, exist_ok=True)
    saved_files: list[str] = []
    for uploaded_file in uploaded_files:
        # ``.name`` removes any folder path supplied by a browser, so uploads stay in doc/.
        filename = Path(uploaded_file.name).name
        if not filename or Path(filename).suffix.lower() != ".pdf":
            raise DocumentIndexError("Only PDF documents can be uploaded.")
        try:
            # A same-named upload replaces the older local file before the index is rebuilt.
            (PDF_FOLDER / filename).write_bytes(uploaded_file.getvalue())
        except Exception as exc:
            raise DocumentIndexError(f"Could not save '{filename}': {exc}") from exc
        saved_files.append(filename)
    # The caller rebuilds the FAISS index after this function succeeds.
    return saved_files


def index_exists() -> bool:
    """Return whether both files needed to load a FAISS index are present."""
    # Both files are required: vectors live in .faiss and document metadata lives in .pkl.
    return (FAISS_INDEX_DIR / "index.faiss").is_file() and (FAISS_INDEX_DIR / "index.pkl").is_file()


def ingest_documents(pdf_folder: Path | str | None = None) -> dict[str, int | list[str]]:
    """Read PDFs, split them into chunks, and persist a FAISS index."""
    # Use an optional folder only for programmatic use; the app normally uses doc/.
    folder = Path(pdf_folder) if pdf_folder is not None else PDF_FOLDER
    # Fail before loading the embedding model if there is nothing useful to process.
    pdf_files = list_pdf_files(folder)
    if not pdf_files:
        raise DocumentIndexError(f"No PDF files found in {folder}")

    try:
        # Each loaded document includes extracted text plus source and page metadata.
        documents = PyPDFDirectoryLoader(str(folder)).load()
    except Exception as exc:
        raise DocumentIndexError(f"Could not read the PDFs in {folder}: {exc}") from exc

    if not documents:
        raise DocumentIndexError("The PDFs did not contain any readable pages.")

    # Smaller overlapping pieces make it easier to retrieve one relevant part of a long meeting.
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(documents)
    if not chunks:
        raise DocumentIndexError("No text could be extracted from the PDFs.")

    try:
        # FAISS stores the embedding vector alongside the original text and page metadata.
        vectorstore = FAISS.from_documents(chunks, get_embeddings())
        vectorstore.save_local(str(FAISS_INDEX_DIR))
    except Exception as exc:
        raise DocumentIndexError(f"Could not create the document index: {exc}") from exc

    # These values are returned to the UI so it can confirm what was indexed.
    return {"total_pages": len(documents), "total_chunks": len(chunks), "pdf_files": pdf_files}


def load_vectorstore() -> FAISS:
    """Load the persisted FAISS index using the matching embedding model."""
    # Checking first produces a friendly message instead of a low-level missing-file error.
    if not index_exists():
        raise DocumentIndexError(
            "No complete document index was found. Process the meeting documents first."
        )
    try:
        # LangChain saves metadata with pickle. This is safe only for indexes created by this app.
        return FAISS.load_local(
            str(FAISS_INDEX_DIR), get_embeddings(), allow_dangerous_deserialization=True
        )
    except Exception as exc:
        raise DocumentIndexError(
            "Could not load the saved document index. Process the PDFs again to rebuild it."
        ) from exc


def get_llm(hf_token: str, model_id: str = DEFAULT_MODEL_ID) -> ChatHuggingFace:
    """Create a Hugging Face chat model without storing the token globally."""
    # Clean user input before validating it. The token is not written to disk or environment variables.
    token = hf_token.strip()
    selected_model = model_id.strip()
    # Catch common input mistakes before the app tries a network request.
    if not token:
        raise ModelConfigurationError("Enter a Hugging Face API token before connecting the model.")
    if not token.startswith("hf_"):
        raise ModelConfigurationError("The Hugging Face API token should begin with 'hf_'.")
    if not selected_model:
        raise ModelConfigurationError("Enter a Hugging Face model ID.")

    try:
        # Creating the wrapper does not answer a question yet; the request happens in query().
        endpoint = HuggingFaceEndpoint(
            repo_id=selected_model,
            huggingfacehub_api_token=token,
            max_new_tokens=512,
            temperature=0.2,
            timeout=120,
        )
        return ChatHuggingFace(llm=endpoint)
    except Exception as exc:
        raise ModelConfigurationError(
            f"Could not configure '{selected_model}'. Check that the model supports Hugging Face Inference."
        ) from exc


def format_docs(documents) -> str:
    """Format retrieved documents with the source data the model must cite."""
    formatted: list[str] = []
    for document in documents:
        # PyPDF page numbers start at zero, while people expect page one to be the first page.
        source = Path(str(document.metadata.get("source", "Unknown"))).name
        page = document.metadata.get("page")
        page_label = str(page + 1) if isinstance(page, int) else "Unknown"
        # The label goes into the prompt with the text, giving the model exact citation values.
        formatted.append(f"[Source: {source}, Page: {page_label}]\n{document.page_content}")
    return "\n\n---\n\n".join(formatted)


# The prompt makes the model use retrieved text and gives it a consistent citation rule.
MEETING_PROMPT = ChatPromptTemplate.from_template(
    """You are a Meeting Document Assistant. Answer using only the supplied context.

Every factual statement must cite its source in this exact form:
(Source: <document_name>, Page: <page_number>)

If the context does not answer the question, respond exactly with:
I could not find information about this in the meeting documents.

CONTEXT:
{context}

QUESTION:
{question}
"""
)


def build_rag_chain(vectorstore: FAISS, chat_model: ChatHuggingFace):
    """Build the retrieval, prompting, and response chain."""
    if vectorstore is None:
        raise DocumentIndexError("Load or process a document index before asking a question.")
    # Find the six nearest document sections for each question.
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    # The pipe reads left to right: search -> format context -> prompt -> model -> plain text.
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | MEETING_PROMPT
        | chat_model
        | StrOutputParser()
    )


def query(chain, question: str) -> str:
    """Ask one non-empty question and return the model response."""
    # Remove accidental spaces so a blank-looking message is not sent to the model.
    cleaned_question = question.strip()
    if not cleaned_question:
        raise ValueError("Enter a question before submitting.")
    if chain is None:
        raise ModelConfigurationError("Load an index and connect a model before asking a question.")
    # This triggers both retrieval and the remote Hugging Face generation request.
    return chain.invoke(cleaned_question)

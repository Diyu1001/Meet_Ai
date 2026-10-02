"""Streamlit interface for searching meeting transcripts."""

from __future__ import annotations  # Allows modern type hints to be read safely by Python.

from hashlib import sha256  # Creates a non-reversible fingerprint for comparing tokens.

import streamlit as st  # Provides the browser interface and session state.

# The backend keeps all non-UI work in one place: files, search, and the AI model.
from backend import (
    DEFAULT_MODEL_ID,
    DocumentIndexError,
    ModelConfigurationError,
    build_rag_chain,
    get_llm,
    index_exists,
    ingest_documents,
    load_vectorstore,
    query,
    save_uploaded_documents,
)


# This must be the first Streamlit command because it configures the browser page.
st.set_page_config(page_title="Meeting Chat Assistant", page_icon="M", layout="wide")  # Set browser title, icon, and page width.


def load_index() -> bool:
    """Load the local index once per browser session."""
    try:
        # The FAISS files contain only stored search data. Loading also creates the embedding
        # model used to turn a future question into comparable search numbers.
        # This happens only after an upload or when the user first asks a question.
        st.session_state.vectorstore = load_vectorstore()  # Keep the loaded index available for later questions.
        return True  # Tell the caller that the index is ready.
    except DocumentIndexError as exc:
        st.error(str(exc))  # Show the friendly backend error in the web page.
        return False  # Stop model setup because searching cannot work without an index.


def connect_model(token: str) -> bool:
    """Connect the default model to the currently available document index."""
    # A model can answer only after the local document search index is ready.
    # If the app was restarted, this loads the existing index before making the chain.
    if st.session_state.vectorstore is None and not load_index():  # Load saved document data when needed.
        return False  # Do not continue if PDF indexing or loading failed.
    try:
        # The chain is the reusable route for one question:
        # question -> find relevant PDF text -> build prompt -> call the AI model -> answer.
        st.session_state.rag_chain = build_rag_chain(
            st.session_state.vectorstore,  # Supplies the local PDF search tool.
            get_llm(token, DEFAULT_MODEL_ID),  # Creates the Hugging Face chat model with the entered token.
        )
        # Store only a fingerprint so the app can notice a token change without retaining a second copy.
        st.session_state.connected_token = sha256(token.encode()).hexdigest()
        return True  # Tell the chat handler that it can now submit the question.
    except (DocumentIndexError, ModelConfigurationError) as exc:
        st.error(str(exc))  # Display expected setup failures, such as an invalid token.
        return False
    except Exception as exc:
        st.error(f"Could not connect the model: {exc}")  # Preserve unexpected details for troubleshooting.
        return False


# Session state survives Streamlit's normal page reruns for this browser session.
# Without it, uploading a file or sending a chat message would erase the conversation.
for key, value in {
    "chat_history": [],       # Messages shown above the current chat input.
    "rag_chain": None,        # Retrieval and AI pipeline created after the token is accepted.
    "vectorstore": None,      # Loaded FAISS search index for the current browser session.
    "index_info": None,       # PDF, page, and chunk totals shown after a new upload.
    "upload_signature": None, # Stops the same Streamlit upload event being indexed repeatedly.
    "connected_token": None,  # SHA-256 fingerprint only; the token itself is not copied here.
}.items():
    if key not in st.session_state:  # Do not overwrite state created during an earlier page rerun.
        st.session_state[key] = value  # Add the missing default value only once per session.


st.title("Meeting Chat Assistant")  # Main heading shown above the chat history.
# This short line tells a new user the entire workflow without adding more controls.
st.caption("Upload meeting PDFs, add your Hugging Face token, and ask a question.")

with st.sidebar:
    st.header("Setup")  # Groups the only two user setup fields in the sidebar.
    # Streamlit masks the token in the browser. It remains in the current browser session only.
    hf_token = st.text_input("Hugging Face API token", type="password")
    # This is the only document control. Uploading starts indexing automatically below.
    uploaded_documents = st.file_uploader(
        "Add meeting documents",  # Text shown beside the file selection control.
        type=["pdf"],  # Browser limits the picker to PDF file extensions.
        accept_multiple_files=True,  # A user can select several meeting documents together.
        help="Uploaded PDFs are saved in the app's document folder and indexed automatically.",
    )

if hf_token:
    # A changed token may have different permissions, so discard a chain made with the old token.
    current_token = sha256(hf_token.encode()).hexdigest()  # Hash the current field value without displaying it.
    if st.session_state.connected_token and st.session_state.connected_token != current_token:  # Compare old and new fingerprints.
        # Do not keep using a model connection created with an older token.
        st.session_state.rag_chain = None  # Force the next question to make a model connection with the new token.
        st.session_state.connected_token = None  # Remove the now outdated fingerprint.

if uploaded_documents:
    # Streamlit reruns after every interaction. The signature prevents indexing the same upload again.
    upload_signature = tuple((document.name, document.size) for document in uploaded_documents)  # Compact identity for this selection.
    if upload_signature != st.session_state.upload_signature:
        with st.spinner("Saving and indexing meeting documents..."):  # Show progress because PDF processing can take time.
            try:
                # Save first, then rebuild the index from every PDF currently stored in doc/.
                # Rebuilding makes old and new meeting files searchable together.
                save_uploaded_documents(uploaded_documents)  # Copy browser-uploaded file bytes into doc/.
                st.session_state.index_info = ingest_documents()  # Extract text, create embeddings, and save the FAISS index.
                st.session_state.vectorstore = load_vectorstore()  # Open the newly saved index for immediate use.
                st.session_state.rag_chain = None  # A new index requires a chain built with the new search data.
                st.session_state.connected_token = None  # The next question will reconnect the chain automatically.
                st.session_state.upload_signature = upload_signature  # Mark this exact upload as handled.
                st.success("Documents are ready to search.")  # Confirm that chat can use the uploaded PDFs.
            except DocumentIndexError as exc:
                st.error(str(exc))  # Explain unreadable files, missing text, or indexing failures.

if st.session_state.index_info:
    info = st.session_state.index_info  # Short alias makes the display line easier to read.
    st.info(f"Ready: {len(info['pdf_files'])} PDFs and {info['total_pages']} pages indexed.")  # Show indexing result.
elif not index_exists():
    # This is shown only when there is no prior index and the user has not uploaded a PDF yet.
    st.info("Add at least one meeting PDF in the sidebar to begin.")

for message in st.session_state.chat_history:
    # Recreate earlier messages after a Streamlit rerun.
    with st.chat_message(message["role"]):  # Use the stored role to choose the user or assistant message style.
        st.markdown(message["content"])  # Render saved Markdown text inside that chat bubble.

user_input = st.chat_input("Ask about your meetings...")  # Returns text only after the user submits a message.
if user_input:
    # Do not send a question to an external model until the user has supplied a token.
    if not hf_token.strip():
        st.warning("Add your Hugging Face API token in the sidebar before asking a question.")  # Prevent an unauthenticated API request.
    elif st.session_state.rag_chain is None:
        # Connection is automatic, so the user does not need a separate button.
        connect_model(hf_token)  # Load the index and prepare the question-answering chain on demand.

    if st.session_state.rag_chain is not None:
        # Keep both sides of the conversation so they remain visible during this session.
        st.session_state.chat_history.append({"role": "user", "content": user_input})  # Save the submitted question first.
        with st.chat_message("user"):  # Show the user's new message immediately.
            st.markdown(user_input)  # Render the question as Markdown text.
        with st.chat_message("assistant"):  # Reserve the next chat bubble for the generated answer.
            with st.spinner("Searching meeting documents..."):  # Keep the UI responsive while retrieval and generation run.
                try:
                    # query() searches first; it does not send the entire document collection to the model.
                    answer = query(st.session_state.rag_chain, user_input)  # Retrieve matching chunks and ask the AI model.
                except Exception as exc:
                    answer = f"Unable to generate an answer: {exc}"  # Turn an API or network failure into visible feedback.
            st.markdown(answer)  # Display the answer or the error message in the assistant bubble.
        st.session_state.chat_history.append({"role": "assistant", "content": answer})  # Keep the answer for later reruns.

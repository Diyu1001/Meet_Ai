# Meeting Chat Assistant

Meeting Chat Assistant is a small web application for asking questions about meeting transcripts stored as PDF files. Upload PDFs, enter a Hugging Face API token, and ask questions in plain language. The app finds relevant parts of the meeting documents first, then asks an AI model to write an answer with page citations.

## What the application does

The application uses retrieval-augmented generation (RAG). This keeps answers tied to the uploaded documents instead of relying only on the AI model's general knowledge.

1. You upload one or more meeting PDFs.
2. The app saves the PDFs in the local `doc/` folder.
3. It reads their text, divides the text into small overlapping sections, and converts each section into numbers called embeddings.
4. It saves those embeddings in a local FAISS search index.
5. When you ask a question, the app finds the six most relevant sections.
6. It sends those sections and the question to the configured Hugging Face model.
7. The model returns an answer that cites the document name and page number supplied in the retrieved sections.

## Features

- Upload multiple PDF meeting documents from the browser.
- Automatic document indexing after upload.
- Persistent local index, so documents do not need to be processed again after every restart.
- Multilingual document search using `paraphrase-multilingual-MiniLM-L12-v2`.
- Source and page citations in generated answers.
- Chat history for the current browser session.
- Token is entered at runtime and is not stored in the source code.

## Requirements

- Python 3.10 or newer.
- An internet connection the first time the embedding model is downloaded.
- A Hugging Face account and an API token beginning with `hf_`.
- A Hugging Face Inference-supported model. The app currently uses `deepseek-ai/DeepSeek-V3` as its configured default.

## Project layout

```text
test/
|-- app.py                  Streamlit user interface
|-- backend.py              PDF, FAISS, embedding, and model helpers
|-- requirements.txt        Python packages required by the app
|-- README.md               This guide
|-- doc/                    Uploaded PDF files are saved here
`-- faiss_meeting_index/    Generated local search index
    |-- index.faiss         Vector data used for similarity search
    `-- index.pkl           Document text and metadata used with the vectors
```

`doc/` and `faiss_meeting_index/` contain application data. The index is regenerated whenever new documents are uploaded.

## Installation

Open PowerShell in the project folder and install the required packages:

```powershell
cd C:\Users\Admin\Downloads\test
python -m pip install -r requirements.txt
```

Using a virtual environment is recommended when setting up the project on a new computer:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Start the application

Run the following command in the project folder:

```powershell
streamlit run app.py
```

Streamlit normally opens the application in a browser. If it does not, open the local address printed in PowerShell, usually `http://localhost:8501`.

To stop the server, return to the PowerShell window and press `Ctrl+C`.

## How to use it

1. Open the application in the browser.
2. In the sidebar, paste your Hugging Face API token into **Hugging Face API token**.
3. Under **Add meeting documents**, select one or more PDF files.
4. Wait for the message confirming that the documents are ready to search.
5. Type a question into the chat box and press Enter.

There are only two setup controls in the sidebar: the API token field and the PDF uploader. Saving documents, creating the index, loading the index, and connecting the model happen automatically when they are needed.

### Example questions

- What decisions were made in the most recent meeting?
- List all action items and the people responsible for them.
- What deadlines were discussed?
- Summarize the meeting from January 14.
- What follow-up work was assigned to the finance team?

## Getting a Hugging Face token

1. Sign in to [Hugging Face](https://huggingface.co/).
2. Open the [Access Tokens page](https://huggingface.co/settings/tokens).
3. Create a token with the permissions required by your chosen inference provider.
4. Copy the token and paste it into the application sidebar.

The token is used only while the Streamlit session is running. It is not written into `app.py`, `backend.py`, or this README.

## How the components work

| Component | Purpose |
|---|---|
| Streamlit | Displays the browser interface, upload field, chat history, and chat input. |
| PyPDF | Extracts text and page information from uploaded PDFs. |
| RecursiveCharacterTextSplitter | Breaks long document text into overlapping sections for better retrieval. |
| Sentence Transformers | Creates multilingual embeddings for text sections and questions. |
| FAISS | Searches the saved embeddings for sections related to a question. |
| LangChain | Connects retrieval, prompt creation, Hugging Face inference, and text output. |
| Hugging Face Inference | Generates the final natural-language answer. |

## Document storage and re-indexing

Each uploaded PDF is copied to `doc/`. Uploading a file with the same name replaces the file already stored there. After upload, the application rebuilds the FAISS index using every PDF in `doc/`.

The index is stored in `faiss_meeting_index/`. Keep this folder if you want the app to reuse the existing index after a restart. If it becomes corrupted or the document set changes outside the application, upload a PDF again to create a fresh index.

## Privacy and security

- Treat the `doc/` folder and `faiss_meeting_index/` folder as sensitive if the meeting transcripts are confidential.
- Do not commit API tokens, private PDFs, or local FAISS index files to a public Git repository.
- The application sends the retrieved document sections and your question to the selected Hugging Face inference service in order to generate an answer.
- Review your organization’s data-handling rules before uploading confidential documents or using a third-party inference provider.
- If a token was previously exposed, revoke it from Hugging Face and create a replacement.

## Troubleshooting

### “Add your Hugging Face API token”

Paste a valid token in the sidebar. Hugging Face tokens begin with `hf_`.

### “No complete document index was found”

Upload at least one PDF through the sidebar. The app will create the index automatically.

### “Could not load the embedding model”

The embedding model is downloaded the first time documents are indexed. Check that the computer can access Hugging Face, then restart the app and upload the documents again. Later runs use the cached model.

### The answer says it cannot find information

The search results did not contain enough relevant text to answer the question. Try using more specific words from the meeting, confirm that the relevant PDF was uploaded, or upload a clearer text-based PDF.

### The PDF does not produce useful answers

The PDF may be scanned images rather than selectable text. This application extracts embedded PDF text; scanned documents need OCR before they can be searched reliably.

### The model cannot generate an answer

Check that the token is active, the computer has internet access, and the default model is available through your Hugging Face Inference account. The error message displayed in the chat will include the underlying cause when available.

### Package installation fails

Upgrade pip and try again:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you are using an existing Python environment with unrelated package conflicts, create and use the virtual environment shown in the installation section.

## Developer notes

- `app.py` owns the Streamlit interface and its session state.
- `backend.py` owns file handling, PDF processing, embeddings, FAISS storage, the Hugging Face model wrapper, and the RAG chain.
- `get_embeddings()` caches the embedding model in memory so a Streamlit session does not load it repeatedly.
- `format_docs()` places the source filename and page number beside each retrieved section. The prompt tells the model to use those values in its citations.
- `allow_dangerous_deserialization=True` is required by LangChain to load FAISS metadata saved with Python pickle. Only load index files created by this trusted application.

## License

No license file is currently included. Add a license before redistributing the project.

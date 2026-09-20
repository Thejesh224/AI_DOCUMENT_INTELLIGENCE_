import os
import re
import json
import uuid
import hashlib
from datetime import datetime
from zoneinfo import ZoneInfo
from io import BytesIO

import streamlit as st
import pandas as pd
import requests

from bs4 import BeautifulSoup
from pypdf import PdfReader
from docx import Document as DocxDocument

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from google import genai
from google.genai import types


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Document Intelligence",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_NAME = "AI Document Intelligence"

GEMINI_MODEL = "gemini-2.5-flash-lite"

MAX_OUTPUT_TOKENS = 4096

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150
RETRIEVER_K = 4

USERS_FILE = "users.json"
CHATS_FILE = "chats.json"


# ============================================================
# SESSION STATE
# ============================================================

def initialize_session_state():

    defaults = {
        "logged_in": False,
        "username": "",
        "current_chat_id": None,
        "documents": [],
        "images": [],
        "vectorstore": None,
        "vectorstore_signature": None,
        "website_url": "",
        "website_loaded": False,
    }

    for key, value in defaults.items():

        if key not in st.session_state:
            st.session_state[key] = value


initialize_session_state()


# ============================================================
# JSON FILE FUNCTIONS
# ============================================================

def ensure_json_file(path, default_value):

    if not os.path.exists(path):

        try:

            with open(
                path,
                "w",
                encoding="utf-8"
            ) as file:

                json.dump(
                    default_value,
                    file,
                    indent=2
                )

        except Exception:
            pass


def load_json(path, default_value):

    ensure_json_file(
        path,
        default_value
    )

    try:

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    except Exception:

        return default_value


def save_json(path, data):

    try:

        temp_path = path + ".tmp"

        with open(
            temp_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                data,
                file,
                indent=2,
                ensure_ascii=False
            )

        os.replace(
            temp_path,
            path
        )

        return True

    except Exception:

        return False


ensure_json_file(
    USERS_FILE,
    {}
)

ensure_json_file(
    CHATS_FILE,
    {}
)


# ============================================================
# PASSWORD
# ============================================================

def hash_password(password):

    return hashlib.sha256(
        password.encode("utf-8")
    ).hexdigest()


# ============================================================
# IST GREETING
# ============================================================

def get_time_greeting():

    ist_now = datetime.now(
        ZoneInfo("Asia/Kolkata")
    )

    hour = ist_now.hour

    if 5 <= hour < 12:

        return "🌅 Good Morning"

    elif 12 <= hour < 17:

        return "☀️ Good Afternoon"

    elif 17 <= hour < 21:

        return "🌇 Good Evening"

    else:

        return "🌙 Good Night"


# ============================================================
# USER REGISTRATION
# ============================================================

def register_user(
    username,
    password
):

    username = username.strip()

    if not username:

        return (
            False,
            "Please enter a username."
        )

    if not password:

        return (
            False,
            "Please enter a password."
        )

    users = load_json(
        USERS_FILE,
        {}
    )

    if username in users:

        return (
            False,
            "Username already exists."
        )

    users[username] = {

        "password": hash_password(
            password
        ),

        "created_at":
            datetime.now().isoformat(),
    }

    if save_json(
        USERS_FILE,
        users
    ):

        return (
            True,
            "Account created successfully."
        )

    return (
        False,
        "Could not save account."
    )


# ============================================================
# USER LOGIN
# ============================================================

def authenticate_user(
    username,
    password
):

    users = load_json(
        USERS_FILE,
        {}
    )

    if username not in users:

        return False

    stored_password = users[
        username
    ].get(
        "password",
        ""
    )

    return (
        stored_password
        == hash_password(password)
    )


# ============================================================
# LOGIN PAGE
# ============================================================

def show_login():

    st.title(
        "🤖 AI Document Intelligence"
    )

    st.caption(
        "Chat with your documents, images and websites using AI."
    )

    st.divider()

    login_tab, signup_tab = st.tabs(
        [
            "🔐 Login",
            "📝 Create Account"
        ]
    )

    with login_tab:

        username = st.text_input(
            "Username",
            key="login_username"
        )

        password = st.text_input(
            "Password",
            type="password",
            key="login_password"
        )

        if st.button(
            "Login",
            type="primary",
            use_container_width=True
        ):

            if authenticate_user(
                username,
                password
            ):

                st.session_state.logged_in = True

                st.session_state.username = (
                    username
                )

                st.session_state.current_chat_id = None

                st.success(
                    "Login successful."
                )

                st.rerun()

            else:

                st.error(
                    "Invalid username or password."
                )

    with signup_tab:

        new_username = st.text_input(
            "Choose username",
            key="signup_username"
        )

        new_password = st.text_input(
            "Choose password",
            type="password",
            key="signup_password"
        )

        confirm_password = st.text_input(
            "Confirm password",
            type="password",
            key="signup_confirm_password"
        )

        if st.button(
            "Create Account",
            use_container_width=True
        ):

            if new_password != confirm_password:

                st.error(
                    "Passwords do not match."
                )

            else:

                success, message = register_user(
                    new_username,
                    new_password
                )

                if success:

                    st.success(message)

                else:

                    st.error(message)


# ============================================================
# CHAT FUNCTIONS
# ============================================================

def get_user_chats(username):

    chats = load_json(
        CHATS_FILE,
        {}
    )

    user_chats = []

    for chat_id, chat_data in chats.items():

        if chat_data.get(
            "username"
        ) == username:

            user_chats.append(
                {
                    "id": chat_id,

                    "title":
                        chat_data.get(
                            "title",
                            "New Chat"
                        ),

                    "updated_at":
                        chat_data.get(
                            "updated_at",
                            ""
                        ),
                }
            )

    user_chats.sort(
        key=lambda x:
            x.get(
                "updated_at",
                ""
            ),
        reverse=True
    )

    return user_chats


def create_chat(username):

    chats = load_json(
        CHATS_FILE,
        {}
    )

    chat_id = str(
        uuid.uuid4()
    )

    now = datetime.now().isoformat()

    chats[chat_id] = {

        "username": username,

        "title": "New Chat",

        "created_at": now,

        "updated_at": now,

        "messages": [],
    }

    save_json(
        CHATS_FILE,
        chats
    )

    return chat_id


def get_chat(chat_id):

    chats = load_json(
        CHATS_FILE,
        {}
    )

    return chats.get(
        chat_id
    )


def save_chat(
    chat_id,
    chat_data
):

    chats = load_json(
        CHATS_FILE,
        {}
    )

    chats[chat_id] = chat_data

    save_json(
        CHATS_FILE,
        chats
    )


def delete_chat(chat_id):

    chats = load_json(
        CHATS_FILE,
        {}
    )

    if chat_id in chats:

        del chats[chat_id]

    save_json(
        CHATS_FILE,
        chats
    )


def make_chat_title(question):

    question = question.strip()

    if not question:

        return "New Chat"

    clean_question = re.sub(
        r"\s+",
        " ",
        question
    )

    if len(clean_question) > 45:

        clean_question = (
            clean_question[:45].rstrip()
            + "..."
        )

    return clean_question


# ============================================================
# TEXT SPLITTER
# ============================================================

def split_text(
    text,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP
):

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    if not text:

        return []

    chunks = []

    start = 0

    text_length = len(text)

    while start < text_length:

        end = min(
            start + chunk_size,
            text_length
        )

        chunk = text[
            start:end
        ].strip()

        if chunk:

            chunks.append(
                chunk
            )

        if end >= text_length:

            break

        start = max(
            end - overlap,
            start + 1
        )

    return chunks


# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_pdf(
    file_bytes,
    filename
):

    documents = []

    try:

        reader = PdfReader(
            BytesIO(file_bytes)
        )

        for page_number, page in enumerate(
            reader.pages,
            start=1
        ):

            text = (
                page.extract_text()
                or ""
            )

            if text.strip():

                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source":
                                filename,

                            "page":
                                page_number,

                            "type":
                                "PDF",
                        },
                    )
                )

    except Exception as error:

        raise RuntimeError(
            f"Could not read PDF '{filename}': {error}"
        )

    return documents


# ============================================================
# DOCX EXTRACTION
# ============================================================

def extract_docx(
    file_bytes,
    filename
):

    documents = []

    try:

        doc = DocxDocument(
            BytesIO(file_bytes)
        )

        paragraphs = []

        for paragraph in doc.paragraphs:

            text = (
                paragraph.text.strip()
            )

            if text:

                paragraphs.append(
                    text
                )

        full_text = "\n".join(
            paragraphs
        )

        if full_text.strip():

            documents.append(
                Document(
                    page_content=full_text,
                    metadata={
                        "source":
                            filename,

                        "type":
                            "DOCX",
                    },
                )
            )

    except Exception as error:

        raise RuntimeError(
            f"Could not read DOCX '{filename}': {error}"
        )

    return documents


# ============================================================
# TXT EXTRACTION
# ============================================================

def extract_txt(
    file_bytes,
    filename
):

    try:

        text = file_bytes.decode(
            "utf-8",
            errors="ignore"
        )

    except Exception as error:

        raise RuntimeError(
            f"Could not read TXT '{filename}': {error}"
        )

    if not text.strip():

        return []

    return [
        Document(
            page_content=text,
            metadata={
                "source":
                    filename,

                "type":
                    "TXT",
            },
        )
    ]


# ============================================================
# EXCEL EXTRACTION
# ============================================================

def extract_excel(
    file_bytes,
    filename
):

    documents = []

    try:

        excel_file = pd.ExcelFile(
            BytesIO(file_bytes)
        )

        for sheet_name in (
            excel_file.sheet_names
        ):

            df = pd.read_excel(
                excel_file,
                sheet_name=sheet_name
            )

            if df.empty:

                continue

            text = df.to_string(
                index=False
            )

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source":
                            filename,

                        "sheet":
                            sheet_name,

                        "type":
                            "EXCEL",
                    },
                )
            )

    except Exception as error:

        raise RuntimeError(
            f"Could not read Excel '{filename}': {error}"
        )

    return documents


# ============================================================
# DOCUMENT ROUTER
# ============================================================

def extract_document(
    file_bytes,
    filename
):

    extension = (
        filename
        .lower()
        .split(".")[-1]
    )

    if extension == "pdf":

        return extract_pdf(
            file_bytes,
            filename
        )

    if extension == "docx":

        return extract_docx(
            file_bytes,
            filename
        )

    if extension == "txt":

        return extract_txt(
            file_bytes,
            filename
        )

    if extension in [
        "xlsx",
        "xls"
    ]:

        return extract_excel(
            file_bytes,
            filename
        )

    return []


# ============================================================
# WEBSITE
# ============================================================

def extract_website(url):

    if not url.strip():

        return []

    if not url.startswith(
        (
            "http://",
            "https://"
        )
    ):

        url = "https://" + url

    try:

        response = requests.get(
            url,
            timeout=15,
            headers={
                "User-Agent":
                    "Mozilla/5.0"
            }
        )

        response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        for tag in soup(
            [
                "script",
                "style",
                "noscript",
                "header",
                "footer",
                "nav",
            ]
        ):

            tag.decompose()

        text = soup.get_text(
            separator=" ",
            strip=True
        )

        if not text:

            return []

        return [
            Document(
                page_content=text,
                metadata={
                    "source": url,
                    "type": "WEBSITE",
                },
            )
        ]

    except Exception as error:

        raise RuntimeError(
            f"Could not read website: {error}"
        )


# ============================================================
# CREATE RAG CHUNKS
# ============================================================

def create_chunks(documents):

    chunks = []

    for document in documents:

        pieces = split_text(
            document.page_content
        )

        for index, piece in enumerate(
            pieces
        ):

            metadata = dict(
                document.metadata
            )

            metadata["chunk"] = (
                index + 1
            )

            chunks.append(
                Document(
                    page_content=piece,
                    metadata=metadata
                )
            )

    return chunks


# ============================================================
# EMBEDDINGS
# ============================================================

@st.cache_resource
def get_embeddings():

    return HuggingFaceEmbeddings(
        model_name=
            "sentence-transformers/all-MiniLM-L6-v2"
    )


# ============================================================
# VECTOR STORE
# ============================================================

def build_vectorstore(
    documents
):

    chunks = create_chunks(
        documents
    )

    if not chunks:

        return None

    embeddings = get_embeddings()

    return FAISS.from_documents(
        chunks,
        embeddings
    )


def create_document_signature(
    documents
):

    values = []

    for document in documents:

        values.append(
            (
                document.metadata.get(
                    "source",
                    ""
                ),

                len(
                    document.page_content
                ),
            )
        )

    return str(values)


def rebuild_rag_if_needed():

    documents = (
        st.session_state.documents
    )

    if not documents:

        st.session_state.vectorstore = None

        st.session_state.vectorstore_signature = None

        return

    signature = (
        create_document_signature(
            documents
        )
    )

    if (
        st.session_state.vectorstore is None
        or
        st.session_state.vectorstore_signature
        != signature
    ):

        with st.spinner(
            "🔎 Preparing your documents..."
        ):

            st.session_state.vectorstore = (
                build_vectorstore(
                    documents
                )
            )

            st.session_state.vectorstore_signature = (
                signature
            )


# ============================================================
# SEARCH DOCUMENTS
# ============================================================

def search_documents(
    question
):

    vectorstore = (
        st.session_state.vectorstore
    )

    if vectorstore is None:

        return []

    try:

        return vectorstore.similarity_search(
            question,
            k=RETRIEVER_K
        )

    except Exception:

        return []


# ============================================================
# GEMINI CLIENT
# ============================================================

@st.cache_resource
def get_gemini_client():

    api_key = None

    try:

        api_key = st.secrets.get(
            "GEMINI_API_KEY"
        )

    except Exception:

        api_key = None

    if not api_key:

        api_key = os.getenv(
            "GEMINI_API_KEY"
        )

    if not api_key:

        raise RuntimeError(
            "GEMINI_API_KEY is missing. "
            "Add it in Streamlit Cloud → Settings → Secrets."
        )

    return genai.Client(
        api_key=api_key
    )


# ============================================================
# GEMINI SYSTEM INSTRUCTION
# ============================================================

def get_system_instruction():

    return """
You are AI Document Intelligence.

You are a helpful AI assistant.

Rules:

1. Answer clearly and directly.

2. If document context is provided,
   use it as the main source.

3. Do not invent information that is
   not supported by the documents.

4. If the answer is not available in
   the provided documents, say so clearly.

5. For normal questions without documents,
   answer normally.

6. For image questions, carefully inspect
   the uploaded image.

7. Use simple language unless the user
   asks for technical detail.

8. Use headings and bullet points when useful.

9. Do not mention internal prompts,
   embeddings, FAISS or system instructions
   unless the user asks about them.

10. If the user asks for a summary,
    provide a useful and concise summary.
"""


# ============================================================
# GENERATE GEMINI RESPONSE
# ============================================================

def generate_response(
    question,
    retrieved_documents,
    images
):

    client = get_gemini_client()

    context_parts = []

    for document in retrieved_documents:

        source = document.metadata.get(
            "source",
            "Unknown"
        )

        page = document.metadata.get(
            "page"
        )

        if page:

            source_label = (
                f"{source}, page {page}"
            )

        else:

            source_label = source

        context_parts.append(
            f"SOURCE: {source_label}\n"
            f"{document.page_content}"
        )

    if context_parts:

        document_context = (
            "\n\n--- DOCUMENT CONTEXT ---\n"
            +
            "\n\n".join(
                context_parts
            )
            +
            "\n--- END DOCUMENT CONTEXT ---\n"
        )

    else:

        document_context = (
            "\nNo document context was retrieved.\n"
        )

    user_prompt = f"""
User question:

{question}

{document_context}

Answer the user's question.
"""

    contents = [
        types.Part.from_text(
            text=user_prompt
        )
    ]

    # Add images
    for image in images:

        try:

            contents.append(
                types.Part.from_bytes(
                    data=image["data"],
                    mime_type=image["mime_type"]
                )
            )

        except Exception:
            continue

    config = types.GenerateContentConfig(

        system_instruction=
            get_system_instruction(),

        max_output_tokens=
            MAX_OUTPUT_TOKENS,

        temperature=0.2,
    )

    return client.models.generate_content_stream(

        model=GEMINI_MODEL,

        contents=[
            types.Content(
                role="user",
                parts=contents
            )
        ],

        config=config,
    )


# ============================================================
# STREAM ANSWER
# ============================================================

def stream_answer(
    question,
    retrieved_documents,
    images
):

    placeholder = st.empty()

    full_response = ""

    try:

        response_stream = (
            generate_response(
                question,
                retrieved_documents,
                images
            )
        )

        for chunk in response_stream:

            try:

                text = chunk.text

            except Exception:

                text = ""

            if text:

                full_response += text

                placeholder.markdown(
                    full_response + "▌"
                )

        placeholder.markdown(
            full_response
        )

        return full_response

    except Exception as error:

        placeholder.empty()

        st.error(
            "❌ AI response failed."
        )

        with st.expander(
            "Show technical error"
        ):

            st.code(
                str(error)
            )

        return ""


# ============================================================
# ADD MESSAGE
# ============================================================

def add_message_to_chat(
    chat_id,
    role,
    content,
    sources=None
):

    chat = get_chat(
        chat_id
    )

    if not chat:

        return

    message = {

        "role": role,

        "content": content,

        "timestamp":
            datetime.now().isoformat(),
    }

    if sources:

        message["sources"] = sources

    chat["messages"].append(
        message
    )

    chat["updated_at"] = (
        datetime.now().isoformat()
    )

    save_chat(
        chat_id,
        chat
    )


# ============================================================
# NEW CHAT
# ============================================================

def start_new_chat():

    chat_id = create_chat(
        st.session_state.username
    )

    st.session_state.current_chat_id = (
        chat_id
    )


# ============================================================
# SIDEBAR
# ============================================================

def show_sidebar():

    with st.sidebar:

        st.title(
            "🤖 AI Document Intelligence"
        )

        st.caption(
            "Your personal AI workspace"
        )

        st.divider()

        if st.button(
            "✏️ New Chat",
            use_container_width=True
        ):

            start_new_chat()

            st.rerun()

        st.divider()

        st.subheader(
            "📚 Knowledge"
        )

        document_files = st.file_uploader(

            "Upload documents",

            type=[
                "pdf",
                "docx",
                "txt",
                "xlsx",
                "xls",
            ],

            accept_multiple_files=True,
        )

        if document_files:

            current_names = {

                doc.metadata.get(
                    "source",
                    ""
                )

                for doc in (
                    st.session_state.documents
                )
            }

            added_count = 0

            for uploaded_file in document_files:

                if (
                    uploaded_file.name
                    in current_names
                ):

                    continue

                try:

                    file_bytes = (
                        uploaded_file.getvalue()
                    )

                    extracted = (
                        extract_document(
                            file_bytes,
                            uploaded_file.name
                        )
                    )

                    if extracted:

                        st.session_state.documents.extend(
                            extracted
                        )

                        current_names.add(
                            uploaded_file.name
                        )

                        added_count += 1

                except Exception as error:

                    st.error(
                        f"{uploaded_file.name}: {error}"
                    )

            if added_count:

                st.session_state.vectorstore = None

                st.session_state.vectorstore_signature = None

                st.success(
                    f"Added {added_count} document(s)."
                )

        # ----------------------------------------------------
        # IMAGES
        # ----------------------------------------------------

        image_files = st.file_uploader(

            "Upload images",

            type=[
                "png",
                "jpg",
                "jpeg",
                "webp",
            ],

            accept_multiple_files=True,
        )

        if image_files:

            existing_names = {

                image["name"]

                for image in (
                    st.session_state.images
                )
            }

            for image_file in image_files:

                if (
                    image_file.name
                    in existing_names
                ):

                    continue

                st.session_state.images.append(
                    {
                        "name":
                            image_file.name,

                        "mime_type":
                            image_file.type
                            or
                            "image/jpeg",

                        "data":
                            image_file.getvalue(),
                    }
                )

        # ----------------------------------------------------
        # WEBSITE
        # ----------------------------------------------------

        st.subheader(
            "🌐 Website"
        )

        website_url = st.text_input(
            "Website URL",
            value=(
                st.session_state.website_url
            ),
            placeholder=
                "https://example.com",
        )

        if st.button(
            "Load Website",
            use_container_width=True
        ):

            if not website_url.strip():

                st.warning(
                    "Enter a website URL first."
                )

            else:

                try:

                    website_documents = (
                        extract_website(
                            website_url
                        )
                    )

                    if website_documents:

                        st.session_state.documents.extend(
                            website_documents
                        )

                        st.session_state.website_url = (
                            website_url
                        )

                        st.session_state.website_loaded = True

                        st.session_state.vectorstore = None

                        st.session_state.vectorstore_signature = None

                        st.success(
                            "Website loaded successfully."
                        )

                    else:

                        st.warning(
                            "No readable text was found."
                        )

                except Exception as error:

                    st.error(
                        str(error)
                    )

        # ----------------------------------------------------
        # LOADED DOCUMENTS
        # ----------------------------------------------------

        if st.session_state.documents:

            st.divider()

            st.subheader(
                "📄 Loaded Documents"
            )

            source_names = []

            for document in (
                st.session_state.documents
            ):

                source = document.metadata.get(
                    "source",
                    "Unknown"
                )

                if source not in source_names:

                    source_names.append(
                        source
                    )

            for source in source_names:

                st.caption(
                    f"📄 {source}"
                )

        # ----------------------------------------------------
        # LOADED IMAGES
        # ----------------------------------------------------

        if st.session_state.images:

            st.subheader(
                "🖼️ Loaded Images"
            )

            for image in (
                st.session_state.images
            ):

                st.caption(
                    f"🖼️ {image['name']}"
                )

        # ----------------------------------------------------
        # CLEAR KNOWLEDGE
        # ----------------------------------------------------

        if (
            st.session_state.documents
            or
            st.session_state.images
        ):

            if st.button(
                "🗑️ Clear Knowledge",
                use_container_width=True
            ):

                st.session_state.documents = []

                st.session_state.images = []

                st.session_state.vectorstore = None

                st.session_state.vectorstore_signature = None

                st.session_state.website_url = ""

                st.session_state.website_loaded = False

                st.rerun()

        # ----------------------------------------------------
        # CHAT HISTORY
        # ----------------------------------------------------

        st.divider()

        st.subheader(
            "💬 Chats"
        )

        user_chats = get_user_chats(
            st.session_state.username
        )

        if not user_chats:

            st.caption(
                "No previous chats."
            )

        else:

            for chat in user_chats:

                chat_id = chat["id"]

                title = chat["title"]

                col1, col2 = st.columns(
                    [5, 1]
                )

                with col1:

                    if st.button(
                        f"💬 {title}",
                        key=f"open_{chat_id}",
                        use_container_width=True,
                    ):

                        st.session_state.current_chat_id = (
                            chat_id
                        )

                        st.rerun()

                with col2:

                    if st.button(
                        "×",
                        key=f"delete_{chat_id}"
                    ):

                        delete_chat(
                            chat_id
                        )

                        if (
                            st.session_state.current_chat_id
                            == chat_id
                        ):

                            st.session_state.current_chat_id = None

                        st.rerun()

        # ----------------------------------------------------
        # LOGOUT
        # ----------------------------------------------------

        st.divider()

        if st.button(
            "🚪 Logout",
            use_container_width=True
        ):

            st.session_state.logged_in = False

            st.session_state.username = ""

            st.session_state.current_chat_id = None

            st.session_state.documents = []

            st.session_state.images = []

            st.session_state.vectorstore = None

            st.rerun()


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

def display_chat_history(
    chat
):

    if not chat:

        return

    messages = chat.get(
        "messages",
        []
    )

    for message in messages:

        role = message.get(
            "role",
            "assistant"
        )

        content = message.get(
            "content",
            ""
        )

        with st.chat_message(
            role
        ):

            st.markdown(
                content
            )

            sources = message.get(
                "sources",
                []
            )

            if sources:

                with st.expander(
                    "📚 Sources"
                ):

                    for source in sources:

                        st.caption(
                            f"• {source}"
                        )


# ============================================================
# WELCOME
# ============================================================

def show_welcome():

    greeting = get_time_greeting()

    st.title(
        f"{greeting}, "
        f"{st.session_state.username}! 👋"
    )

    st.write(
        "How can I help you today?"
    )

    st.divider()

    col1, col2, col3 = st.columns(
        3
    )

    with col1:

        st.info(
            "📄 **Documents**\n\n"
            "Upload PDF, Word, TXT or Excel "
            "files and ask questions."
        )

    with col2:

        st.info(
            "🖼️ **Images**\n\n"
            "Upload images and ask questions "
            "about their content."
        )

    with col3:

        st.info(
            "🌐 **Websites**\n\n"
            "Load a website and ask questions "
            "about its content."
        )

    st.divider()

    st.caption(
        "💡 Upload your files from the sidebar "
        "and ask your question below."
    )


# ============================================================
# MAIN CHAT
# ============================================================

def show_chat():

    rebuild_rag_if_needed()

    if (
        st.session_state.current_chat_id
        is None
    ):

        show_welcome()

    else:

        chat = get_chat(
            st.session_state.current_chat_id
        )

        if not chat:

            start_new_chat()

            chat = get_chat(
                st.session_state.current_chat_id
            )

        if chat:

            title = chat.get(
                "title",
                "New Chat"
            )

            st.title(
                f"💬 {title}"
            )

            display_chat_history(
                chat
            )

    question = st.chat_input(
        "Ask anything about your documents..."
    )

    if not question:

        return

    question = question.strip()

    if not question:

        return

    if (
        st.session_state.current_chat_id
        is None
    ):

        start_new_chat()

    chat_id = (
        st.session_state.current_chat_id
    )

    chat = get_chat(
        chat_id
    )

    if not chat:

        return

    if chat.get(
        "title"
    ) == "New Chat":

        chat["title"] = (
            make_chat_title(
                question
            )
        )

        save_chat(
            chat_id,
            chat
        )

    with st.chat_message(
        "user"
    ):

        st.markdown(
            question
        )

    add_message_to_chat(
        chat_id,
        "user",
        question
    )

    retrieved_documents = (
        search_documents(
            question
        )
    )

    source_names = []

    for document in (
        retrieved_documents
    ):

        source = document.metadata.get(
            "source"
        )

        if (
            source
            and
            source not in source_names
        ):

            source_names.append(
                source
            )

    with st.chat_message(
        "assistant"
    ):

        answer = stream_answer(

            question,

            retrieved_documents,

            st.session_state.images
        )

        if answer:

            if source_names:

                with st.expander(
                    "📚 Sources"
                ):

                    for source in source_names:

                        st.caption(
                            f"• {source}"
                        )

            add_message_to_chat(

                chat_id,

                "assistant",

                answer,

                source_names
            )


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():

    if not st.session_state.logged_in:

        show_login()

        return

    show_sidebar()

    show_chat()


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except Exception:

        st.error(
            "❌ Something went wrong."
        )

        with st.expander(
            "Show technical error"
        ):

            st.exception(
                Exception
            )

# ============================================================
# AI DOCUMENT INTELLIGENCE
# Full Streamlit Application
# ============================================================

import os
import re
import json
import uuid
import hashlib
from io import BytesIO
from datetime import datetime
from zoneinfo import ZoneInfo

import streamlit as st
import pandas as pd
import requests

from bs4 import BeautifulSoup
from PIL import Image

from pypdf import PdfReader
from docx import Document as DocxDocument

from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from google import genai
from google.genai import types


# ============================================================
# CONFIGURATION
# ============================================================

APP_NAME = "AI Document Intelligence"

# IMPORTANT:
# Changed from gemini-2.5-flash-lite
# to gemini-2.5-flash
GEMINI_MODEL = "gemini-2.5-flash"

MAX_OUTPUT_TOKENS = 4096

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150

RETRIEVER_K = 4

USERS_FILE = "users.json"
CHATS_FILE = "chats.json"


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title=APP_NAME,
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background-color: #0e1117;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #262730;
    }

    /* Chat input */
    div[data-testid="stChatInput"] {
        background-color: #1b1e26;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
    }

    /* File uploader */
    div[data-testid="stFileUploader"] {
        background-color: #11141b;
        border-radius: 12px;
        padding: 8px;
    }

    /* Info boxes */
    div[data-testid="stAlert"] {
        border-radius: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# TIME GREETING
# ============================================================

def get_time_greeting():
    """Return greeting based on Indian Standard Time."""

    ist_now = datetime.now(ZoneInfo("Asia/Kolkata"))
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
# FILE HELPERS
# ============================================================

def safe_json_load(filename, default):
    """Safely load JSON file."""

    if not os.path.exists(filename):
        return default

    try:
        with open(filename, "r", encoding="utf-8") as file:
            return json.load(file)

    except Exception:
        return default


def safe_json_save(filename, data):
    """Safely save JSON file."""

    try:
        with open(filename, "w", encoding="utf-8") as file:
            json.dump(
                data,
                file,
                indent=2,
                ensure_ascii=False
            )

    except Exception:
        pass


# ============================================================
# PASSWORD HASHING
# ============================================================

def hash_password(password):
    return hashlib.sha256(
        password.encode("utf-8")
    ).hexdigest()


# ============================================================
# USER MANAGEMENT
# ============================================================

def load_users():
    return safe_json_load(USERS_FILE, {})


def save_users(users):
    safe_json_save(USERS_FILE, users)


def create_user(username, password):
    users = load_users()

    username = username.strip().lower()

    if not username or not password:
        return False, "Username and password are required."

    if username in users:
        return False, "Username already exists."

    users[username] = {
        "password": hash_password(password),
        "created_at": datetime.now().isoformat(),
    }

    save_users(users)

    return True, "Account created successfully."


def authenticate_user(username, password):
    users = load_users()

    username = username.strip().lower()

    if username not in users:
        return False

    stored_password = users[username].get("password")

    return stored_password == hash_password(password)


# ============================================================
# CHAT STORAGE
# ============================================================

def load_chats():
    return safe_json_load(CHATS_FILE, {})


def save_chats(chats):
    safe_json_save(CHATS_FILE, chats)


def get_user_chats(username):
    chats = load_chats()

    if username not in chats:
        chats[username] = {}

    return chats


def create_new_chat(username):
    chats = load_chats()

    chat_id = str(uuid.uuid4())

    if username not in chats:
        chats[username] = {}

    chats[username][chat_id] = {
        "title": "New Chat",
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "messages": [],
    }

    save_chats(chats)

    return chat_id


def delete_chat(username, chat_id):
    chats = load_chats()

    if username in chats:
        if chat_id in chats[username]:
            del chats[username][chat_id]

    save_chats(chats)


def save_message(username, chat_id, role, content):
    chats = load_chats()

    if username not in chats:
        chats[username] = {}

    if chat_id not in chats[username]:
        chats[username][chat_id] = {
            "title": "New Chat",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "messages": [],
        }

    chats[username][chat_id]["messages"].append(
        {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        }
    )

    chats[username][chat_id]["updated_at"] = datetime.now().isoformat()

    save_chats(chats)


def update_chat_title(username, chat_id, title):
    chats = load_chats()

    if username in chats and chat_id in chats[username]:

        chats[username][chat_id]["title"] = title[:60]
        chats[username][chat_id]["updated_at"] = (
            datetime.now().isoformat()
        )

    save_chats(chats)


# ============================================================
# CHAT TITLE
# ============================================================

def generate_chat_title(question):
    question = question.strip()

    if len(question) <= 45:
        return question

    return question[:45].rstrip() + "..."


# ============================================================
# SESSION STATE
# ============================================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

if "chat_id" not in st.session_state:
    st.session_state.chat_id = None

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "documents" not in st.session_state:
    st.session_state.documents = []

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

if "uploaded_images" not in st.session_state:
    st.session_state.uploaded_images = []

if "website_loaded" not in st.session_state:
    st.session_state.website_loaded = False

if "website_text" not in st.session_state:
    st.session_state.website_text = ""


# ============================================================
# LOGIN / SIGNUP PAGE
# ============================================================

def show_auth_page():

    st.title("🤖 AI Document Intelligence")

    st.caption(
        "Your personal AI workspace for documents, images and websites."
    )

    st.divider()

    tab1, tab2 = st.tabs(
        ["🔐 Login", "📝 Create Account"]
    )

    # --------------------------------------------------------
    # LOGIN
    # --------------------------------------------------------

    with tab1:

        st.subheader("Welcome Back")

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
            use_container_width=True,
            type="primary"
        ):

            if authenticate_user(username, password):

                st.session_state.logged_in = True
                st.session_state.username = (
                    username.strip().lower()
                )

                chat_data = get_user_chats(
                    st.session_state.username
                )

                user_chats = chat_data[
                    st.session_state.username
                ]

                if user_chats:

                    latest_chat = sorted(
                        user_chats.items(),
                        key=lambda item:
                        item[1].get(
                            "updated_at",
                            ""
                        ),
                        reverse=True
                    )[0][0]

                    st.session_state.chat_id = latest_chat

                else:

                    st.session_state.chat_id = (
                        create_new_chat(
                            st.session_state.username
                        )
                    )

                st.rerun()

            else:

                st.error(
                    "Invalid username or password."
                )

    # --------------------------------------------------------
    # SIGNUP
    # --------------------------------------------------------

    with tab2:

        st.subheader("Create Your Account")

        new_username = st.text_input(
            "Choose Username",
            key="signup_username"
        )

        new_password = st.text_input(
            "Choose Password",
            type="password",
            key="signup_password"
        )

        confirm_password = st.text_input(
            "Confirm Password",
            type="password",
            key="signup_confirm_password"
        )

        if st.button(
            "Create Account",
            use_container_width=True,
            type="primary"
        ):

            if new_password != confirm_password:

                st.error(
                    "Passwords do not match."
                )

            else:

                success, message = create_user(
                    new_username,
                    new_password
                )

                if success:
                    st.success(message)
                    st.info(
                        "You can now login."
                    )

                else:
                    st.error(message)


# ============================================================
# GEMINI CLIENT
# ============================================================

@st.cache_resource
def get_gemini_client():

    api_key = None

    # Streamlit Cloud Secrets
    try:
        api_key = st.secrets["GEMINI_API_KEY"]

    except Exception:
        api_key = None

    # Local environment variable
    if not api_key:
        api_key = os.getenv(
            "GEMINI_API_KEY"
        )

    if not api_key:

        raise RuntimeError(
            "GEMINI_API_KEY is missing. "
            "Go to Streamlit Cloud → "
            "Manage app → Settings → Secrets "
            "and add GEMINI_API_KEY."
        )

    return genai.Client(
        api_key=api_key
    )


# ============================================================
# EMBEDDINGS
# ============================================================

@st.cache_resource
def get_embeddings():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


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

        end = start + chunk_size

        chunk = text[start:end]

        if end < text_length:

            last_break = max(
                chunk.rfind(". "),
                chunk.rfind("\n"),
                chunk.rfind(" ")
            )

            if last_break > chunk_size * 0.5:
                end = start + last_break + 1
                chunk = text[start:end]

        chunks.append(
            chunk.strip()
        )

        next_start = end - overlap

        if next_start <= start:
            next_start = end

        start = next_start

    return [
        chunk
        for chunk in chunks
        if chunk
    ]


# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_pdf(file_bytes):

    reader = PdfReader(
        BytesIO(file_bytes)
    )

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        try:
            text = page.extract_text() or ""

        except Exception:
            text = ""

        if text.strip():

            pages.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": "PDF",
                        "page": page_number,
                    }
                )
            )

    return pages


# ============================================================
# DOCX EXTRACTION
# ============================================================

def extract_docx(file_bytes):

    document = DocxDocument(
        BytesIO(file_bytes)
    )

    paragraphs = []

    for paragraph in document.paragraphs:

        text = paragraph.text.strip()

        if text:
            paragraphs.append(text)

    full_text = "\n".join(paragraphs)

    if not full_text:
        return []

    return [
        Document(
            page_content=full_text,
            metadata={
                "source": "Word Document"
            }
        )
    ]


# ============================================================
# TXT EXTRACTION
# ============================================================

def extract_txt(file_bytes):

    text = file_bytes.decode(
        "utf-8",
        errors="ignore"
    )

    if not text.strip():
        return []

    return [
        Document(
            page_content=text,
            metadata={
                "source": "Text File"
            }
        )
    ]


# ============================================================
# EXCEL EXTRACTION
# ============================================================

def extract_excel(file_bytes):

    excel_file = BytesIO(
        file_bytes
    )

    workbook = pd.ExcelFile(
        excel_file
    )

    documents = []

    for sheet_name in workbook.sheet_names:

        dataframe = pd.read_excel(
            excel_file,
            sheet_name=sheet_name
        )

        dataframe = dataframe.fillna("")

        text = dataframe.to_string(
            index=False
        )

        if text.strip():

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": "Excel",
                        "sheet": sheet_name,
                    }
                )
            )

    return documents


# ============================================================
# GENERIC FILE EXTRACTION
# ============================================================

def extract_file(file):

    file_bytes = file.getvalue()

    filename = file.name.lower()

    if filename.endswith(".pdf"):

        return extract_pdf(
            file_bytes
        )

    elif filename.endswith(".docx"):

        return extract_docx(
            file_bytes
        )

    elif filename.endswith(".txt"):

        return extract_txt(
            file_bytes
        )

    elif filename.endswith(
        (".xlsx", ".xls")
    ):

        return extract_excel(
            file_bytes
        )

    return []


# ============================================================
# BUILD VECTOR DATABASE
# ============================================================

def build_vectorstore(documents):

    if not documents:
        return None

    all_chunks = []

    for document in documents:

        chunks = split_text(
            document.page_content
        )

        for chunk in chunks:

            all_chunks.append(
                Document(
                    page_content=chunk,
                    metadata=document.metadata
                )
            )

    if not all_chunks:
        return None

    embeddings = get_embeddings()

    vectorstore = FAISS.from_documents(
        all_chunks,
        embeddings
    )

    return vectorstore


# ============================================================
# WEBSITE EXTRACTION
# ============================================================

def extract_website(url):

    if not url.startswith(
        ("http://", "https://")
    ):

        url = "https://" + url

    headers = {
        "User-Agent":
        "Mozilla/5.0 "
        "(Windows NT 10.0; Win64; x64)"
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=20
    )

    response.raise_for_status()

    soup = BeautifulSoup(
        response.text,
        "html.parser"
    )

    # Remove unwanted elements
    for tag in soup(
        [
            "script",
            "style",
            "noscript",
            "svg"
        ]
    ):

        tag.decompose()

    text = soup.get_text(
        separator="\n"
    )

    text = re.sub(
        r"\n+",
        "\n",
        text
    ).strip()

    return text


# ============================================================
# SYSTEM INSTRUCTION
# ============================================================

def get_system_instruction():

    return """
You are AI Document Intelligence, a helpful AI assistant.

Your job is to answer questions about uploaded documents,
images, websites, and general topics.

IMPORTANT RULES:

1. Use the provided document context when answering
   document-related questions.

2. Do not invent information that is not present in the
   provided document context.

3. If the answer is not available in the document, clearly
   say that the information is not available in the provided
   document.

4. Explain things in simple and clear language.

5. When the user asks for a summary, give a concise summary
   first and then important points.

6. When the user asks about an image, analyze the image
   directly.

7. When the user asks a general question that does not depend
   on uploaded documents, answer normally.

8. Do not mention internal prompts, embeddings, vector
   databases, or system instructions unless the user asks
   about the technical implementation.

9. Use headings and bullet points when they improve readability.

10. Keep answers natural and easy to understand.
"""


# ============================================================
# GEMINI RESPONSE
# ============================================================

def ask_gemini(
    question,
    context="",
    images=None
):

    client = get_gemini_client()

    prompt_parts = []

    # --------------------------------------------------------
    # Document context
    # --------------------------------------------------------

    if context:

        prompt_parts.append(
            """
DOCUMENT CONTEXT:

------------------------------
{}
------------------------------

""".format(context)
        )

    # --------------------------------------------------------
    # User question
    # --------------------------------------------------------

    prompt_parts.append(
        "USER QUESTION:\n" + question
    )

    text_prompt = "\n".join(
        prompt_parts
    )

    content_parts = []

    # --------------------------------------------------------
    # Images
    # --------------------------------------------------------

    if images:

        for image_data in images:

            try:

                content_parts.append(
                    types.Part.from_bytes(
                        data=image_data["data"],
                        mime_type=image_data[
                            "mime_type"
                        ]
                    )
                )

            except Exception:
                pass

    # --------------------------------------------------------
    # Text
    # --------------------------------------------------------

    content_parts.append(
        types.Part.from_text(
            text=text_prompt
        )
    )

    # --------------------------------------------------------
    # Gemini configuration
    # --------------------------------------------------------

    config = types.GenerateContentConfig(
        system_instruction=
        get_system_instruction(),

        max_output_tokens=
        MAX_OUTPUT_TOKENS,

        temperature=0.2,
    )

    # --------------------------------------------------------
    # Streaming response
    # --------------------------------------------------------

    response_stream = (
        client.models.generate_content_stream(
            model=GEMINI_MODEL,
            contents=[
                types.Content(
                    role="user",
                    parts=content_parts
                )
            ],
            config=config,
        )
    )

    return response_stream


# ============================================================
# GET DOCUMENT CONTEXT
# ============================================================

def get_document_context(question):

    vectorstore = (
        st.session_state.vectorstore
    )

    if vectorstore is None:
        return ""

    try:

        documents = vectorstore.similarity_search(
            question,
            k=RETRIEVER_K
        )

    except Exception:
        return ""

    if not documents:
        return ""

    context_parts = []

    for index, document in enumerate(
        documents,
        start=1
    ):

        source = document.metadata.get(
            "source",
            "Document"
        )

        page = document.metadata.get(
            "page"
        )

        sheet = document.metadata.get(
            "sheet"
        )

        metadata_text = source

        if page:
            metadata_text += (
                f" | Page {page}"
            )

        if sheet:
            metadata_text += (
                f" | Sheet {sheet}"
            )

        context_parts.append(
            f"[Source {index}: "
            f"{metadata_text}]\n"
            f"{document.page_content}"
        )

    return "\n\n".join(
        context_parts
    )


# ============================================================
# SIDEBAR
# ============================================================

def show_sidebar():

    with st.sidebar:

        st.title("🤖 AI Document")
        st.title("Intelligence")

        st.caption(
            "Your personal AI workspace"
        )

        st.divider()

        # ----------------------------------------------------
        # New Chat
        # ----------------------------------------------------

        if st.button(
            "📝 New Chat",
            use_container_width=True
        ):

            st.session_state.chat_id = (
                create_new_chat(
                    st.session_state.username
                )
            )

            st.rerun()

        st.divider()

        # ----------------------------------------------------
        # Knowledge
        # ----------------------------------------------------

        st.subheader("📚 Knowledge")

        uploaded_files = st.file_uploader(
            "Upload documents",
            type=[
                "pdf",
                "docx",
                "txt",
                "xlsx",
                "xls",
            ],
            accept_multiple_files=True,
            key="document_uploader",
        )

        if uploaded_files:

            current_names = [
                file.name
                for file in uploaded_files
            ]

            previous_names = [
                file.name
                for file in
                st.session_state.uploaded_files
            ]

            if current_names != previous_names:

                all_documents = []

                successful_files = []

                for file in uploaded_files:

                    try:

                        extracted_documents = (
                            extract_file(file)
                        )

                        if extracted_documents:

                            all_documents.extend(
                                extracted_documents
                            )

                            successful_files.append(
                                file
                            )

                    except Exception as error:

                        st.warning(
                            f"Could not read "
                            f"{file.name}: "
                            f"{error}"
                        )

                if all_documents:

                    with st.spinner(
                        "Building document knowledge..."
                    ):

                        st.session_state.vectorstore = (
                            build_vectorstore(
                                all_documents
                            )
                        )

                    st.session_state.documents = (
                        all_documents
                    )

                    st.session_state.uploaded_files = (
                        successful_files
                    )

                    st.success(
                        f"{len(successful_files)} "
                        f"document(s) loaded."
                    )

        if st.session_state.uploaded_files:

            st.caption(
                "Uploaded documents:"
            )

            for file in (
                st.session_state.uploaded_files
            ):

                st.write(
                    f"📄 {file.name}"
                )

        # ----------------------------------------------------
        # Clear Knowledge
        # ----------------------------------------------------

        if st.session_state.vectorstore:

            if st.button(
                "🗑️ Clear Knowledge",
                use_container_width=True
            ):

                st.session_state.vectorstore = None
                st.session_state.documents = []
                st.session_state.uploaded_files = []

                st.success(
                    "Knowledge cleared."
                )

                st.rerun()

        st.divider()

        # ----------------------------------------------------
        # Images
        # ----------------------------------------------------

        st.subheader("🖼️ Upload Images")

        image_files = st.file_uploader(
            "Upload images",
            type=[
                "png",
                "jpg",
                "jpeg",
                "webp",
            ],
            accept_multiple_files=True,
            key="image_uploader",
        )

        if image_files:

            image_list = []

            for image_file in image_files:

                try:

                    image_bytes = (
                        image_file.getvalue()
                    )

                    # Validate image
                    Image.open(
                        BytesIO(image_bytes)
                    )

                    mime_type = (
                        image_file.type
                        or "image/png"
                    )

                    image_list.append(
                        {
                            "name":
                            image_file.name,

                            "data":
                            image_bytes,

                            "mime_type":
                            mime_type,
                        }
                    )

                except Exception as error:

                    st.warning(
                        f"Could not load "
                        f"{image_file.name}: "
                        f"{error}"
                    )

            st.session_state.uploaded_images = (
                image_list
            )

        if st.session_state.uploaded_images:

            st.caption(
                "Uploaded images:"
            )

            for image in (
                st.session_state.uploaded_images
            ):

                st.write(
                    f"🖼️ {image['name']}"
                )

        st.divider()

        # ----------------------------------------------------
        # Website
        # ----------------------------------------------------

        st.subheader("🌐 Website")

        website_url = st.text_input(
            "Website URL",
            placeholder=
            "https://example.com",
            key="website_url",
        )

        if st.button(
            "Load Website",
            use_container_width=True
        ):

            if not website_url.strip():

                st.warning(
                    "Please enter a website URL."
                )

            else:

                with st.spinner(
                    "Loading website..."
                ):

                    try:

                        website_text = (
                            extract_website(
                                website_url.strip()
                            )
                        )

                        if website_text:

                            website_document = (
                                Document(
                                    page_content=
                                    website_text,
                                    metadata={
                                        "source":
                                        website_url
                                    }
                                )
                            )

                            st.session_state.vectorstore = (
                                build_vectorstore(
                                    [website_document]
                                )
                            )

                            st.session_state.documents = [
                                website_document
                            ]

                            st.session_state.website_text = (
                                website_text
                            )

                            st.session_state.website_loaded = (
                                True
                            )

                            st.success(
                                "Website loaded successfully."
                            )

                        else:

                            st.error(
                                "No readable content found."
                            )

                    except Exception as error:

                        st.error(
                            f"Could not load website: "
                            f"{error}"
                        )

        # ----------------------------------------------------
        # Chat History
        # ----------------------------------------------------

        st.divider()

        st.subheader("💬 Chat History")

        chats = load_chats()

        user_chats = chats.get(
            st.session_state.username,
            {}
        )

        if user_chats:

            sorted_chats = sorted(
                user_chats.items(),
                key=lambda item:
                item[1].get(
                    "updated_at",
                    ""
                ),
                reverse=True
            )

            for chat_id, chat in sorted_chats:

                title = chat.get(
                    "title",
                    "New Chat"
                )

                col1, col2 = st.columns(
                    [5, 1]
                )

                with col1:

                    if st.button(
                        f"💬 {title}",
                        key=f"open_{chat_id}",
                        use_container_width=True
                    ):

                        st.session_state.chat_id = (
                            chat_id
                        )

                        st.rerun()

                with col2:

                    if st.button(
                        "🗑️",
                        key=f"delete_{chat_id}"
                    ):

                        delete_chat(
                            st.session_state.username,
                            chat_id
                        )

                        remaining_chats = (
                            load_chats()
                            .get(
                                st.session_state.username,
                                {}
                            )
                        )

                        if remaining_chats:

                            st.session_state.chat_id = (
                                list(
                                    remaining_chats.keys()
                                )[0]
                            )

                        else:

                            st.session_state.chat_id = (
                                create_new_chat(
                                    st.session_state.username
                                )
                            )

                        st.rerun()

        # ----------------------------------------------------
        # Logout
        # ----------------------------------------------------

        st.divider()

        if st.button(
            "🚪 Logout",
            use_container_width=True
        ):

            st.session_state.logged_in = False
            st.session_state.username = ""
            st.session_state.chat_id = None
            st.session_state.vectorstore = None
            st.session_state.documents = []
            st.session_state.uploaded_files = []
            st.session_state.uploaded_images = []

            st.rerun()


# ============================================================
# MAIN CHAT PAGE
# ============================================================

def show_main_app():

    show_sidebar()

    username = (
        st.session_state.username
    )

    chat_id = (
        st.session_state.chat_id
    )

    if not chat_id:

        chat_id = create_new_chat(
            username
        )

        st.session_state.chat_id = chat_id

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    st.title(
        f"{get_time_greeting()} 👋"
    )

    st.subheader(
        "🤖 AI Document Intelligence"
    )

    st.caption(
        "Ask questions about your documents, "
        "images or websites."
    )

    # --------------------------------------------------------
    # Feature Cards
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        st.info(
            """
            📄 **Documents**

            Upload PDF, Word, TXT or Excel
            files and ask questions.
            """
        )

    with col2:

        st.info(
            """
            🖼️ **Images**

            Upload images and ask questions
            about their content.
            """
        )

    with col3:

        st.info(
            """
            🌐 **Websites**

            Load a website and ask questions
            about its content.
            """
        )

    st.divider()

    # --------------------------------------------------------
    # Knowledge status
    # --------------------------------------------------------

    if st.session_state.vectorstore:

        st.success(
            "📚 Document knowledge is ready. "
            "Ask your question below."
        )

    elif st.session_state.uploaded_images:

        st.info(
            "🖼️ Image uploaded. "
            "Ask your question below."
        )

    else:

        st.info(
            "💡 Upload a document or image "
            "from the sidebar and ask your question below."
        )

    # --------------------------------------------------------
    # Load current chat
    # --------------------------------------------------------

    chats = load_chats()

    current_chat = (
        chats
        .get(username, {})
        .get(chat_id)
    )

    if current_chat is None:

        chat_id = create_new_chat(
            username
        )

        st.session_state.chat_id = chat_id

        chats = load_chats()

        current_chat = (
            chats
            .get(username, {})
            .get(chat_id)
        )

    # --------------------------------------------------------
    # Display chat messages
    # --------------------------------------------------------

    messages = current_chat.get(
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

        with st.chat_message(role):

            st.markdown(content)

    # --------------------------------------------------------
    # Chat input
    # --------------------------------------------------------

    question = st.chat_input(
        "Ask anything about your documents..."
    )

    if question:

        question = question.strip()

        if not question:
            st.stop()

        # ----------------------------------------------------
        # Save user message
        # ----------------------------------------------------

        save_message(
            username,
            chat_id,
            "user",
            question
        )

        # ----------------------------------------------------
        # Automatically create title
        # ----------------------------------------------------

        current_title = current_chat.get(
            "title",
            "New Chat"
        )

        if current_title == "New Chat":

            update_chat_title(
                username,
                chat_id,
                generate_chat_title(
                    question
                )
            )

        # ----------------------------------------------------
        # Display user question
        # ----------------------------------------------------

        with st.chat_message("user"):

            st.markdown(question)

        # ----------------------------------------------------
        # Get relevant document context
        # ----------------------------------------------------

        context = get_document_context(
            question
        )

        # ----------------------------------------------------
        # Images
        # ----------------------------------------------------

        images = (
            st.session_state.uploaded_images
            if st.session_state.uploaded_images
            else []
        )

        # ----------------------------------------------------
        # Generate AI response
        # ----------------------------------------------------

        with st.chat_message("assistant"):

            response_placeholder = st.empty()

            full_response = ""

            try:

                response_stream = ask_gemini(
                    question=question,
                    context=context,
                    images=images,
                )

                for chunk in response_stream:

                    try:

                        text = chunk.text

                    except Exception:

                        text = ""

                    if text:

                        full_response += text

                        response_placeholder.markdown(
                            full_response
                        )

                if not full_response.strip():

                    full_response = (
                        "I could not generate a response."
                    )

                    response_placeholder.markdown(
                        full_response
                    )

            except Exception as error:

                error_message = str(error)

                st.error(
                    "❌ AI response failed."
                )

                with st.expander(
                    "Show technical error"
                ):

                    st.code(
                        error_message
                    )

                full_response = (
                    "AI response failed. "
                    "Please check the technical error."
                )

        # ----------------------------------------------------
        # Save AI response
        # ----------------------------------------------------

        save_message(
            username,
            chat_id,
            "assistant",
            full_response
        )

        st.rerun()


# ============================================================
# APPLICATION START
# ============================================================

if not st.session_state.logged_in:

    show_auth_page()

else:

    show_main_app()

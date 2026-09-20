import os
import json
import uuid
import re
import html
from datetime import datetime

import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from docx import Document as DocxDocument
from pypdf import PdfReader

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Document Intelligence",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# APPLICATION SETTINGS
# ============================================================

USERS_FILE = "users.json"
CHATS_FILE = "chats.json"
USAGE_FILE = "usage.json"

MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

MAX_NEW_TOKENS = 80

CHUNK_SIZE = 800

CHUNK_OVERLAP = 120

RETRIEVER_K = 2


# ============================================================
# GLOBAL CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ======================================================
       GLOBAL
       ====================================================== */

    html,
    body,
    [class*="css"] {
        font-family:
            -apple-system,
            BlinkMacSystemFont,
            "Segoe UI",
            sans-serif;
    }

    .stApp {
        background: #1f1e1b;
        color: #f4f1ea;
    }

    [data-testid="stAppViewContainer"] {
        background: #1f1e1b;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    /* ======================================================
       REMOVE DEFAULT TOP SPACE
       ====================================================== */

    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 7rem !important;
        max-width: 1100px !important;
    }

    /* ======================================================
       SIDEBAR
       ====================================================== */

    [data-testid="stSidebar"] {
        background: #171614;
        border-right: 1px solid #302e2a;
    }

    [data-testid="stSidebar"] > div:first-child {
        background: #171614;
        padding-top: 18px;
    }

    .sidebar-logo {
        display: flex;
        align-items: center;
        gap: 9px;
        color: #f4f1ea;
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 4px;
    }

    .sidebar-logo-star {
        color: #f0a45d;
        font-size: 23px;
    }

    .sidebar-subtitle {
        color: #8f8980;
        font-size: 11px;
        margin-left: 31px;
        margin-bottom: 22px;
    }

    /* ======================================================
       SIDEBAR BUTTONS
       ====================================================== */

    [data-testid="stSidebar"] .stButton > button {
        width: 100%;
        background: transparent;
        border: 1px solid transparent;
        color: #c9c3ba;
        border-radius: 8px;
        text-align: left;
        min-height: 38px;
        padding: 7px 10px;
        font-size: 13px;
    }

    [data-testid="stSidebar"] .stButton > button:hover {
        background: #25231f;
        border-color: #37342e;
        color: #f4f1ea;
    }

    /* ======================================================
       NEW CHAT
       ====================================================== */

    .new-chat-wrapper {
        margin-bottom: 15px;
    }

    /* ======================================================
       SIDEBAR SECTION LABEL
       ====================================================== */

    .sidebar-section-label {
        color: #777169;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 0.8px;
        margin-top: 18px;
        margin-bottom: 7px;
    }

    /* ======================================================
       CHAT HISTORY
       ====================================================== */

    .history-empty {
        color: #777169;
        font-size: 12px;
        padding: 7px 2px;
    }

    /* ======================================================
       SIDEBAR DIVIDER
       ====================================================== */

    .sidebar-divider {
        height: 1px;
        background: #302e2a;
        margin: 18px 0;
    }

    /* ======================================================
       SOURCE AREA
       ====================================================== */

    .source-title {
        color: #c9c3ba;
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 7px;
    }

    /* ======================================================
       MAIN CHAT
       ====================================================== */

    .main-top-space {
        height: 20px;
    }

    /* ======================================================
       WELCOME SCREEN
       ====================================================== */

    .welcome {
        text-align: center;
        margin-top: 135px;
        margin-bottom: 40px;
    }

    .welcome-star {
        color: #f0a45d;
        font-size: 34px;
        margin-bottom: 18px;
    }

    .welcome-title {
        color: #f4f1ea;
        font-size: 31px;
        font-weight: 650;
        letter-spacing: -0.5px;
    }

    .welcome-subtitle {
        color: #8f8980;
        font-size: 15px;
        margin-top: 9px;
    }

    /* ======================================================
       SUGGESTION CARDS
       ====================================================== */

    .suggestion-card {
        background: #25231f;
        border: 1px solid #37342e;
        border-radius: 12px;
        padding: 14px 15px;
        color: #bdb7ae;
        font-size: 13px;
        min-height: 64px;
    }

    .suggestion-title {
        color: #f0ece5;
        font-weight: 600;
        margin-bottom: 4px;
    }

    .suggestion-text {
        color: #817b72;
        font-size: 12px;
    }

    /* ======================================================
       SOURCE CHIP
       ====================================================== */

    .source-chip {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        background: #292722;
        border: 1px solid #3b3831;
        border-radius: 8px;
        padding: 7px 10px;
        color: #aaa39a;
        font-size: 12px;
        margin-bottom: 20px;
    }

    .source-chip-icon {
        color: #f0a45d;
    }

    /* ======================================================
       CHAT MESSAGES
       ====================================================== */

    [data-testid="stChatMessage"] {
        background: transparent !important;
        border: none !important;
        padding-top: 12px;
        padding-bottom: 12px;
    }

    [data-testid="stChatMessageContent"] {
        color: #e9e4dc;
        font-size: 15px;
        line-height: 1.7;
    }

    /* Hide default avatar background styling */
    [data-testid="stChatMessageAvatarUser"],
    [data-testid="stChatMessageAvatarAssistant"] {
        background: #2c2924 !important;
        color: #f0a45d !important;
    }

    /* ======================================================
       CHAT INPUT
       ====================================================== */

    [data-testid="stChatInput"] {
        background: #292722 !important;
        border: 1px solid #4a463e !important;
        border-radius: 15px !important;
        box-shadow: 0 5px 25px rgba(0, 0, 0, 0.22);
    }

    [data-testid="stChatInput"]:focus-within {
        border-color: #6b6459 !important;
    }

    [data-testid="stChatInput"] textarea {
        background: transparent !important;
        color: #f4f1ea !important;
        font-size: 14px !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: #777169 !important;
    }

    [data-testid="stChatInput"] button {
        background: #f0a45d !important;
        color: #171614 !important;
        border-radius: 8px !important;
    }

    /* ======================================================
       SOURCE INFO
       ====================================================== */

    .source-info {
        display: flex;
        align-items: center;
        gap: 8px;
        color: #8f8980;
        font-size: 12px;
        margin-top: 4px;
        margin-bottom: 15px;
    }

    /* ======================================================
       LOGIN
       ====================================================== */

    .login-container {
        max-width: 450px;
        margin: 100px auto 0 auto;
        text-align: center;
    }

    .login-star {
        color: #f0a45d;
        font-size: 40px;
        margin-bottom: 15px;
    }

    .login-title {
        color: #f4f1ea;
        font-size: 32px;
        font-weight: 700;
    }

    .login-subtitle {
        color: #8f8980;
        font-size: 14px;
        margin-top: 8px;
        margin-bottom: 28px;
    }

    /* ======================================================
       INPUTS
       ====================================================== */

    .stTextInput input {
        background: #292722 !important;
        color: #f4f1ea !important;
        border: 1px solid #3b3831 !important;
        border-radius: 8px !important;
    }

    .stTextInput input:focus {
        border-color: #6b6459 !important;
    }

    /* ======================================================
       FILE UPLOADER
       ====================================================== */

    [data-testid="stFileUploader"] {
        background: #211f1c;
        border: 1px solid #302e2a;
        border-radius: 9px;
        padding: 5px;
    }

    /* ======================================================
       EXPANDER
       ====================================================== */

    [data-testid="stExpander"] {
        background: #211f1c;
        border: 1px solid #302e2a;
        border-radius: 9px;
    }

    /* ======================================================
       HIDE TOKEN CAPTIONS
       ====================================================== */

    .token-caption {
        display: none;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# JSON STORAGE
# ============================================================

def load_json(filename, default):
    if not os.path.exists(filename):
        return default

    try:
        with open(
            filename,
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    except Exception:
        return default


def save_json(filename, data):
    try:
        with open(
            filename,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                data,
                file,
                indent=2,
            )

    except Exception:
        pass


# ============================================================
# USERS
# ============================================================

def get_users():
    return load_json(
        USERS_FILE,
        {},
    )


def save_users(users):
    save_json(
        USERS_FILE,
        users,
    )


def create_user(
    username,
    password,
):
    users = get_users()

    username = username.strip()

    if not username:
        return False, "Please enter a username."

    if not password:
        return False, "Please enter a password."

    if username in users:
        return False, "Username already exists."

    users[username] = {
        "password": password,
        "created_at": datetime.now().isoformat(),
    }

    save_users(users)

    return True, "Account created successfully."


def authenticate_user(
    username,
    password,
):
    users = get_users()

    if username not in users:
        return False

    return (
        users[username].get("password")
        == password
    )


# ============================================================
# INTERNAL USAGE
# ============================================================

def get_usage(username):

    usage = load_json(
        USAGE_FILE,
        {},
    )

    if username not in usage:

        usage[username] = {
            "requests": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }

        save_json(
            USAGE_FILE,
            usage,
        )

    return usage[username]


def update_usage(
    username,
    input_tokens,
    output_tokens,
):

    usage = load_json(
        USAGE_FILE,
        {},
    )

    if username not in usage:

        usage[username] = {
            "requests": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }

    usage[username]["requests"] += 1

    usage[username]["input_tokens"] += (
        input_tokens
    )

    usage[username]["output_tokens"] += (
        output_tokens
    )

    usage[username]["total_tokens"] += (
        input_tokens + output_tokens
    )

    save_json(
        USAGE_FILE,
        usage,
    )


# ============================================================
# CHAT STORAGE
# ============================================================

def get_all_chats():
    return load_json(
        CHATS_FILE,
        {},
    )


def save_all_chats(chats):
    save_json(
        CHATS_FILE,
        chats,
    )


def create_new_chat():

    st.session_state.current_chat_id = (
        str(uuid.uuid4())
    )

    st.session_state.messages = []

    st.session_state.retriever = None

    st.session_state.document_text = ""

    st.session_state.document_name = ""

    st.session_state.source_type = ""

    st.session_state.last_input_tokens = 0

    st.session_state.last_output_tokens = 0

    st.session_state.last_total_tokens = 0


def save_current_chat():

    username = st.session_state.username

    chat_id = (
        st.session_state.current_chat_id
    )

    if not username or not chat_id:
        return

    chats = get_all_chats()

    if username not in chats:
        chats[username] = {}

    title = "New chat"

    for message in st.session_state.messages:

        if message.get("role") == "user":

            title = (
                message
                .get("content", "")
                .strip()
            )

            if len(title) > 45:
                title = (
                    title[:45]
                    + "..."
                )

            break

    chats[username][chat_id] = {

        "title": title,

        "messages": (
            st.session_state.messages
        ),

        "document_name": (
            st.session_state.document_name
        ),

        "document_text": (
            st.session_state.document_text
        ),

        "source_type": (
            st.session_state.source_type
        ),

        "updated_at": (
            datetime.now().isoformat()
        ),
    }

    save_all_chats(chats)


def load_chat(chat_id):

    username = (
        st.session_state.username
    )

    chats = get_all_chats()

    if username not in chats:
        return

    if chat_id not in chats[username]:
        return

    chat = chats[username][chat_id]

    st.session_state.current_chat_id = (
        chat_id
    )

    st.session_state.messages = (
        chat.get(
            "messages",
            [],
        )
    )

    st.session_state.document_name = (
        chat.get(
            "document_name",
            "",
        )
    )

    st.session_state.document_text = (
        chat.get(
            "document_text",
            "",
        )
    )

    st.session_state.source_type = (
        chat.get(
            "source_type",
            "",
        )
    )

    st.session_state.last_input_tokens = 0

    st.session_state.last_output_tokens = 0

    st.session_state.last_total_tokens = 0

    if st.session_state.document_text:

        try:

            st.session_state.retriever = (
                create_retriever(
                    st.session_state.document_text
                )
            )

        except Exception:

            st.session_state.retriever = None

    else:

        st.session_state.retriever = None


def delete_chat(chat_id):

    username = (
        st.session_state.username
    )

    chats = get_all_chats()

    if (
        username in chats
        and chat_id in chats[username]
    ):

        del chats[username][chat_id]

        save_all_chats(chats)

    if (
        st.session_state.current_chat_id
        == chat_id
    ):

        create_new_chat()


# ============================================================
# SESSION STATE
# ============================================================

defaults = {

    "logged_in": False,

    "username": "",

    "current_chat_id": None,

    "messages": [],

    "retriever": None,

    "document_text": "",

    "document_name": "",

    "source_type": "",

    "last_input_tokens": 0,

    "last_output_tokens": 0,

    "last_total_tokens": 0,

    "document_processed_name": "",

}


for key, value in defaults.items():

    if key not in st.session_state:

        st.session_state[key] = value


# ============================================================
# GREETING
# ============================================================

def get_greeting():

    hour = datetime.now().hour

    if hour < 12:
        return "Good Morning"

    if hour < 18:
        return "Good Afternoon"

    return "Good Evening"


# ============================================================
# TOKEN COUNT
# ============================================================

def count_tokens(
    tokenizer,
    text,
):

    try:

        return len(
            tokenizer.encode(
                text,
                add_special_tokens=True,
            )
        )

    except Exception:

        return 0


# ============================================================
# MODEL
# ============================================================

@st.cache_resource
def load_model():

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_NAME
        )
    )

    model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL_NAME
        )
    )

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    return tokenizer, generator


# ============================================================
# EMBEDDINGS
# ============================================================

@st.cache_resource
def load_embeddings():

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )


# ============================================================
# TEXT SPLITTER
# ============================================================

def split_text(
    text,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP,
):

    if not text:
        return []

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    chunks = []

    start = 0

    while start < len(text):

        end = (
            start
            + chunk_size
        )

        chunk = text[start:end]

        if chunk.strip():
            chunks.append(
                chunk.strip()
            )

        if end >= len(text):
            break

        start = (
            end
            - overlap
        )

    return chunks


# ============================================================
# CREATE RETRIEVER
# ============================================================

def create_retriever(text):

    chunks = split_text(text)

    if not chunks:
        return None

    documents = [
        Document(
            page_content=chunk
        )
        for chunk in chunks
    ]

    embeddings = load_embeddings()

    vectorstore = (
        FAISS.from_documents(
            documents,
            embeddings,
        )
    )

    return vectorstore.as_retriever(
        search_kwargs={
            "k": RETRIEVER_K
        }
    )


# ============================================================
# PDF READER
# ============================================================

def read_pdf(uploaded_file):

    reader = PdfReader(
        uploaded_file
    )

    pages = []

    for page in reader.pages:

        try:

            text = page.extract_text()

            if text:
                pages.append(text)

        except Exception:
            continue

    return "\n".join(pages)


# ============================================================
# DOCX READER
# ============================================================

def read_docx(uploaded_file):

    document = DocxDocument(
        uploaded_file
    )

    paragraphs = []

    for paragraph in document.paragraphs:

        text = paragraph.text.strip()

        if text:

            paragraphs.append(text)

    return "\n".join(
        paragraphs
    )


# ============================================================
# XLSX READER
# ============================================================

def read_xlsx(uploaded_file):

    excel_file = pd.ExcelFile(
        uploaded_file
    )

    parts = []

    for sheet_name in (
        excel_file.sheet_names
    ):

        try:

            dataframe = pd.read_excel(
                excel_file,
                sheet_name=sheet_name,
            )

            parts.append(
                f"Sheet: {sheet_name}"
            )

            parts.append(
                dataframe.to_string(
                    index=False
                )
            )

        except Exception:
            continue

    return "\n".join(parts)


# ============================================================
# TXT READER
# ============================================================

def read_txt(uploaded_file):

    try:

        return (
            uploaded_file
            .read()
            .decode("utf-8")
        )

    except Exception:

        uploaded_file.seek(0)

        return (
            uploaded_file
            .read()
            .decode("latin-1")
        )


# ============================================================
# WEBSITE READER
# ============================================================

def read_website(url):

    headers = {

        "User-Agent":
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "Chrome/120.0 Safari/537.36"

    }

    response = requests.get(
        url,
        headers=headers,
        timeout=20,
    )

    response.raise_for_status()

    soup = BeautifulSoup(
        response.text,
        "html.parser",
    )

    for element in soup(
        [
            "script",
            "style",
            "noscript",
            "header",
            "footer",
            "nav",
        ]
    ):

        element.decompose()

    return soup.get_text(
        separator=" ",
        strip=True,
    )


# ============================================================
# DOCUMENT PROCESSOR
# ============================================================

def process_document(
    uploaded_file
):

    filename = (
        uploaded_file.name.lower()
    )

    if filename.endswith(".pdf"):
        return read_pdf(
            uploaded_file
        )

    if filename.endswith(".docx"):
        return read_docx(
            uploaded_file
        )

    if filename.endswith(".xlsx"):
        return read_xlsx(
            uploaded_file
        )

    if filename.endswith(".txt"):
        return read_txt(
            uploaded_file
        )

    return ""


# ============================================================
# PROCESS SOURCE
# ============================================================

def set_document_source(
    text,
    name,
    source_type,
):

    if not text.strip():

        return False

    retriever = create_retriever(
        text
    )

    st.session_state.document_text = text

    st.session_state.document_name = name

    st.session_state.source_type = source_type

    st.session_state.retriever = retriever

    st.session_state.document_processed_name = name

    save_current_chat()

    return True


# ============================================================
# LOGIN PAGE
# ============================================================

def show_login_page():

    st.html(
        """
        <div class="login-container">

            <div class="login-star">
                ✦
            </div>

            <div class="login-title">
                AI Document Intelligence
            </div>

            <div class="login-subtitle">
                Your documents. Your questions. One intelligent workspace.
            </div>

        </div>
        """
    )

    left, center, right = st.columns(
        [1, 1.4, 1]
    )

    with center:

        login_tab, signup_tab = st.tabs(
            [
                "Sign in",
                "Create account",
            ]
        )

        # ====================================================
        # LOGIN
        # ====================================================

        with login_tab:

            username = st.text_input(
                "Username",
                key="login_username",
            )

            password = st.text_input(
                "Password",
                type="password",
                key="login_password",
            )

            if st.button(
                "Sign in",
                use_container_width=True,
                key="sign_in_button",
            ):

                if authenticate_user(
                    username,
                    password,
                ):

                    st.session_state.logged_in = True

                    st.session_state.username = (
                        username
                    )

                    create_new_chat()

                    st.rerun()

                else:

                    st.error(
                        "Invalid username or password."
                    )

        # ====================================================
        # CREATE ACCOUNT
        # ====================================================

        with signup_tab:

            username = st.text_input(
                "Username",
                key="signup_username",
            )

            password = st.text_input(
                "Password",
                type="password",
                key="signup_password",
            )

            confirm = st.text_input(
                "Confirm password",
                type="password",
                key="signup_confirm",
            )

            if st.button(
                "Create account",
                use_container_width=True,
                key="create_account",
            ):

                if password != confirm:

                    st.error(
                        "Passwords do not match."
                    )

                else:

                    success, message = (
                        create_user(
                            username,
                            password,
                        )
                    )

                    if success:

                        st.success(
                            message
                        )

                    else:

                        st.error(
                            message
                        )


# ============================================================
# SIDEBAR
# ============================================================

def show_sidebar():

    with st.sidebar:

        # ====================================================
        # BRAND
        # ====================================================

        st.html(
            """
            <div class="sidebar-logo">
                <span class="sidebar-logo-star">
                    ✦
                </span>
                AI Document Intelligence
            </div>

            <div class="sidebar-subtitle">
                Your private document workspace
            </div>
            """
        )

        # ====================================================
        # NEW CHAT
        # ====================================================

        if st.button(
            "＋  New chat",
            use_container_width=True,
            key="new_chat_button",
        ):

            create_new_chat()

            st.rerun()

        # ====================================================
        # SOURCE
        # ====================================================

        st.html(
            """
            <div class="sidebar-divider"></div>

            <div class="sidebar-section-label">
                KNOWLEDGE
            </div>
            """
        )

        with st.expander(
            "Add a document",
            expanded=False,
        ):

            uploaded_file = st.file_uploader(
                "PDF, DOCX, XLSX or TXT",
                type=[
                    "pdf",
                    "docx",
                    "xlsx",
                    "txt",
                ],
                key="sidebar_uploader",
            )

            if uploaded_file is not None:

                if (
                    st.session_state.document_processed_name
                    != uploaded_file.name
                ):

                    with st.spinner(
                        "Reading document..."
                    ):

                        try:

                            text = process_document(
                                uploaded_file
                            )

                            if text.strip():

                                set_document_source(
                                    text,
                                    uploaded_file.name,
                                    "Document",
                                )

                                st.success(
                                    "Document ready."
                                )

                            else:

                                st.error(
                                    "Could not extract text."
                                )

                        except Exception as error:

                            st.error(
                                f"Could not read document: {error}"
                            )

        with st.expander(
            "Add a website",
            expanded=False,
        ):

            website_url = st.text_input(
                "Website URL",
                placeholder="https://example.com",
                key="sidebar_website_url",
            )

            if st.button(
                "Load website",
                use_container_width=True,
                key="load_website_button",
            ):

                if not website_url.strip():

                    st.warning(
                        "Enter a website URL."
                    )

                else:

                    with st.spinner(
                        "Reading website..."
                    ):

                        try:

                            text = read_website(
                                website_url.strip()
                            )

                            if text.strip():

                                set_document_source(
                                    text,
                                    website_url.strip(),
                                    "Website",
                                )

                                st.success(
                                    "Website ready."
                                )

                            else:

                                st.error(
                                    "Could not extract website text."
                                )

                        except Exception as error:

                            st.error(
                                f"Could not load website: {error}"
                            )

        # ====================================================
        # CURRENT SOURCE
        # ====================================================

        if st.session_state.document_name:

            safe_name = html.escape(
                st.session_state.document_name
            )

            icon = (
                "🌐"
                if st.session_state.source_type
                == "Website"
                else "📄"
            )

            st.html(
                f"""
                <div style="
                    margin-top:10px;
                    padding:9px 10px;
                    background:#211f1c;
                    border:1px solid #302e2a;
                    border-radius:8px;
                    color:#a8a198;
                    font-size:11px;
                    line-height:1.4;
                ">
                    <span style="color:#f0a45d;">
                        {icon}
                    </span>
                    &nbsp;
                    {safe_name}
                </div>
                """
            )

        # ====================================================
        # CONVERSATIONS
        # ====================================================

        st.html(
            """
            <div class="sidebar-divider"></div>

            <div class="sidebar-section-label">
                CHATS
            </div>
            """
        )

        chats = get_all_chats()

        user_chats = chats.get(
            st.session_state.username,
            {},
        )

        sorted_chats = sorted(
            user_chats.items(),
            key=lambda item: item[1].get(
                "updated_at",
                "",
            ),
            reverse=True,
        )

        if not sorted_chats:

            st.html(
                """
                <div class="history-empty">
                    Your conversations will appear here.
                </div>
                """
            )

        else:

            for chat_id, chat in sorted_chats:

                title = chat.get(
                    "title",
                    "New chat",
                )

                if len(title) > 30:

                    title = (
                        title[:30]
                        + "..."
                    )

                col1, col2 = st.columns(
                    [5, 1]
                )

                with col1:

                    if st.button(
                        title,
                        key=f"open_{chat_id}",
                    ):

                        load_chat(
                            chat_id
                        )

                        st.rerun()

                with col2:

                    if st.button(
                        "×",
                        key=f"delete_{chat_id}",
                    ):

                        delete_chat(
                            chat_id
                        )

                        st.rerun()

        # ====================================================
        # BOTTOM
        # ====================================================

        st.html(
            """
            <div class="sidebar-divider"></div>
            """
        )

        if st.button(
            "Sign out",
            use_container_width=True,
            key="sign_out_button",
        ):

            st.session_state.logged_in = False

            st.session_state.username = ""

            st.session_state.current_chat_id = None

            st.session_state.messages = []

            st.session_state.retriever = None

            st.session_state.document_text = ""

            st.session_state.document_name = ""

            st.rerun()


# ============================================================
# WELCOME SCREEN
# ============================================================

def show_welcome():

    greeting = get_greeting()

    st.html(
        f"""
        <div class="welcome">

            <div class="welcome-star">
                ✦
            </div>

            <div class="welcome-title">
                {greeting}
            </div>

            <div class="welcome-subtitle">
                What would you like to know?
            </div>

        </div>
        """
    )

    col1, col2 = st.columns(
        2
    )

    with col1:

        st.html(
            """
            <div class="suggestion-card">
                <div class="suggestion-title">
                    Summarize a document
                </div>

                <div class="suggestion-text">
                    Get the main points in simple language.
                </div>
            </div>
            """
        )

        st.markdown(
            "<div style='height:10px'></div>",
            unsafe_allow_html=True,
        )

        st.html(
            """
            <div class="suggestion-card">
                <div class="suggestion-title">
                    Find important information
                </div>

                <div class="suggestion-text">
                    Ask questions about specific details.
                </div>
            </div>
            """
        )

    with col2:

        st.html(
            """
            <div class="suggestion-card">
                <div class="suggestion-title">
                    Explain something simply
                </div>

                <div class="suggestion-text">
                    Turn complex document content into clear answers.
                </div>
            </div>
            """
        )

        st.markdown(
            "<div style='height:10px'></div>",
            unsafe_allow_html=True,
        )

        st.html(
            """
            <div class="suggestion-card">
                <div class="suggestion-title">
                    Compare information
                </div>

                <div class="suggestion-text">
                    Ask questions across the uploaded content.
                </div>
            </div>
            """
        )


# ============================================================
# MAIN CHAT
# ============================================================

def show_main_chat():

    # ========================================================
    # CURRENT SOURCE
    # ========================================================

    if st.session_state.document_name:

        safe_name = html.escape(
            st.session_state.document_name
        )

        icon = (
            "🌐"
            if st.session_state.source_type
            == "Website"
            else "📄"
        )

        st.html(
            f"""
            <div class="source-chip">

                <span class="source-chip-icon">
                    {icon}
                </span>

                <span>
                    {safe_name}
                </span>

            </div>
            """
        )

    # ========================================================
    # CHAT HISTORY
    # ========================================================

    if not st.session_state.messages:

        show_welcome()

    else:

        for message in (
            st.session_state.messages
        ):

            role = message.get(
                "role"
            )

            content = message.get(
                "content",
                "",
            )

            if role == "user":

                with st.chat_message(
                    "user"
                ):

                    st.markdown(
                        content
                    )

            elif role == "assistant":

                with st.chat_message(
                    "assistant"
                ):

                    st.markdown(
                        content
                    )

    # ========================================================
    # CHAT INPUT
    # ========================================================

    question = st.chat_input(
        "Ask anything about your document..."
    )

    if question:

        # ====================================================
        # USER MESSAGE
        # ====================================================

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        # ====================================================
        # SOURCE CHECK
        # ====================================================

        if (
            st.session_state.retriever
            is None
        ):

            answer = (
                "Please add a document or website "
                "from the sidebar first."
            )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "tokens": 0,
                }
            )

            save_current_chat()

            st.rerun()

        # ====================================================
        # GENERATION
        # ====================================================

        with st.chat_message(
            "assistant"
        ):

            with st.spinner(
                "Thinking..."
            ):

                try:

                    tokenizer, generator = (
                        load_model()
                    )

                    # ----------------------------------------
                    # RETRIEVE
                    # ----------------------------------------

                    documents = (
                        st.session_state
                        .retriever
                        .invoke(
                            question
                        )
                    )

                    context_parts = []

                    for document in documents:

                        content = (
                            document.page_content
                        )

                        if len(content) > 900:

                            content = (
                                content[:900]
                                + "..."
                            )

                        context_parts.append(
                            content
                        )

                    context = (
                        "\n\n".join(
                            context_parts
                        )
                    )

                    # ----------------------------------------
                    # SMALLER PROMPT
                    # ----------------------------------------

                    prompt = f"""
You answer questions using the provided document context.

Be concise and clear.
Do not invent information.
If the answer is not in the context, say:
"I could not find that information in the document."

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

                    # ----------------------------------------
                    # INPUT TOKENS
                    # ----------------------------------------

                    input_tokens = (
                        count_tokens(
                            tokenizer,
                            prompt,
                        )
                    )

                    # ----------------------------------------
                    # GENERATE
                    # ----------------------------------------

                    result = generator(
                        prompt,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=False,
                        return_full_text=False,
                        pad_token_id=(
                            tokenizer.eos_token_id
                        ),
                    )

                    answer = (
                        result[0]
                        .get(
                            "generated_text",
                            "",
                        )
                        .strip()
                    )

                    # ----------------------------------------
                    # CLEAN ANSWER
                    # ----------------------------------------

                    if "ANSWER:" in answer:

                        answer = (
                            answer
                            .split(
                                "ANSWER:",
                                1,
                            )[1]
                            .strip()
                        )

                    if not answer:

                        answer = (
                            "I could not generate "
                            "an answer from the document."
                        )

                    # ----------------------------------------
                    # OUTPUT TOKENS
                    # ----------------------------------------

                    output_tokens = (
                        count_tokens(
                            tokenizer,
                            answer,
                        )
                    )

                    total_tokens = (
                        input_tokens
                        + output_tokens
                    )

                    # ----------------------------------------
                    # INTERNAL TOKEN DATA
                    # ----------------------------------------

                    st.session_state.last_input_tokens = (
                        input_tokens
                    )

                    st.session_state.last_output_tokens = (
                        output_tokens
                    )

                    st.session_state.last_total_tokens = (
                        total_tokens
                    )

                    # ----------------------------------------
                    # DISPLAY
                    # ----------------------------------------

                    st.markdown(
                        answer
                    )

                    # ----------------------------------------
                    # SAVE MESSAGE
                    # ----------------------------------------

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "tokens": total_tokens,
                        }
                    )

                    # ----------------------------------------
                    # INTERNAL USAGE
                    # ----------------------------------------

                    update_usage(
                        st.session_state.username,
                        input_tokens,
                        output_tokens,
                    )

                    # ----------------------------------------
                    # SAVE CHAT
                    # ----------------------------------------

                    save_current_chat()

                except Exception as error:

                    answer = (
                        "I ran into a problem while "
                        "processing your question."
                    )

                    st.error(
                        answer
                    )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": (
                                answer
                                + "\n\n"
                                + str(error)
                            ),
                            "tokens": 0,
                        }
                    )

                    save_current_chat()

        st.rerun()


# ============================================================
# MAIN APPLICATION
# ============================================================

def show_application():

    show_sidebar()

    show_main_chat()


# ============================================================
# START
# ============================================================

if not st.session_state.logged_in:

    show_login_page()

else:

    show_application()

import os
import json
import uuid
import re
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
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Document Intelligence",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# FILES / MODELS
# ============================================================

USERS_FILE = "users.json"
CHATS_FILE = "chats.json"
USAGE_FILE = "usage.json"

MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #1f1e1b;
        color: #f4f1ea;
    }

    [data-testid="stAppViewContainer"] {
        background-color: #1f1e1b;
    }

    [data-testid="stHeader"] {
        background-color: #1f1e1b;
    }

    [data-testid="stSidebar"] {
        background-color: #171614;
        border-right: 1px solid #302e2a;
    }

    [data-testid="stSidebar"] > div:first-child {
        background-color: #171614;
    }

    .sidebar-brand {
        font-size: 22px;
        font-weight: 700;
        color: #f4f1ea;
        margin-bottom: 5px;
    }

    .sidebar-subtitle {
        font-size: 12px;
        color: #9f9990;
        margin-bottom: 20px;
    }

    .stButton > button {
        width: 100%;
        background-color: #292722;
        color: #f4f1ea;
        border: 1px solid #3a3731;
        border-radius: 8px;
        min-height: 42px;
        font-size: 14px;
    }

    .stButton > button:hover {
        border-color: #f0a45d;
        color: #f0a45d;
    }

    [data-testid="stChatMessage"] {
        background-color: transparent;
    }

    [data-testid="stChatMessageContent"] {
        color: #f4f1ea;
    }

    [data-testid="stChatInput"] {
        background-color: #292722;
        border: 1px solid #454139;
        border-radius: 14px;
    }

    [data-testid="stChatInput"] textarea {
        background-color: transparent !important;
        color: #f4f1ea !important;
        font-size: 15px;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: #817c73 !important;
    }

    [data-testid="stChatInput"] button {
        background-color: #f0a45d !important;
        color: #171614 !important;
        border-radius: 8px;
    }

    .welcome-container {
        text-align: center;
        margin-top: 110px;
        margin-bottom: 35px;
    }

    .welcome-star {
        font-size: 42px;
        color: #f0a45d;
        margin-bottom: 14px;
    }

    .welcome-title {
        font-size: 32px;
        font-weight: 700;
        color: #f4f1ea;
    }

    .welcome-text {
        color: #9f9990;
        margin-top: 8px;
        font-size: 15px;
    }

    .token-bar {
        width: 100%;
        display: flex;
        justify-content: flex-end;
        margin-top: 8px;
        margin-bottom: 8px;
    }

    .token-badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 8px;
        background-color: #292722;
        border: 1px solid #3a3731;
        color: #9f9990;
        font-size: 12px;
    }

    .token-number {
        color: #f0a45d;
        font-weight: 600;
    }

    .token-limit {
        color: #f4f1ea;
        font-weight: 600;
    }

    .usage-card {
        background-color: #211f1c;
        border: 1px solid #302e2a;
        border-radius: 10px;
        padding: 14px;
        margin-top: 20px;
    }

    .usage-header {
        font-size: 14px;
        font-weight: 700;
        color: #f4f1ea;
        margin-bottom: 12px;
    }

    .usage-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        color: #9f9990;
        font-size: 12px;
        margin: 9px 0;
    }

    .usage-number {
        color: #f4f1ea;
        font-weight: 600;
    }

    .usage-total {
        color: #f0a45d;
        font-weight: 700;
    }

    .source-info {
        background-color: #292722;
        border: 1px solid #3a3731;
        border-radius: 8px;
        padding: 10px 12px;
        color: #9f9990;
        font-size: 12px;
        margin-bottom: 12px;
    }

    .login-title {
        text-align: center;
        font-size: 34px;
        font-weight: 700;
        color: #f4f1ea;
        margin-top: 60px;
    }

    .login-subtitle {
        text-align: center;
        color: #9f9990;
        margin-bottom: 30px;
    }

    [data-testid="stFileUploader"] {
        background-color: #292722;
        border-radius: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# JSON HELPERS
# ============================================================

def load_json(filename, default):
    if not os.path.exists(filename):
        return default

    try:
        with open(filename, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return default


def save_json(filename, data):
    try:
        with open(filename, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2)
    except Exception:
        pass


# ============================================================
# USER FUNCTIONS
# ============================================================

def get_users():
    return load_json(USERS_FILE, {})


def save_users(users):
    save_json(USERS_FILE, users)


def create_user(username, password):
    users = get_users()

    if username in users:
        return False, "Username already exists."

    users[username] = {
        "password": password,
        "created_at": datetime.now().isoformat(),
    }

    save_users(users)

    return True, "Account created successfully."


def authenticate_user(username, password):
    users = get_users()

    if username not in users:
        return False

    return users[username].get("password") == password


# ============================================================
# USAGE FUNCTIONS
# ============================================================

def get_usage(username):
    usage = load_json(USAGE_FILE, {})

    if username not in usage:
        usage[username] = {
            "requests": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }

        save_json(USAGE_FILE, usage)

    return usage[username]


def update_usage(username, input_tokens, output_tokens):
    usage = load_json(USAGE_FILE, {})

    if username not in usage:
        usage[username] = {
            "requests": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }

    usage[username]["requests"] += 1
    usage[username]["input_tokens"] += input_tokens
    usage[username]["output_tokens"] += output_tokens
    usage[username]["total_tokens"] += (
        input_tokens + output_tokens
    )

    save_json(USAGE_FILE, usage)


# ============================================================
# CHAT FUNCTIONS
# ============================================================

def get_all_chats():
    return load_json(CHATS_FILE, {})


def save_all_chats(chats):
    save_json(CHATS_FILE, chats)


def create_new_chat():
    st.session_state.current_chat_id = str(uuid.uuid4())
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
    chat_id = st.session_state.current_chat_id

    if not username or not chat_id:
        return

    chats = get_all_chats()

    if username not in chats:
        chats[username] = {}

    title = "New Chat"

    for message in st.session_state.messages:
        if message.get("role") == "user":
            title = message.get("content", "").strip()

            if len(title) > 45:
                title = title[:45] + "..."

            break

    chats[username][chat_id] = {
        "title": title,
        "messages": st.session_state.messages,
        "document_name": st.session_state.document_name,
        "document_text": st.session_state.document_text,
        "source_type": st.session_state.source_type,
        "updated_at": datetime.now().isoformat(),
    }

    save_all_chats(chats)


def load_chat(chat_id):
    username = st.session_state.username

    chats = get_all_chats()

    if username not in chats:
        return

    if chat_id not in chats[username]:
        return

    chat = chats[username][chat_id]

    st.session_state.current_chat_id = chat_id
    st.session_state.messages = chat.get("messages", [])
    st.session_state.document_name = chat.get(
        "document_name",
        "",
    )
    st.session_state.document_text = chat.get(
        "document_text",
        "",
    )
    st.session_state.source_type = chat.get(
        "source_type",
        "",
    )

    st.session_state.last_input_tokens = 0
    st.session_state.last_output_tokens = 0
    st.session_state.last_total_tokens = 0

    if st.session_state.document_text:

        try:
            st.session_state.retriever = create_retriever(
                st.session_state.document_text
            )
        except Exception:
            st.session_state.retriever = None

    else:
        st.session_state.retriever = None


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

def count_tokens(tokenizer, text):

    try:

        tokens = tokenizer.encode(
            text,
            add_special_tokens=True,
        )

        return len(tokens)

    except Exception:

        return 0


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME
    )

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    return tokenizer, generator


# ============================================================
# LOAD EMBEDDINGS
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
    chunk_size=1000,
    overlap=200,
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

        end = start + chunk_size

        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk.strip())

        if end >= len(text):
            break

        start = end - overlap

    return chunks


# ============================================================
# CREATE RETRIEVER
# ============================================================

def create_retriever(text):

    chunks = split_text(
        text,
        1000,
        200,
    )

    if not chunks:
        return None

    documents = [
        Document(page_content=chunk)
        for chunk in chunks
    ]

    embeddings = load_embeddings()

    vectorstore = FAISS.from_documents(
        documents,
        embeddings,
    )

    return vectorstore.as_retriever(
        search_kwargs={
            "k": 2
        }
    )


# ============================================================
# DOCUMENT READERS
# ============================================================

def read_pdf(uploaded_file):

    reader = PdfReader(uploaded_file)

    pages = []

    for page in reader.pages:

        try:

            text = page.extract_text()

            if text:
                pages.append(text)

        except Exception:
            pass

    return "\n".join(pages)


def read_docx(uploaded_file):

    document = DocxDocument(uploaded_file)

    paragraphs = []

    for paragraph in document.paragraphs:

        if paragraph.text.strip():
            paragraphs.append(
                paragraph.text
            )

    return "\n".join(paragraphs)


def read_xlsx(uploaded_file):

    excel_file = pd.ExcelFile(
        uploaded_file
    )

    parts = []

    for sheet in excel_file.sheet_names:

        try:

            dataframe = pd.read_excel(
                excel_file,
                sheet_name=sheet,
            )

            parts.append(
                f"Sheet: {sheet}"
            )

            parts.append(
                dataframe.to_string(
                    index=False
                )
            )

        except Exception:
            pass

    return "\n".join(parts)


def read_txt(uploaded_file):

    try:

        return uploaded_file.read().decode(
            "utf-8"
        )

    except Exception:

        uploaded_file.seek(0)

        return uploaded_file.read().decode(
            "latin-1"
        )


def read_website(url):

    headers = {
        "User-Agent": (
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "Chrome/120.0 Safari/537.36"
        )
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


def process_document(uploaded_file):

    name = uploaded_file.name.lower()

    if name.endswith(".pdf"):
        return read_pdf(uploaded_file)

    if name.endswith(".docx"):
        return read_docx(uploaded_file)

    if name.endswith(".xlsx"):
        return read_xlsx(uploaded_file)

    if name.endswith(".txt"):
        return read_txt(uploaded_file)

    return ""


# ============================================================
# LOGIN PAGE
# ============================================================

def show_login_page():

    st.html(
        """
        <div style="
            text-align:center;
            margin-top:60px;
        ">

            <div style="
                font-size:42px;
                color:#f0a45d;
                margin-bottom:14px;
            ">
                ✦
            </div>

            <div style="
                font-size:34px;
                font-weight:700;
                color:#f4f1ea;
            ">
                AI Document Intelligence
            </div>

            <div style="
                color:#9f9990;
                margin-top:8px;
                font-size:15px;
            ">
                Ask questions about your documents and websites.
            </div>

        </div>
        """
    )

    left, center, right = st.columns(
        [1, 2, 1]
    )

    with center:

        login_tab, create_tab = st.tabs(
            [
                "Login",
                "Create Account",
            ]
        )

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
                "Login",
                key="login_button",
            ):

                if authenticate_user(
                    username,
                    password,
                ):

                    st.session_state.logged_in = True
                    st.session_state.username = username

                    create_new_chat()

                    st.rerun()

                else:

                    st.error(
                        "Invalid username or password."
                    )

        with create_tab:

            new_username = st.text_input(
                "Username",
                key="new_username",
            )

            new_password = st.text_input(
                "Password",
                type="password",
                key="new_password",
            )

            confirm_password = st.text_input(
                "Confirm Password",
                type="password",
                key="confirm_password",
            )

            if st.button(
                "Create Account",
                key="create_button",
            ):

                if not new_username or not new_password:

                    st.error(
                        "Please enter username and password."
                    )

                elif new_password != confirm_password:

                    st.error(
                        "Passwords do not match."
                    )

                else:

                    success, message = create_user(
                        new_username,
                        new_password,
                    )

                    if success:
                        st.success(message)
                    else:
                        st.error(message)


# ============================================================
# SIDEBAR
# ============================================================

def show_sidebar():

    with st.sidebar:

        st.html(
            """
            <div class="sidebar-brand">
                ✦ AI Document Intelligence
            </div>

            <div class="sidebar-subtitle">
                Document & website assistant
            </div>
            """
        )

        if st.button(
            "＋  New Chat",
            key="new_chat",
        ):

            create_new_chat()

            st.rerun()

        st.html(
            """
            <div style="
                color:#9f9990;
                font-size:12px;
                margin-top:20px;
                margin-bottom:8px;
            ">
                CONVERSATIONS
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
                <div style="
                    color:#817c73;
                    font-size:12px;
                    padding:8px 0;
                ">
                    No conversations yet.
                </div>
                """
            )

        else:

            for chat_id, chat in sorted_chats:

                title = chat.get(
                    "title",
                    "New Chat",
                )

                if len(title) > 32:
                    title = title[:32] + "..."

                if st.button(
                    title,
                    key=f"conversation_{chat_id}",
                ):

                    load_chat(chat_id)

                    st.rerun()

        # ----------------------------------------------------
        # USAGE CARD
        # ----------------------------------------------------

        usage = get_usage(
            st.session_state.username
        )

        st.html(
            f"""
            <div class="usage-card">

                <div class="usage-header">
                    Token Usage
                </div>

                <div class="usage-row">
                    <span>Requests</span>
                    <span class="usage-number">
                        {usage.get("requests", 0)}
                    </span>
                </div>

                <div class="usage-row">
                    <span>Input tokens</span>
                    <span class="usage-number">
                        {usage.get("input_tokens", 0)}
                    </span>
                </div>

                <div class="usage-row">
                    <span>Output tokens</span>
                    <span class="usage-number">
                        {usage.get("output_tokens", 0)}
                    </span>
                </div>

                <div class="usage-row">
                    <span>Total tokens</span>
                    <span class="usage-total">
                        {usage.get("total_tokens", 0)}
                    </span>
                </div>

            </div>
            """
        )

        st.markdown("")

        if st.button(
            "Logout",
            key="logout",
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
# MAIN APP
# ============================================================

def show_main_app():

    show_sidebar()

    # ========================================================
    # TOP HEADER
    # ========================================================

    st.html(
        """
        <div style="
            text-align:center;
            margin-top:35px;
            margin-bottom:25px;
        ">

            <div style="
                font-size:42px;
                color:#f0a45d;
                margin-bottom:14px;
            ">
                ✦
            </div>

            <div style="
                font-size:32px;
                font-weight:700;
                color:#f4f1ea;
            ">
                AI Document Intelligence
            </div>

            <div style="
                color:#9f9990;
                margin-top:8px;
                font-size:15px;
            ">
                Ask questions about your documents and websites.
            </div>

        </div>
        """
    )

    # ========================================================
    # SOURCE TABS
    # ========================================================

    document_tab, website_tab = st.tabs(
        [
            "📄 Document",
            "🌐 Website",
        ]
    )

    with document_tab:

        uploaded_file = st.file_uploader(
            "Upload a document",
            type=[
                "pdf",
                "txt",
                "docx",
                "xlsx",
            ],
            key="document_upload",
        )

        if uploaded_file is not None:

            if (
                st.session_state.document_name
                != uploaded_file.name
            ):

                with st.spinner(
                    "Processing document..."
                ):

                    try:

                        text = process_document(
                            uploaded_file
                        )

                        if not text.strip():

                            st.error(
                                "Could not extract text from this document."
                            )

                        else:

                            retriever = create_retriever(
                                text
                            )

                            st.session_state.document_text = text
                            st.session_state.document_name = (
                                uploaded_file.name
                            )
                            st.session_state.source_type = (
                                "Document"
                            )
                            st.session_state.retriever = (
                                retriever
                            )

                            st.success(
                                f"Loaded: {uploaded_file.name}"
                            )

                            save_current_chat()

                    except Exception as error:

                        st.error(
                            f"Error processing document: {error}"
                        )

    with website_tab:

        website_url = st.text_input(
            "Enter website URL",
            placeholder="https://example.com",
            key="website_url",
        )

        if st.button(
            "Load Website",
            key="load_website",
        ):

            if not website_url.strip():

                st.warning(
                    "Please enter a website URL."
                )

            else:

                with st.spinner(
                    "Reading website..."
                ):

                    try:

                        text = read_website(
                            website_url.strip()
                        )

                        if not text.strip():

                            st.error(
                                "Could not extract text from this website."
                            )

                        else:

                            retriever = create_retriever(
                                text
                            )

                            st.session_state.document_text = text
                            st.session_state.document_name = (
                                website_url.strip()
                            )
                            st.session_state.source_type = (
                                "Website"
                            )
                            st.session_state.retriever = (
                                retriever
                            )

                            st.success(
                                "Website loaded successfully."
                            )

                            save_current_chat()

                    except Exception as error:

                        st.error(
                            f"Error loading website: {error}"
                        )

    # ========================================================
    # SOURCE INFORMATION
    # ========================================================

    if st.session_state.document_name:

        st.html(
            f"""
            <div class="source-info">
                <strong>Source:</strong>
                {st.session_state.document_name}
            </div>
            """
        )

    # ========================================================
    # CHAT HISTORY
    # ========================================================

    for message in st.session_state.messages:

        role = message.get(
            "role"
        )

        content = message.get(
            "content",
            "",
        )

        if role == "user":

            with st.chat_message("user"):
                st.markdown(content)

        elif role == "assistant":

            with st.chat_message("assistant"):

                st.markdown(content)

                tokens = message.get(
                    "tokens",
                    0,
                )

                if tokens:

                    st.caption(
                        f"{tokens} tokens"
                    )

    # ========================================================
    # WELCOME MESSAGE
    # ========================================================

    if not st.session_state.messages:

        greeting = get_greeting()

        # IMPORTANT:
        # This uses st.html().
        # There is NO raw HTML outside Python.

        st.html(
            f"""
            <div class="welcome-container">

                <div class="welcome-star">
                    ✦
                </div>

                <div class="welcome-title">
                    {greeting}
                </div>

                <div class="welcome-text">
                    What would you like to know?
                </div>

            </div>
            """
        )

    # ========================================================
    # TOKEN BADGE
    # ========================================================

    # IMPORTANT:
    # This also uses st.html().
    # There is NO raw HTML outside Python.

    st.html(
        f"""
        <div class="token-bar">

            <div class="token-badge">

                Last response:
                <span class="token-number">
                    {st.session_state.last_total_tokens}
                </span>
                tokens

                &nbsp;•&nbsp;

                Max response:
                <span class="token-limit">
                    100
                </span>
                tokens

            </div>

        </div>
        """
    )

    # ========================================================
    # CHAT INPUT
    # ========================================================

    user_question = st.chat_input(
        "Ask anything about your document..."
    )

    if user_question:

        # ----------------------------------------------------
        # USER MESSAGE
        # ----------------------------------------------------

        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_question,
            }
        )

        # ----------------------------------------------------
        # CHECK SOURCE
        # ----------------------------------------------------

        if st.session_state.retriever is None:

            answer = (
                "Please upload a document "
                "or load a website first."
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

        # ----------------------------------------------------
        # GENERATE ANSWER
        # ----------------------------------------------------

        with st.chat_message("assistant"):

            with st.spinner(
                "Thinking..."
            ):

                try:

                    tokenizer, generator = load_model()

                    # ----------------------------------------
                    # RETRIEVE
                    # ----------------------------------------

                    documents = (
                        st.session_state.retriever.invoke(
                            user_question
                        )
                    )

                    context = "\n\n".join(
                        [
                            document.page_content
                            for document in documents
                        ]
                    )

                    # ----------------------------------------
                    # PROMPT
                    # ----------------------------------------

                    prompt = f"""
You are an AI Document Intelligence assistant.

Use only the CONTEXT below to answer the question.

If the answer is not present in the context, say:

"I could not find that information in the uploaded document."

Keep the answer short and clear.

CONTEXT:
{context}

QUESTION:
{user_question}

ANSWER:
"""

                    # ----------------------------------------
                    # INPUT TOKENS
                    # ----------------------------------------

                    input_tokens = count_tokens(
                        tokenizer,
                        prompt,
                    )

                    # ----------------------------------------
                    # GENERATE
                    # ----------------------------------------

                    result = generator(
                        prompt,
                        max_new_tokens=100,
                        do_sample=False,
                        return_full_text=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                    answer = result[0][
                        "generated_text"
                    ].strip()

                    # ----------------------------------------
                    # CLEAN ANSWER
                    # ----------------------------------------

                    if "ANSWER:" in answer:

                        answer = answer.split(
                            "ANSWER:",
                            1,
                        )[1].strip()

                    # ----------------------------------------
                    # OUTPUT TOKENS
                    # ----------------------------------------

                    output_tokens = count_tokens(
                        tokenizer,
                        answer,
                    )

                    total_tokens = (
                        input_tokens
                        + output_tokens
                    )

                    # ----------------------------------------
                    # UPDATE TOKEN DISPLAY
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
                    # SHOW ANSWER
                    # ----------------------------------------

                    st.markdown(answer)

                    st.caption(
                        f"{total_tokens} tokens"
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
                    # USAGE
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

                    error_message = (
                        "Sorry, I couldn't generate "
                        "the answer.\n\n"
                        f"Error: {error}"
                    )

                    st.error(
                        error_message
                    )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": error_message,
                            "tokens": 0,
                        }
                    )

                    save_current_chat()

        st.rerun()


# ============================================================
# START APPLICATION
# ============================================================

if st.session_state.logged_in:

    show_main_app()

else:

    show_login_page()

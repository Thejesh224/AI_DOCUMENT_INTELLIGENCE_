import os
import json
import uuid
import time
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

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


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
# DARK CLAUDE-STYLE THEME
# ============================================================

st.markdown(
    """
    <style>

    /* =========================
       GLOBAL
       ========================= */

    html, body, [class*="css"] {
        font-family: Inter, -apple-system, BlinkMacSystemFont,
        "Segoe UI", sans-serif;
    }

    .stApp {
        background: #1f1e1b;
        color: #f5f5f0;
    }

    [data-testid="stAppViewContainer"] {
        background: #1f1e1b;
    }

    [data-testid="stHeader"] {
        background: #1f1e1b;
    }

    /* =========================
       SIDEBAR
       ========================= */

    section[data-testid="stSidebar"] {
        background: #171614;
        border-right: 1px solid #34322e;
    }

    section[data-testid="stSidebar"] > div {
        background: #171614;
    }

    .sidebar-brand {
        padding: 18px 4px 22px 4px;
        font-size: 19px;
        font-weight: 700;
        color: #f4f0e8;
    }

    .sidebar-brand span {
        color: #f0a45d;
    }

    .sidebar-section {
        color: #9f9a90;
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 25px;
        margin-bottom: 10px;
    }

    .usage-card {
        background: #24231f;
        border: 1px solid #3a3832;
        border-radius: 12px;
        padding: 13px;
        margin-top: 8px;
    }

    .usage-line {
        display: flex;
        justify-content: space-between;
        margin: 7px 0;
        color: #aaa59b;
        font-size: 13px;
    }

    .usage-value {
        color: #f3eee5;
        font-weight: 600;
    }

    /* =========================
       MAIN HEADER
       ========================= */

    .main-title {
        text-align: center;
        margin-top: 8px;
        font-size: 31px;
        font-weight: 700;
        color: #f4f1ea;
        letter-spacing: -0.02em;
    }

    .main-title .star {
        color: #f0a45d;
    }

    .main-subtitle {
        text-align: center;
        color: #aaa59b;
        margin-top: 5px;
        font-size: 14px;
    }

    /* =========================
       WELCOME
       ========================= */

    .welcome-area {
        text-align: center;
        padding: 105px 20px 80px 20px;
    }

    .welcome-star {
        font-size: 34px;
        color: #f0a45d;
        margin-bottom: 14px;
    }

    .welcome-title {
        font-size: 30px;
        font-weight: 650;
        color: #f5f2eb;
        margin-bottom: 10px;
    }

    .welcome-text {
        color: #aaa59b;
        font-size: 16px;
    }

    /* =========================
       SOURCE BOX
       ========================= */

    .source-label {
        color: #ddd8cf;
        font-size: 14px;
        font-weight: 600;
        margin-bottom: 6px;
    }

    /* =========================
       TOKEN BADGE
       ========================= */

    .token-bar {
        display: flex;
        justify-content: flex-end;
        align-items: center;
        margin: 5px 6px 5px 6px;
    }

    .token-badge {
        background: #292722;
        border: 1px solid #454139;
        border-radius: 8px;
        padding: 5px 10px;
        color: #aaa59b;
        font-size: 11px;
    }

    .token-badge strong {
        color: #e6e0d5;
    }

    /* =========================
       CHAT MESSAGES
       ========================= */

    [data-testid="stChatMessage"] {
        background: transparent;
    }

    [data-testid="stChatMessageContent"] {
        color: #eeeae2;
        line-height: 1.65;
    }

    /* =========================
       CHAT INPUT
       ========================= */

    [data-testid="stChatInput"] {
        background: #262521;
        border: 1px solid #48443c;
        border-radius: 16px;
    }

    [data-testid="stChatInput"] textarea {
        background: #262521 !important;
        color: #f4f0e8 !important;
        border: none !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: #888278 !important;
    }

    [data-testid="stChatInput"] button {
        background: #f0a45d !important;
        color: #1f1e1b !important;
        border-radius: 10px !important;
    }

    /* =========================
       BUTTONS
       ========================= */

    .stButton > button {
        background: #292722 !important;
        color: #eeeae2 !important;
        border: 1px solid #454139 !important;
        border-radius: 9px !important;
        min-height: 40px;
    }

    .stButton > button:hover {
        background: #34312b !important;
        border-color: #5a5449 !important;
    }

    /* =========================
       INPUTS
       ========================= */

    input,
    textarea {
        background-color: #262521 !important;
        color: #eeeae2 !important;
    }

    div[data-baseweb="input"] {
        background-color: #262521 !important;
        border-color: #48443c !important;
    }

    div[data-baseweb="select"] > div {
        background-color: #262521 !important;
        color: #eeeae2 !important;
        border-color: #48443c !important;
    }

    /* =========================
       FILE UPLOADER
       ========================= */

    [data-testid="stFileUploader"] {
        background: #262521;
        border: 1px dashed #4b463d;
        border-radius: 12px;
        padding: 8px;
    }

    /* =========================
       EXPANDER
       ========================= */

    [data-testid="stExpander"] {
        background: #24231f;
        border: 1px solid #454139;
        border-radius: 12px;
    }

    /* =========================
       ALERTS
       ========================= */

    .stAlert {
        border-radius: 10px;
    }

    /* =========================
       HIDE STREAMLIT BRANDING
       ========================= */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# FILE STORAGE
# ============================================================

USERS_FILE = "users.json"
CHATS_FILE = "chats.json"
USAGE_FILE = "usage.json"


# ============================================================
# JSON HELPERS
# ============================================================

def load_json(filename, default):
    if not os.path.exists(filename):
        return default

    try:
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def save_json(filename, data):
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


# ============================================================
# DATA
# ============================================================

users = load_json(USERS_FILE, {})
chats = load_json(CHATS_FILE, {})
usage = load_json(USAGE_FILE, {})


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
    "chat_loaded": False,
    "last_input_tokens": 0,
    "last_output_tokens": 0,
    "last_total_tokens": 0,
    "show_login": True,
    "source_mode": "Document",
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# TOKENIZER + MODEL
# ============================================================

MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

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
# EMBEDDINGS
# ============================================================

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# ============================================================
# TOKEN COUNT
# ============================================================

def count_tokens(tokenizer, text):
    if not text:
        return 0

    try:
        return len(
            tokenizer.encode(
                text,
                add_special_tokens=True
            )
        )
    except Exception:
        return 0


# ============================================================
# TIME GREETING
# ============================================================

def get_greeting():
    hour = datetime.now().hour

    if hour < 12:
        return "Good Morning"

    if hour < 17:
        return "Good Afternoon"

    return "Good Evening"


# ============================================================
# USER USAGE
# ============================================================

def get_user_usage(username):

    if username not in usage:
        usage[username] = {
            "requests": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }

    return usage[username]


def update_usage(
    username,
    input_tokens,
    output_tokens
):

    user_usage = get_user_usage(username)

    user_usage["requests"] += 1
    user_usage["input_tokens"] += input_tokens
    user_usage["output_tokens"] += output_tokens
    user_usage["total_tokens"] += (
        input_tokens + output_tokens
    )

    save_json(USAGE_FILE, usage)


# ============================================================
# SIMPLE TEXT CHUNKING
# ============================================================

def split_text(text, chunk_size=1000, overlap=200):

    if not text:
        return []

    text = re.sub(r"\s+", " ", text).strip()

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

    if not text:
        return None

    chunks = split_text(
        text,
        chunk_size=1000,
        overlap=200
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
        embeddings
    )

    return vectorstore.as_retriever(
        search_kwargs={
            "k": 2
        }
    )


# ============================================================
# PDF READER
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
            continue

    return "\n".join(pages)


# ============================================================
# DOCX READER
# ============================================================

def read_docx(uploaded_file):

    doc = DocxDocument(uploaded_file)

    paragraphs = []

    for paragraph in doc.paragraphs:

        if paragraph.text.strip():
            paragraphs.append(
                paragraph.text
            )

    return "\n".join(paragraphs)


# ============================================================
# XLSX READER
# ============================================================

def read_xlsx(uploaded_file):

    excel_file = pd.ExcelFile(uploaded_file)

    all_text = []

    for sheet in excel_file.sheet_names:

        df = pd.read_excel(
            excel_file,
            sheet_name=sheet
        )

        all_text.append(
            f"Sheet: {sheet}"
        )

        all_text.append(
            df.to_string(index=False)
        )

    return "\n".join(all_text)


# ============================================================
# DOCUMENT PROCESSOR
# ============================================================

def process_document(uploaded_file):

    filename = uploaded_file.name.lower()

    try:

        if filename.endswith(".pdf"):
            text = read_pdf(uploaded_file)

        elif filename.endswith(".txt"):
            text = uploaded_file.read().decode(
                "utf-8",
                errors="ignore"
            )

        elif filename.endswith(".docx"):
            text = read_docx(uploaded_file)

        elif filename.endswith(".xlsx"):
            text = read_xlsx(uploaded_file)

        else:
            return None

        return text

    except Exception as e:

        st.error(
            f"Could not read document: {e}"
        )

        return None


# ============================================================
# WEBSITE READER
# ============================================================

def read_website(url):

    try:

        headers = {
            "User-Agent": "Mozilla/5.0"
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

        for tag in soup([
            "script",
            "style",
            "noscript",
            "header",
            "footer",
            "nav"
        ]):
            tag.decompose()

        text = soup.get_text(
            separator=" ",
            strip=True
        )

        return text

    except Exception as e:

        st.error(
            f"Could not read website: {e}"
        )

        return None


# ============================================================
# CHAT SAVE
# ============================================================

def save_current_chat():

    username = st.session_state.username

    chat_id = st.session_state.current_chat_id

    if not username or not chat_id:
        return

    if username not in chats:
        chats[username] = {}

    messages = st.session_state.messages

    if messages:

        first_user_message = next(
            (
                msg["content"]
                for msg in messages
                if msg["role"] == "user"
            ),
            "New Chat"
        )

        title = first_user_message[:45]

        if len(first_user_message) > 45:
            title += "..."

    else:
        title = "New Chat"

    chats[username][chat_id] = {
        "title": title,
        "messages": messages,
        "document_text": st.session_state.document_text,
        "document_name": st.session_state.document_name,
        "source_type": st.session_state.source_type,
        "total_tokens": st.session_state.last_total_tokens,
        "updated_at": datetime.now().isoformat(),
    }

    save_json(
        CHATS_FILE,
        chats
    )


# ============================================================
# NEW CHAT
# ============================================================

def start_new_chat():

    st.session_state.current_chat_id = str(
        uuid.uuid4()
    )

    st.session_state.messages = []

    st.session_state.retriever = None

    st.session_state.document_text = ""

    st.session_state.document_name = ""

    st.session_state.source_type = ""

    st.session_state.chat_loaded = True

    st.session_state.last_input_tokens = 0

    st.session_state.last_output_tokens = 0

    st.session_state.last_total_tokens = 0


# ============================================================
# LOAD CHAT
# ============================================================

def load_chat(chat_id):

    username = st.session_state.username

    user_chats = chats.get(
        username,
        {}
    )

    chat = user_chats.get(chat_id)

    if not chat:
        return

    st.session_state.current_chat_id = chat_id

    st.session_state.messages = chat.get(
        "messages",
        []
    )

    st.session_state.document_text = chat.get(
        "document_text",
        ""
    )

    st.session_state.document_name = chat.get(
        "document_name",
        ""
    )

    st.session_state.source_type = chat.get(
        "source_type",
        ""
    )

    st.session_state.last_total_tokens = chat.get(
        "total_tokens",
        0
    )

    st.session_state.chat_loaded = True

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
# DELETE CHAT
# ============================================================

def delete_chat(chat_id):

    username = st.session_state.username

    if username in chats:

        if chat_id in chats[username]:

            del chats[username][chat_id]

            save_json(
                CHATS_FILE,
                chats
            )


# ============================================================
# LOGIN PAGE
# ============================================================

def show_auth_page():

    st.markdown(
        """
        <div style="
            max-width:760px;
            margin:80px auto 0 auto;
            text-align:center;
        ">

            <div style="
                font-size:38px;
                color:#f0a45d;
                margin-bottom:12px;
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
                color:#aaa59b;
                margin-top:8px;
                margin-bottom:50px;
            ">
                Ask questions about your documents and websites.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    tab_login, tab_create = st.tabs(
        ["Login", "Create Account"]
    )

    with tab_login:

        username = st.text_input(
            "Username",
            placeholder="Enter your username",
            key="login_username"
        )

        password = st.text_input(
            "Password",
            placeholder="Enter your password",
            type="password",
            key="login_password"
        )

        if st.button(
            "Login",
            use_container_width=True
        ):

            if (
                username in users
                and users[username] == password
            ):

                st.session_state.logged_in = True

                st.session_state.username = username

                st.session_state.current_chat_id = None

                st.session_state.messages = []

                st.session_state.retriever = None

                st.rerun()

            else:

                st.error(
                    "Invalid username or password."
                )

    with tab_create:

        new_username = st.text_input(
            "Username",
            placeholder="Choose a username",
            key="create_username"
        )

        new_password = st.text_input(
            "Password",
            placeholder="Create a password",
            type="password",
            key="create_password"
        )

        if st.button(
            "Create Account",
            use_container_width=True
        ):

            if not new_username or not new_password:

                st.warning(
                    "Please enter username and password."
                )

            elif new_username in users:

                st.warning(
                    "Username already exists."
                )

            else:

                users[new_username] = new_password

                save_json(
                    USERS_FILE,
                    users
                )

                st.success(
                    "Account created. Please login."
                )


# ============================================================
# AUTH CHECK
# ============================================================

if not st.session_state.logged_in:

    show_auth_page()

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-brand">
            <span>✦</span> AI Document Intelligence
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.button(
        "+ New Chat",
        use_container_width=True
    ):

        start_new_chat()

        st.rerun()

    st.markdown(
        '<div class="sidebar-section">Chats</div>',
        unsafe_allow_html=True
    )

    user_chats = chats.get(
        st.session_state.username,
        {}
    )

    if user_chats:

        sorted_chats = sorted(
            user_chats.items(),
            key=lambda x: x[1].get(
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

            if st.button(
                title,
                key=f"chat_{chat_id}",
                use_container_width=True
            ):

                load_chat(chat_id)

                st.rerun()

    else:

        st.caption(
            "No conversations yet."
        )

    st.markdown(
        '<div class="sidebar-section">Usage</div>',
        unsafe_allow_html=True
    )

    current_usage = get_user_usage(
        st.session_state.username
    )

    st.markdown(
        f"""
        <div class="usage-card">

            <div class="usage-line">
                <span>Requests</span>
                <span class="usage-value">
                    {current_usage["requests"]}
                </span>
            </div>

            <div class="usage-line">
                <span>Input tokens</span>
                <span class="usage-value">
                    {current_usage["input_tokens"]}
                </span>
            </div>

            <div class="usage-line">
                <span>Output tokens</span>
                <span class="usage-value">
                    {current_usage["output_tokens"]}
                </span>
            </div>

            <div class="usage-line">
                <span>Total tokens</span>
                <span class="usage-value">
                    {current_usage["total_tokens"]}
                </span>
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("")

    if st.button(
        "⚙ Settings",
        use_container_width=True
    ):

        st.info(
            "Settings can be added here later."
        )

    if st.button(
        "Logout",
        use_container_width=True
    ):

        st.session_state.logged_in = False
        st.session_state.username = ""

        st.session_state.messages = []

        st.session_state.retriever = None

        st.rerun()


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    """
    <div class="main-title">
        <span class="star">✦</span>
        AI Document Intelligence
    </div>

    <div class="main-subtitle">
        Document & Website Assistant
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# FIRST CHAT
# ============================================================

if not st.session_state.current_chat_id:

    start_new_chat()


# ============================================================
# DOCUMENT / WEBSITE SOURCE
# ============================================================

with st.expander(
    "📎  Add a document or website",
    expanded=not bool(
        st.session_state.document_text
    )
):

    st.markdown(
        '<div class="source-label">Choose source</div>',
        unsafe_allow_html=True
    )

    source_mode = st.radio(
        "Source",
        [
            "Document",
            "Website URL"
        ],
        horizontal=True,
        label_visibility="collapsed"
    )

    # --------------------------------------------------------
    # DOCUMENT
    # --------------------------------------------------------

    if source_mode == "Document":

        uploaded_file = st.file_uploader(
            "Upload PDF, TXT, DOCX or XLSX",
            type=[
                "pdf",
                "txt",
                "docx",
                "xlsx"
            ]
        )

        if uploaded_file:

            if (
                st.session_state.document_name
                != uploaded_file.name
            ):

                with st.spinner(
                    "Reading your document..."
                ):

                    text = process_document(
                        uploaded_file
                    )

                    if text:

                        st.session_state.document_text = text

                        st.session_state.document_name = (
                            uploaded_file.name
                        )

                        st.session_state.source_type = (
                            "document"
                        )

                        try:

                            st.session_state.retriever = (
                                create_retriever(text)
                            )

                            st.success(
                                f"Loaded: {uploaded_file.name}"
                            )

                        except Exception as e:

                            st.error(
                                f"Could not create search index: {e}"
                            )

                        save_current_chat()

    # --------------------------------------------------------
    # WEBSITE
    # --------------------------------------------------------

    else:

        website_url = st.text_input(
            "Website URL",
            placeholder="https://example.com"
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
                    "Reading website..."
                ):

                    text = read_website(
                        website_url.strip()
                    )

                    if text:

                        st.session_state.document_text = text

                        st.session_state.document_name = (
                            website_url.strip()
                        )

                        st.session_state.source_type = (
                            "website"
                        )

                        try:

                            st.session_state.retriever = (
                                create_retriever(text)
                            )

                            st.success(
                                "Website loaded successfully."
                            )

                        except Exception as e:

                            st.error(
                                f"Could not create search index: {e}"
                            )

                        save_current_chat()


# ============================================================
# CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# WELCOME SCREEN
# ============================================================

if not st.session_state.messages:

    greeting = get_greeting()

    st.markdown(
        f"""
        <div class="welcome-area">

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
        """,
        unsafe_allow_html=True
    )


# ============================================================
# TOKEN USAGE NEAR KEYBOARD
# ============================================================

st.markdown(
    f"""
    <div class="token-bar">

        <div class="token-badge">
            Last response:
            <strong>
                {st.session_state.last_total_tokens}
            </strong>
            tokens
            &nbsp;•&nbsp;
            Max response:
            <strong>
                100
            </strong>
            tokens
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CHAT INPUT
# ============================================================

prompt = st.chat_input(
    "Ask anything about your document..."
)


# ============================================================
# PROCESS QUESTION
# ============================================================

if prompt:

    # --------------------------------------------------------
    # USER MESSAGE
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    with st.chat_message("user"):

        st.markdown(prompt)

    # --------------------------------------------------------
    # CHECK DOCUMENT
    # --------------------------------------------------------

    if not st.session_state.retriever:

        answer = (
            "Please upload a document or load a website "
            "before asking questions."
        )

        with st.chat_message("assistant"):

            st.markdown(answer)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        save_current_chat()

        st.rerun()

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    with st.spinner(
        "Thinking..."
    ):

        try:

            tokenizer, generator = load_model()

            # ------------------------------------------------
            # RETRIEVE RELEVANT DOCUMENT PARTS
            # ------------------------------------------------

            relevant_docs = (
                st.session_state.retriever.invoke(
                    prompt
                )
            )

            context_parts = []

            for doc in relevant_docs:

                context_parts.append(
                    doc.page_content
                )

            context = "\n\n".join(
                context_parts
            )

            # ------------------------------------------------
            # PROMPT
            # ------------------------------------------------

            system_prompt = """
You are an AI Document Intelligence assistant.

Answer the user's question using ONLY the provided context.

If the answer cannot be found in the context,
say:

"I could not find that information in the uploaded document."

Keep the answer clear and concise.
"""

            final_prompt = f"""
{system_prompt}

CONTEXT:
{context}

USER QUESTION:
{prompt}

ANSWER:
"""

            # ------------------------------------------------
            # INPUT TOKENS
            # ------------------------------------------------

            input_tokens = count_tokens(
                tokenizer,
                final_prompt
            )

            # ------------------------------------------------
            # GENERATE
            # ------------------------------------------------

            result = generator(
                final_prompt,
                max_new_tokens=100,
                do_sample=False,
                return_full_text=False,
                pad_token_id=tokenizer.eos_token_id
            )

            answer = result[0]["generated_text"].strip()

            # ------------------------------------------------
            # CLEAN ANSWER
            # ------------------------------------------------

            if "ANSWER:" in answer:

                answer = answer.split(
                    "ANSWER:",
                    1
                )[-1].strip()

            # Remove accidental prompt repetition
            if "CONTEXT:" in answer:

                answer = answer.split(
                    "CONTEXT:",
                    1
                )[0].strip()

            if not answer:

                answer = (
                    "I could not find that information "
                    "in the uploaded document."
                )

            # ------------------------------------------------
            # OUTPUT TOKENS
            # ------------------------------------------------

            output_tokens = count_tokens(
                tokenizer,
                answer
            )

            total_tokens = (
                input_tokens
                + output_tokens
            )

            st.session_state.last_input_tokens = (
                input_tokens
            )

            st.session_state.last_output_tokens = (
                output_tokens
            )

            st.session_state.last_total_tokens = (
                total_tokens
            )

            # ------------------------------------------------
            # UPDATE USER USAGE
            # ------------------------------------------------

            update_usage(
                st.session_state.username,
                input_tokens,
                output_tokens
            )

        except Exception as e:

            answer = (
                f"Sorry, I couldn't process the question.\n\n"
                f"Error: {e}"
            )

            input_tokens = 0
            output_tokens = 0
            total_tokens = 0

    # --------------------------------------------------------
    # ASSISTANT MESSAGE
    # --------------------------------------------------------

    with st.chat_message(
        "assistant"
    ):

        st.markdown(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    # --------------------------------------------------------
    # SAVE CHAT
    # --------------------------------------------------------

    save_current_chat()

    # --------------------------------------------------------
    # REFRESH
    # --------------------------------------------------------

    st.rerun()


# ============================================================
# CURRENT TOKEN INFO BELOW CHAT
# ============================================================

if st.session_state.last_total_tokens > 0:

    st.caption(
        f"Last response: "
        f"{st.session_state.last_input_tokens} input • "
        f"{st.session_state.last_output_tokens} output • "
        f"{st.session_state.last_total_tokens} total tokens"
    )

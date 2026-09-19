import os
import json
import uuid
import time
from datetime import datetime

import streamlit as st
import pandas as pd
import requests

from bs4 import BeautifulSoup

from docx import Document as DocxDocument

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
)

import torch


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Document Intelligence",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CONSTANTS
# ============================================================

USERS_FILE = "users.json"
CHATS_FILE = "chats.json"
USAGE_FILE = "usage.json"

MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

MAX_RESPONSE_TOKENS = 100
RETRIEVER_K = 2


# ============================================================
# DARK CLAUDE-STYLE THEME
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       MAIN PAGE
       ======================================================== */

    .stApp {
        background: #1f1e1b;
        color: #f5f1e8;
    }

    .main {
        background: #1f1e1b;
    }

    [data-testid="stAppViewContainer"] {
        background: #1f1e1b;
    }

    [data-testid="stHeader"] {
        background: #1f1e1b;
    }

    /* ========================================================
       SIDEBAR
       ======================================================== */

    [data-testid="stSidebar"] {
        background: #171614;
        border-right: 1px solid #37342f;
    }

    [data-testid="stSidebar"] * {
        color: #eee9df;
    }

    /* ========================================================
       SIDEBAR BRAND
       ======================================================== */

    .sidebar-brand {
        font-size: 20px;
        font-weight: 700;
        color: #f5f1e8;
        padding: 12px 4px 20px 4px;
    }

    .sidebar-brand span {
        color: #f0a45d;
    }

    /* ========================================================
       BUTTONS
       ======================================================== */

    .stButton > button {
        background: #2b2925 !important;
        color: #f5f1e8 !important;
        border: 1px solid #45413b !important;
        border-radius: 10px !important;
        min-height: 42px;
        font-weight: 500;
    }

    .stButton > button:hover {
        background: #34312c !important;
        border-color: #625b51 !important;
        color: #ffffff !important;
    }

    /* Primary buttons */

    .primary-button > button {
        background: #f0a45d !important;
        color: #1f1e1b !important;
        border: none !important;
        font-weight: 700 !important;
    }

    .primary-button > button:hover {
        background: #f5b275 !important;
    }

    /* ========================================================
       INPUTS
       ======================================================== */

    input,
    textarea {
        background: #292723 !important;
        color: #f5f1e8 !important;
        border: 1px solid #45413b !important;
        border-radius: 10px !important;
    }

    input::placeholder,
    textarea::placeholder {
        color: #8f8a82 !important;
    }

    input:focus,
    textarea:focus {
        border-color: #f0a45d !important;
        box-shadow: 0 0 0 1px #f0a45d !important;
    }

    /* ========================================================
       SELECTBOX / RADIO
       ======================================================== */

    [data-baseweb="select"] > div {
        background: #292723 !important;
        border-color: #45413b !important;
        color: #f5f1e8 !important;
    }

    [data-testid="stRadio"] label {
        color: #ddd7ce !important;
    }

    /* ========================================================
       EXPANDER
       ======================================================== */

    [data-testid="stExpander"] {
        background: #24221f !important;
        border: 1px solid #45413b !important;
        border-radius: 12px !important;
    }

    [data-testid="stExpander"] summary {
        color: #f5f1e8 !important;
    }

    /* ========================================================
       APP HEADER
       ======================================================== */

    .app-header {
        text-align: center;
        padding-top: 10px;
        padding-bottom: 18px;
    }

    .app-title {
        font-size: 30px;
        font-weight: 700;
        color: #f5f1e8;
        margin-bottom: 5px;
    }

    .app-title .star {
        color: #f0a45d;
    }

    .app-subtitle {
        color: #9d988f;
        font-size: 15px;
    }

    /* ========================================================
       WELCOME AREA
       ======================================================== */

    .welcome-area {
        text-align: center;
        margin-top: 100px;
        margin-bottom: 100px;
    }

    .welcome-symbol {
        font-size: 42px;
        color: #f0a45d;
        margin-bottom: 15px;
    }

    .welcome-title {
        font-size: 30px;
        font-weight: 650;
        color: #f5f1e8;
        margin-bottom: 12px;
    }

    .welcome-text {
        color: #9d988f;
        font-size: 16px;
    }

    /* ========================================================
       USER MESSAGE
       ======================================================== */

    [data-testid="stChatMessage"] {
        background: transparent !important;
    }

    /* ========================================================
       CHAT MESSAGE
       ======================================================== */

    .assistant-bubble {
        background: #292723;
        border: 1px solid #3d3933;
        border-radius: 14px;
        padding: 15px 18px;
        margin: 8px 0;
        color: #eee9df;
        line-height: 1.65;
    }

    .user-bubble {
        background: #34312c;
        border-radius: 14px;
        padding: 13px 17px;
        color: #f5f1e8;
        line-height: 1.6;
    }

    /* ========================================================
       TOKEN INFORMATION
       ======================================================== */

    .token-info {
        text-align: right;
        color: #858078;
        font-size: 11px;
        margin-top: 5px;
        padding-right: 4px;
    }

    .token-info strong {
        color: #aaa49b;
    }

    /* ========================================================
       USAGE BOX
       ======================================================== */

    .usage-box {
        background: #25231f;
        border: 1px solid #3b3832;
        border-radius: 12px;
        padding: 13px;
        margin-top: 8px;
    }

    .usage-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 6px 0;
        font-size: 13px;
        color: #aaa49f;
    }

    .usage-row strong {
        color: #f1ece4;
        font-weight: 600;
    }

    .usage-divider {
        height: 1px;
        background: #3b3832;
        margin: 7px 0;
    }

    .usage-total {
        color: #f0a45d;
    }

    .usage-total strong {
        color: #f0a45d;
    }

    /* ========================================================
       COMPOSER AREA
       ======================================================== */

    .composer-wrapper {
        background: #24221f;
        border: 1px solid #45413b;
        border-radius: 16px;
        padding: 8px 12px 7px 12px;
        margin-top: 10px;
    }

    .composer-label {
        color: #777169;
        font-size: 11px;
        text-align: right;
        padding-right: 8px;
        margin-top: -2px;
    }

    /* ========================================================
       MAX RESPONSE BADGE
       ======================================================== */

    .max-token-badge {
        display: inline-block;
        background: #2d2a26;
        border: 1px solid #48433b;
        color: #aaa39a;
        border-radius: 8px;
        padding: 4px 9px;
        font-size: 11px;
        float: right;
        margin-top: -3px;
    }

    .max-token-badge strong {
        color: #d2cbc0;
    }

    /* ========================================================
       SOURCE BOX
       ======================================================== */

    .source-info {
        background: #25231f;
        border: 1px solid #3d3933;
        border-radius: 10px;
        padding: 10px 13px;
        color: #aaa49b;
        font-size: 13px;
        margin-bottom: 15px;
    }

    .source-info strong {
        color: #eee9df;
    }

    /* ========================================================
       LOGIN PAGE
       ======================================================== */

    .login-container {
        max-width: 560px;
        margin: 80px auto 0 auto;
    }

    .login-title {
        text-align: center;
        font-size: 34px;
        font-weight: 700;
        color: #f5f1e8;
    }

    .login-subtitle {
        text-align: center;
        color: #9d988f;
        margin-top: 8px;
        margin-bottom: 35px;
    }

    /* ========================================================
       ALERTS
       ======================================================== */

    [data-testid="stAlert"] {
        border-radius: 10px !important;
    }

    /* ========================================================
       HIDE STREAMLIT BRANDING
       ======================================================== */

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
# FILE HELPERS
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
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ============================================================
# LOAD DATA
# ============================================================

users = load_json(USERS_FILE, {})
chats = load_json(CHATS_FILE, {})
usage_data = load_json(USAGE_FILE, {})


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
    "new_chat": False,
    "prompt_text": "",
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# TOKEN COUNTER
# ============================================================

@st.cache_resource
def load_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


def count_tokens(text):
    if not text:
        return 0

    try:
        tokenizer = load_tokenizer()
        tokens = tokenizer.encode(
            text,
            add_special_tokens=True
        )
        return len(tokens)

    except Exception:
        return 0


# ============================================================
# MODEL
# ============================================================

@st.cache_resource
def load_llm():

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
        max_new_tokens=MAX_RESPONSE_TOKENS,
        do_sample=False,
        return_full_text=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    return generator, tokenizer


# ============================================================
# EMBEDDINGS
# ============================================================

@st.cache_resource
def load_embeddings():

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )


# ============================================================
# TIME GREETING
# ============================================================

def get_greeting():

    hour = datetime.now().hour

    if hour < 12:
        return "Good Morning"

    elif hour < 17:
        return "Good Afternoon"

    else:
        return "Good Evening"


# ============================================================
# USER USAGE
# ============================================================

def get_user_usage(username):

    if username not in usage_data:

        usage_data[username] = {
            "requests": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }

        save_json(USAGE_FILE, usage_data)

    return usage_data[username]


# ============================================================
# CREATE CHAT
# ============================================================

def create_new_chat():

    chat_id = str(uuid.uuid4())

    chats[chat_id] = {
        "username": st.session_state.username,
        "title": "New Chat",
        "messages": [],
        "document_text": "",
        "document_name": "",
        "source_type": "",
        "total_tokens": 0,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
    }

    save_json(CHATS_FILE, chats)

    st.session_state.current_chat_id = chat_id
    st.session_state.messages = []
    st.session_state.retriever = None
    st.session_state.document_text = ""
    st.session_state.document_name = ""
    st.session_state.source_type = ""
    st.session_state.chat_loaded = False
    st.session_state.new_chat = True


# ============================================================
# SAVE CURRENT CHAT
# ============================================================

def save_current_chat():

    chat_id = st.session_state.current_chat_id

    if not chat_id:
        return

    if chat_id not in chats:
        return

    chats[chat_id]["messages"] = st.session_state.messages
    chats[chat_id]["document_text"] = st.session_state.document_text
    chats[chat_id]["document_name"] = st.session_state.document_name
    chats[chat_id]["source_type"] = st.session_state.source_type

    chats[chat_id]["updated_at"] = datetime.now().isoformat()

    if st.session_state.messages:

        first_user_message = None

        for msg in st.session_state.messages:

            if msg["role"] == "user":
                first_user_message = msg["content"]
                break

        if first_user_message:

            title = first_user_message[:40]

            if len(first_user_message) > 40:
                title += "..."

            chats[chat_id]["title"] = title

    save_json(CHATS_FILE, chats)


# ============================================================
# LOAD CHAT
# ============================================================

def load_chat(chat_id):

    if chat_id not in chats:
        return

    chat = chats[chat_id]

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

    st.session_state.retriever = None

    # Rebuild retriever if document exists

    if st.session_state.document_text:

        try:

            docs = [
                Document(
                    page_content=st.session_state.document_text
                )
            ]

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            chunks = splitter.split_documents(docs)

            embeddings = load_embeddings()

            vectorstore = FAISS.from_documents(
                chunks,
                embeddings
            )

            st.session_state.retriever = (
                vectorstore.as_retriever(
                    search_kwargs={
                        "k": RETRIEVER_K
                    }
                )
            )

        except Exception as e:

            st.session_state.retriever = None

    st.session_state.chat_loaded = True
    st.session_state.new_chat = False


# ============================================================
# DOCUMENT PROCESSING
# ============================================================

def process_document(uploaded_file):

    file_name = uploaded_file.name

    extension = file_name.lower().split(".")[-1]

    text = ""

    # --------------------------------------------------------
    # PDF
    # --------------------------------------------------------

    if extension == "pdf":

        temp_path = f"_temp_{uuid.uuid4()}.pdf"

        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        try:

            loader = PyPDFLoader(temp_path)

            pages = loader.load()

            text = "\n\n".join(
                page.page_content
                for page in pages
            )

        finally:

            if os.path.exists(temp_path):
                os.remove(temp_path)

    # --------------------------------------------------------
    # TXT
    # --------------------------------------------------------

    elif extension == "txt":

        raw = uploaded_file.read()

        try:
            text = raw.decode("utf-8")

        except UnicodeDecodeError:

            text = raw.decode(
                "latin-1",
                errors="ignore"
            )

    # --------------------------------------------------------
    # DOCX
    # --------------------------------------------------------

    elif extension == "docx":

        doc = DocxDocument(uploaded_file)

        paragraphs = []

        for paragraph in doc.paragraphs:

            if paragraph.text.strip():

                paragraphs.append(
                    paragraph.text
                )

        text = "\n".join(paragraphs)

    # --------------------------------------------------------
    # XLSX
    # --------------------------------------------------------

    elif extension == "xlsx":

        excel_file = pd.ExcelFile(
            uploaded_file
        )

        sheets = []

        for sheet_name in excel_file.sheet_names:

            df = pd.read_excel(
                uploaded_file,
                sheet_name=sheet_name
            )

            sheets.append(
                f"Sheet: {sheet_name}\n\n"
                + df.to_string(index=False)
            )

        text = "\n\n".join(sheets)

    else:

        raise ValueError(
            "Unsupported file format."
        )

    return text


# ============================================================
# WEBSITE PROCESSING
# ============================================================

def process_website(url):

    headers = {
        "User-Agent": (
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )
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

    # Remove unnecessary elements

    for tag in soup(
        [
            "script",
            "style",
            "noscript",
            "header",
            "footer",
            "nav",
            "svg",
        ]
    ):

        tag.decompose()

    text = soup.get_text(
        separator="\n"
    )

    lines = []

    for line in text.splitlines():

        cleaned = line.strip()

        if cleaned:
            lines.append(cleaned)

    return "\n".join(lines)


# ============================================================
# BUILD RETRIEVER
# ============================================================

def build_retriever(text):

    documents = [
        Document(
            page_content=text
        )
    ]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(
        documents
    )

    embeddings = load_embeddings()

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    return vectorstore.as_retriever(
        search_kwargs={
            "k": RETRIEVER_K
        }
    )


# ============================================================
# ANSWER QUESTION
# ============================================================

def answer_question(question):

    if not st.session_state.retriever:

        return (
            "Please upload a document or add a website "
            "before asking a question."
        ), 0, 0

    # --------------------------------------------------------
    # Retrieve context
    # --------------------------------------------------------

    documents = st.session_state.retriever.invoke(
        question
    )

    context = "\n\n".join(
        doc.page_content
        for doc in documents
    )

    # --------------------------------------------------------
    # Prompt
    # --------------------------------------------------------

    prompt = f"""
You are an AI document assistant.

Answer the user's question using ONLY the provided context.

If the answer cannot be found in the context, say:

"I could not find that information in the document."

Be concise and direct.

Context:
{context}

Question:
{question}

Answer:
"""

    # --------------------------------------------------------
    # Token count
    # --------------------------------------------------------

    input_tokens = count_tokens(
        prompt
    )

    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    generator, tokenizer = load_llm()

    result = generator(
        prompt,
        max_new_tokens=MAX_RESPONSE_TOKENS,
        do_sample=False,
        return_full_text=False,
    )

    answer = result[0]["generated_text"].strip()

    # --------------------------------------------------------
    # Clean answer
    # --------------------------------------------------------

    if "Answer:" in answer:

        answer = answer.split(
            "Answer:",
            1
        )[-1].strip()

    output_tokens = count_tokens(
        answer
    )

    return (
        answer,
        input_tokens,
        output_tokens,
    )


# ============================================================
# LOGIN PAGE
# ============================================================

def login_page():

    st.markdown(
        """
        <div class="login-container">

            <div class="login-title">
                <span style="color:#f0a45d;">✦</span>
                AI Document Intelligence
            </div>

            <div class="login-subtitle">
                Ask questions about your documents and websites.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    login_tab, create_tab = st.tabs(
        ["Login", "Create Account"]
    )

    # ========================================================
    # LOGIN
    # ========================================================

    with login_tab:

        username = st.text_input(
            "Username",
            key="login_username",
            placeholder="Enter your username"
        )

        password = st.text_input(
            "Password",
            type="password",
            key="login_password",
            placeholder="Enter your password"
        )

        st.markdown(
            '<div class="primary-button">',
            unsafe_allow_html=True
        )

        login_clicked = st.button(
            "Login",
            use_container_width=True
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )

        if login_clicked:

            if (
                username in users
                and users[username]["password"] == password
            ):

                st.session_state.logged_in = True
                st.session_state.username = username

                # Find user's latest chat

                user_chat_ids = [
                    cid
                    for cid, chat in chats.items()
                    if chat.get("username") == username
                ]

                if user_chat_ids:

                    latest_chat = max(
                        user_chat_ids,
                        key=lambda cid:
                        chats[cid].get(
                            "updated_at",
                            ""
                        )
                    )

                    load_chat(latest_chat)

                else:

                    create_new_chat()

                st.rerun()

            else:

                st.error(
                    "Invalid username or password."
                )

    # ========================================================
    # CREATE ACCOUNT
    # ========================================================

    with create_tab:

        new_username = st.text_input(
            "Username",
            key="create_username",
            placeholder="Choose a username"
        )

        new_password = st.text_input(
            "Password",
            type="password",
            key="create_password",
            placeholder="Create a password"
        )

        confirm_password = st.text_input(
            "Confirm Password",
            type="password",
            key="confirm_password",
            placeholder="Confirm your password"
        )

        st.markdown(
            '<div class="primary-button">',
            unsafe_allow_html=True
        )

        create_clicked = st.button(
            "Create Account",
            use_container_width=True
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )

        if create_clicked:

            new_username = new_username.strip()

            if not new_username:

                st.error(
                    "Please enter a username."
                )

            elif new_username in users:

                st.error(
                    "Username already exists."
                )

            elif not new_password:

                st.error(
                    "Please enter a password."
                )

            elif new_password != confirm_password:

                st.error(
                    "Passwords do not match."
                )

            else:

                users[new_username] = {
                    "password": new_password,
                    "created_at": datetime.now().isoformat(),
                }

                save_json(
                    USERS_FILE,
                    users
                )

                usage_data[new_username] = {
                    "requests": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                }

                save_json(
                    USAGE_FILE,
                    usage_data
                )

                st.success(
                    "Account created successfully. Please login."
                )


# ============================================================
# SHOW LOGIN IF NOT AUTHENTICATED
# ============================================================

if not st.session_state.logged_in:

    login_page()

    st.stop()


# ============================================================
# CURRENT USER USAGE
# ============================================================

user_usage = get_user_usage(
    st.session_state.username
)


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

    # --------------------------------------------------------
    # New Chat
    # --------------------------------------------------------

    if st.button(
        "+ New Chat",
        use_container_width=True
    ):

        create_new_chat()

        st.rerun()

    st.markdown("---")

    # --------------------------------------------------------
    # Chats
    # --------------------------------------------------------

    st.markdown(
        "### Chats"
    )

    user_chats = [
        (cid, chat)
        for cid, chat in chats.items()
        if chat.get("username")
        == st.session_state.username
    ]

    user_chats.sort(
        key=lambda item:
        item[1].get(
            "updated_at",
            ""
        ),
        reverse=True
    )

    if not user_chats:

        st.caption(
            "No conversations yet."
        )

    else:

        for chat_id, chat in user_chats:

            title = chat.get(
                "title",
                "New Chat"
            )

            if len(title) > 30:
                title = title[:30] + "..."

            is_current = (
                chat_id
                == st.session_state.current_chat_id
            )

            button_text = (
                "● " + title
                if is_current
                else title
            )

            if st.button(
                button_text,
                key=f"chat_{chat_id}",
                use_container_width=True
            ):

                load_chat(chat_id)

                st.rerun()

    st.markdown("---")

    # --------------------------------------------------------
    # Usage
    # --------------------------------------------------------

    st.markdown(
        "### Usage"
    )

    st.markdown(
        f"""
        <div class="usage-box">

            <div class="usage-row">
                <span>Requests</span>
                <strong>
                    {user_usage.get("requests", 0)}
                </strong>
            </div>

            <div class="usage-row">
                <span>Input Tokens</span>
                <strong>
                    {user_usage.get("input_tokens", 0)}
                </strong>
            </div>

            <div class="usage-row">
                <span>Output Tokens</span>
                <strong>
                    {user_usage.get("output_tokens", 0)}
                </strong>
            </div>

            <div class="usage-divider"></div>

            <div class="usage-row usage-total">
                <span>Total Tokens</span>
                <strong>
                    {user_usage.get("total_tokens", 0)}
                </strong>
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("")

    # --------------------------------------------------------
    # Settings
    # --------------------------------------------------------

    with st.expander("⚙ Settings"):

        st.write(
            f"Logged in as: **{st.session_state.username}**"
        )

        st.write(
            f"Maximum response: **{MAX_RESPONSE_TOKENS} tokens**"
        )

        st.write(
            f"Retriever documents: **{RETRIEVER_K}**"
        )

    # --------------------------------------------------------
    # Delete Chat
    # --------------------------------------------------------

    if st.session_state.current_chat_id:

        if st.button(
            "Delete Current Chat",
            use_container_width=True
        ):

            chat_id = (
                st.session_state.current_chat_id
            )

            if chat_id in chats:

                del chats[chat_id]

                save_json(
                    CHATS_FILE,
                    chats
                )

            create_new_chat()

            st.rerun()

    # --------------------------------------------------------
    # Logout
    # --------------------------------------------------------

    if st.button(
        "Logout",
        use_container_width=True
    ):

        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.current_chat_id = None
        st.session_state.messages = []
        st.session_state.retriever = None

        st.rerun()


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    """
    <div class="app-header">

        <div class="app-title">
            <span class="star">✦</span>
            AI Document Intelligence
        </div>

        <div class="app-subtitle">
            Document & Website Assistant
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SOURCE UPLOAD
# ============================================================

with st.expander(
    "📎  Add a document or website",
    expanded=(
        not bool(
            st.session_state.document_text
        )
    )
):

    source_type = st.radio(
        "Choose source",
        [
            "Document",
            "Website URL"
        ],
        horizontal=True,
        key="source_selector"
    )

    # ========================================================
    # DOCUMENT
    # ========================================================

    if source_type == "Document":

        uploaded_file = st.file_uploader(
            "Upload PDF, TXT, DOCX or XLSX",
            type=[
                "pdf",
                "txt",
                "docx",
                "xlsx"
            ],
            key="document_uploader"
        )

        if uploaded_file:

            # Only process if new file

            if (
                uploaded_file.name
                != st.session_state.document_name
            ):

                with st.spinner(
                    "Reading document..."
                ):

                    try:

                        text = process_document(
                            uploaded_file
                        )

                        if not text.strip():

                            st.error(
                                "No readable text was found."
                            )

                        else:

                            with st.spinner(
                                "Building semantic search..."
                            ):

                                retriever = build_retriever(
                                    text
                                )

                            st.session_state.document_text = text
                            st.session_state.document_name = uploaded_file.name
                            st.session_state.source_type = "Document"
                            st.session_state.retriever = retriever

                            save_current_chat()

                            st.success(
                                f"{uploaded_file.name} is ready."
                            )

                    except Exception as e:

                        st.error(
                            f"Could not process the document: {e}"
                        )

    # ========================================================
    # WEBSITE
    # ========================================================

    else:

        url = st.text_input(
            "Website URL",
            placeholder="https://example.com",
            key="website_url"
        )

        if st.button(
            "Load Website",
            use_container_width=False
        ):

            if not url.strip():

                st.warning(
                    "Please enter a website URL."
                )

            else:

                with st.spinner(
                    "Reading website..."
                ):

                    try:

                        text = process_website(
                            url.strip()
                        )

                        if not text.strip():

                            st.error(
                                "No readable text was found on this website."
                            )

                        else:

                            with st.spinner(
                                "Building semantic search..."
                            ):

                                retriever = build_retriever(
                                    text
                                )

                            st.session_state.document_text = text
                            st.session_state.document_name = url.strip()
                            st.session_state.source_type = "Website"
                            st.session_state.retriever = retriever

                            save_current_chat()

                            st.success(
                                "Website is ready."
                            )

                    except Exception as e:

                        st.error(
                            f"Could not load website: {e}"
                        )


# ============================================================
# SOURCE INFORMATION
# ============================================================

if st.session_state.document_text:

    source_name = (
        st.session_state.document_name
        or "Unknown source"
    )

    st.markdown(
        f"""
        <div class="source-info">
            <strong>Source:</strong> {source_name}
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# CHAT HISTORY
# ============================================================

if st.session_state.messages:

    for message in st.session_state.messages:

        if message["role"] == "user":

            with st.chat_message("user"):

                st.markdown(
                    message["content"]
                )

        else:

            with st.chat_message("assistant"):

                st.markdown(
                    message["content"]
                )

                if "input_tokens" in message:

                    st.markdown(
                        f"""
                        <div class="token-info">
                            <strong>
                                {message.get("input_tokens", 0)}
                            </strong>
                            input ·
                            <strong>
                                {message.get("output_tokens", 0)}
                            </strong>
                            output ·
                            <strong>
                                {message.get("total_tokens", 0)}
                            </strong>
                            total
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


# ============================================================
# WELCOME SCREEN
# ============================================================

if not st.session_state.messages:

    greeting = get_greeting()

    st.markdown(
        f"""
        <div class="welcome-area">

            <div class="welcome-symbol">
                ✦
            </div>

            <div class="welcome-title">
                {greeting}, {st.session_state.username}
            </div>

            <div class="welcome-text">
                What would you like to know?
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# CHAT COMPOSER
# ============================================================

st.markdown(
    """
    <div class="composer-wrapper">

        <div class="max-token-badge">
            Max response:
            <strong>100 tokens</strong>
        </div>

        <div style="clear:both;"></div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CHAT INPUT
# ============================================================

prompt = st.chat_input(
    "Ask anything about your document...",
    key="chat_input"
)


# ============================================================
# PROCESS QUESTION
# ============================================================

if prompt:

    prompt = prompt.strip()

    if not prompt:
        st.stop()

    # --------------------------------------------------------
    # Add user message
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    # --------------------------------------------------------
    # Display user message
    # --------------------------------------------------------

    with st.chat_message("user"):

        st.markdown(prompt)

    # --------------------------------------------------------
    # Generate answer
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        if not st.session_state.retriever:

            answer = (
                "Please upload a document or add a "
                "website before asking a question."
            )

            input_tokens = count_tokens(
                prompt
            )

            output_tokens = count_tokens(
                answer
            )

        else:

            with st.spinner(
                "Thinking..."
            ):

                try:

                    (
                        answer,
                        input_tokens,
                        output_tokens,
                    ) = answer_question(
                        prompt
                    )

                except Exception as e:

                    answer = (
                        "Sorry, I couldn't process "
                        "your question."
                    )

                    input_tokens = count_tokens(
                        prompt
                    )

                    output_tokens = count_tokens(
                        answer
                    )

                    st.error(
                        str(e)
                    )

        # ----------------------------------------------------
        # Show answer
        # ----------------------------------------------------

        st.markdown(answer)

        total_tokens = (
            input_tokens
            + output_tokens
        )

        # ----------------------------------------------------
        # Token information
        # ----------------------------------------------------

        st.markdown(
            f"""
            <div class="token-info">
                <strong>{input_tokens}</strong>
                input ·
                <strong>{output_tokens}</strong>
                output ·
                <strong>{total_tokens}</strong>
                total
            </div>
            """,
            unsafe_allow_html=True
        )

    # --------------------------------------------------------
    # Save assistant message
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        }
    )

    # --------------------------------------------------------
    # Update user usage
    # --------------------------------------------------------

    user_usage["requests"] += 1

    user_usage["input_tokens"] += input_tokens

    user_usage["output_tokens"] += output_tokens

    user_usage["total_tokens"] += total_tokens

    save_json(
        USAGE_FILE,
        usage_data
    )

    # --------------------------------------------------------
    # Update chat token usage
    # --------------------------------------------------------

    if st.session_state.current_chat_id:

        chat_id = (
            st.session_state.current_chat_id
        )

        if chat_id in chats:

            chats[chat_id]["total_tokens"] = (
                chats[chat_id].get(
                    "total_tokens",
                    0
                )
                + total_tokens
            )

    # --------------------------------------------------------
    # Save chat
    # --------------------------------------------------------

    save_current_chat()

    # --------------------------------------------------------
    # Rerun
    # --------------------------------------------------------

    st.rerun()

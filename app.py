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

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Document Intelligence",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# DARK CLAUDE-STYLE THEME
# ============================================================

st.markdown(
    """
    <style>

    /* ==============================
       MAIN APPLICATION
       ============================== */

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

    /* ==============================
       SIDEBAR
       ============================== */

    section[data-testid="stSidebar"] {
        background-color: #171614;
        border-right: 1px solid #34312c;
    }

    section[data-testid="stSidebar"] > div {
        background-color: #171614;
    }

    /* ==============================
       GENERAL TEXT
       ============================== */

    p,
    label,
    span {
        color: #eeeae2;
    }

    /* ==============================
       SIDEBAR BRAND
       ============================== */

    .brand {
        font-size: 18px;
        font-weight: 700;
        color: #f4f1ea;
        padding: 8px 0 22px 0;
    }

    .brand-star {
        color: #f0a45d;
    }

    /* ==============================
       SIDEBAR SECTION TITLE
       ============================== */

    .sidebar-title {
        color: #969087;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-top: 25px;
        margin-bottom: 10px;
    }

    /* ==============================
       USAGE CARD
       ============================== */

    .usage-card {
        background-color: #25231f;
        border: 1px solid #3b3832;
        border-radius: 13px;
        padding: 15px;
        margin-top: 8px;
    }

    .usage-header {
        color: #f4f1ea;
        font-size: 15px;
        font-weight: 700;
        margin-bottom: 14px;
    }

    .usage-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin: 9px 0;
        color: #a8a198;
        font-size: 13px;
    }

    .usage-number {
        color: #f1ece4;
        font-weight: 600;
    }

    .usage-total {
        color: #f0a45d;
        font-weight: 700;
    }

    /* ==============================
       MAIN TITLE
       ============================== */

    .main-title {
        text-align: center;
        margin-top: 12px;
        color: #f4f1ea;
        font-size: 31px;
        font-weight: 700;
        letter-spacing: -0.02em;
    }

    .main-star {
        color: #f0a45d;
    }

    .main-subtitle {
        text-align: center;
        color: #9f9990;
        font-size: 14px;
        margin-top: 6px;
    }

    /* ==============================
       WELCOME SCREEN
       ============================== */

    .welcome {
        text-align: center;
        margin-top: 105px;
    }

    .welcome-star {
        color: #f0a45d;
        font-size: 40px;
        margin-bottom: 15px;
    }

    .welcome-title {
        color: #f4f1ea;
        font-size: 31px;
        font-weight: 650;
        margin-bottom: 10px;
    }

    .welcome-text {
        color: #9f9990;
        font-size: 17px;
    }

    /* ==============================
       TOKEN BAR ABOVE KEYBOARD
       ============================== */

    .token-bar {
        display: flex;
        justify-content: flex-end;
        align-items: center;
        margin-top: 5px;
        margin-bottom: 7px;
        padding-right: 8px;
    }

    .token-badge {
        background-color: #292722;
        border: 1px solid #454139;
        border-radius: 8px;
        padding: 6px 11px;
        font-size: 11px;
        color: #9f9990;
    }

    .token-number {
        color: #f1ece4;
        font-weight: 700;
    }

    .token-limit {
        color: #f0a45d;
        font-weight: 700;
    }

    /* ==============================
       CHAT INPUT
       ============================== */

    [data-testid="stChatInput"] {
        background-color: #292722 !important;
        border: 1px solid #4a463e !important;
        border-radius: 16px !important;
    }

    [data-testid="stChatInput"] textarea {
        background-color: #292722 !important;
        color: #f4f1ea !important;
        border: none !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: #858078 !important;
    }

    [data-testid="stChatInput"] button {
        background-color: #f0a45d !important;
        color: #1f1e1b !important;
        border-radius: 10px !important;
    }

    [data-testid="stChatInput"] button:hover {
        background-color: #e8954e !important;
    }

    /* ==============================
       CHAT MESSAGES
       ============================== */

    [data-testid="stChatMessage"] {
        background-color: transparent !important;
    }

    [data-testid="stChatMessageContent"] {
        color: #eeeae2 !important;
        line-height: 1.65;
    }

    /* ==============================
       BUTTONS
       ============================== */

    .stButton > button {
        background-color: #292722 !important;
        color: #eeeae2 !important;
        border: 1px solid #454139 !important;
        border-radius: 9px !important;
    }

    .stButton > button:hover {
        background-color: #34312b !important;
        border-color: #5a554b !important;
    }

    /* ==============================
       TEXT INPUTS
       ============================== */

    input,
    textarea {
        background-color: #292722 !important;
        color: #eeeae2 !important;
    }

    div[data-baseweb="input"] {
        background-color: #292722 !important;
        border-color: #48443d !important;
    }

    /* ==============================
       SELECT BOX
       ============================== */

    div[data-baseweb="select"] > div {
        background-color: #292722 !important;
        color: #eeeae2 !important;
        border-color: #48443d !important;
    }

    /* ==============================
       FILE UPLOADER
       ============================== */

    [data-testid="stFileUploader"] {
        background-color: #25231f;
        border: 1px dashed #4a463d;
        border-radius: 12px;
        padding: 10px;
    }

    /* ==============================
       EXPANDER
       ============================== */

    [data-testid="stExpander"] {
        background-color: #25231f;
        border: 1px solid #403c35;
        border-radius: 12px;
    }

    /* ==============================
       HIDE STREAMLIT BRANDING
       ============================== */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# FILE NAMES
# ============================================================

USERS_FILE = "users.json"
CHATS_FILE = "chats.json"
USAGE_FILE = "usage.json"


# ============================================================
# JSON FUNCTIONS
# ============================================================

def load_json(filename, default):

    if not os.path.exists(filename):
        return default

    try:

        with open(
            filename,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    except Exception:

        return default


def save_json(filename, data):

    try:

        with open(
            filename,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                data,
                file,
                indent=2,
                ensure_ascii=False
            )

    except Exception:

        pass


# ============================================================
# LOAD DATA
# ============================================================

users = load_json(
    USERS_FILE,
    {}
)

chats = load_json(
    CHATS_FILE,
    {}
)

usage = load_json(
    USAGE_FILE,
    {}
)


# ============================================================
# SESSION STATE
# ============================================================

session_defaults = {

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

    "chat_initialized": False
}


for key, value in session_defaults.items():

    if key not in st.session_state:

        st.session_state[key] = value


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"


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
        tokenizer=tokenizer
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
# TOKEN COUNTER
# ============================================================

def count_tokens(
    tokenizer,
    text
):

    if not text:
        return 0

    try:

        tokens = tokenizer.encode(
            text,
            add_special_tokens=True
        )

        return len(tokens)

    except Exception:

        return 0


# ============================================================
# GREETING
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

def get_usage(username):

    if username not in usage:

        usage[username] = {

            "requests": 0,

            "input_tokens": 0,

            "output_tokens": 0,

            "total_tokens": 0
        }

    return usage[username]


def update_usage(
    username,
    input_tokens,
    output_tokens
):

    user_usage = get_usage(username)

    user_usage["requests"] += 1

    user_usage["input_tokens"] += input_tokens

    user_usage["output_tokens"] += output_tokens

    user_usage["total_tokens"] += (
        input_tokens + output_tokens
    )

    save_json(
        USAGE_FILE,
        usage
    )


# ============================================================
# TEXT CHUNKING
# ============================================================

def split_text(
    text,
    chunk_size=1000,
    overlap=200
):

    if not text:
        return []

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        if chunk.strip():

            chunks.append(
                chunk.strip()
            )

        if end >= len(text):

            break

        start = end - overlap

    return chunks


# ============================================================
# CREATE VECTOR SEARCH
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

    documents = []

    for chunk in chunks:

        documents.append(
            Document(
                page_content=chunk
            )
        )

    embeddings = load_embeddings()

    vectorstore = FAISS.from_documents(
        documents,
        embeddings
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 2
        }
    )

    return retriever


# ============================================================
# READ PDF
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

            pass

    return "\n".join(pages)


# ============================================================
# READ DOCX
# ============================================================

def read_docx(uploaded_file):

    document = DocxDocument(
        uploaded_file
    )

    paragraphs = []

    for paragraph in document.paragraphs:

        if paragraph.text.strip():

            paragraphs.append(
                paragraph.text
            )

    return "\n".join(paragraphs)


# ============================================================
# READ XLSX
# ============================================================

def read_xlsx(uploaded_file):

    excel = pd.ExcelFile(
        uploaded_file
    )

    all_text = []

    for sheet in excel.sheet_names:

        dataframe = pd.read_excel(
            excel,
            sheet_name=sheet
        )

        all_text.append(
            f"Sheet: {sheet}"
        )

        all_text.append(
            dataframe.to_string(
                index=False
            )
        )

    return "\n".join(all_text)


# ============================================================
# READ DOCUMENT
# ============================================================

def process_document(
    uploaded_file
):

    filename = uploaded_file.name.lower()

    try:

        if filename.endswith(".pdf"):

            return read_pdf(
                uploaded_file
            )

        elif filename.endswith(".txt"):

            return uploaded_file.read().decode(
                "utf-8",
                errors="ignore"
            )

        elif filename.endswith(".docx"):

            return read_docx(
                uploaded_file
            )

        elif filename.endswith(".xlsx"):

            return read_xlsx(
                uploaded_file
            )

        else:

            return None

    except Exception as error:

        st.error(
            f"Could not read document: {error}"
        )

        return None


# ============================================================
# READ WEBSITE
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

        for tag in soup(
            [
                "script",
                "style",
                "noscript",
                "header",
                "footer",
                "nav"
            ]
        ):

            tag.decompose()

        text = soup.get_text(
            separator=" ",
            strip=True
        )

        return text

    except Exception as error:

        st.error(
            f"Could not read website: {error}"
        )

        return None


# ============================================================
# CREATE NEW CHAT
# ============================================================

def create_new_chat():

    st.session_state.current_chat_id = str(
        uuid.uuid4()
    )

    st.session_state.messages = []

    st.session_state.retriever = None

    st.session_state.document_text = ""

    st.session_state.document_name = ""

    st.session_state.source_type = ""

    st.session_state.last_input_tokens = 0

    st.session_state.last_output_tokens = 0

    st.session_state.last_total_tokens = 0

    st.session_state.chat_initialized = True


# ============================================================
# SAVE CHAT
# ============================================================

def save_chat():

    username = st.session_state.username

    chat_id = st.session_state.current_chat_id

    if not username or not chat_id:

        return

    if username not in chats:

        chats[username] = {}

    user_messages = (
        st.session_state.messages
    )

    first_question = next(
        (
            message["content"]
            for message in user_messages
            if message["role"] == "user"
        ),
        "New Chat"
    )

    title = first_question[:45]

    if len(first_question) > 45:

        title += "..."

    chats[username][chat_id] = {

        "title": title,

        "messages": user_messages,

        "document_text":
            st.session_state.document_text,

        "document_name":
            st.session_state.document_name,

        "source_type":
            st.session_state.source_type,

        "total_tokens":
            st.session_state.last_total_tokens,

        "updated_at":
            datetime.now().isoformat()
    }

    save_json(
        CHATS_FILE,
        chats
    )


# ============================================================
# LOAD CHAT
# ============================================================

def load_chat(chat_id):

    username = st.session_state.username

    user_chats = chats.get(
        username,
        {}
    )

    chat = user_chats.get(
        chat_id
    )

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

    st.session_state.chat_initialized = True

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
# LOGIN / CREATE ACCOUNT
# ============================================================

def show_auth_page():

    st.markdown(
        """
        <div style="
            text-align:center;
            padding-top:60px;
            margin-bottom:40px;
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
        """,
        unsafe_allow_html=True
    )

    login_tab, create_tab = st.tabs(
        [
            "Login",
            "Create Account"
        ]
    )

    # ========================================================
    # LOGIN
    # ========================================================

    with login_tab:

        username = st.text_input(
            "Username",
            key="login_username",
            placeholder="Enter username"
        )

        password = st.text_input(
            "Password",
            type="password",
            key="login_password",
            placeholder="Enter password"
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

                st.session_state.messages = []

                st.session_state.current_chat_id = None

                st.session_state.retriever = None

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
            key="new_username",
            placeholder="Choose username"
        )

        new_password = st.text_input(
            "Password",
            type="password",
            key="new_password",
            placeholder="Choose password"
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
                    "Account created successfully. Please login."
                )


# ============================================================
# LOGIN CHECK
# ============================================================

if not st.session_state.logged_in:

    show_auth_page()

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    # --------------------------------------------------------
    # BRAND
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="brand">
            <span class="brand-star">✦</span>
            AI Document Intelligence
        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # NEW CHAT
    # --------------------------------------------------------

    if st.button(
        "+ New Chat",
        use_container_width=True
    ):

        create_new_chat()

        st.rerun()

    # --------------------------------------------------------
    # CHAT HISTORY TITLE
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="sidebar-title">
            Conversations
        </div>
        """,
        unsafe_allow_html=True
    )

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

            if st.button(
                title,
                key=f"chat_{chat_id}",
                use_container_width=True
            ):

                load_chat(
                    chat_id
                )

                st.rerun()

    else:

        st.caption(
            "No conversations yet."
        )

    # --------------------------------------------------------
    # USAGE TITLE
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="sidebar-title">
            Usage
        </div>
        """,
        unsafe_allow_html=True
    )

    current_usage = get_usage(
        st.session_state.username
    )

    requests_count = current_usage[
        "requests"
    ]

    input_tokens = current_usage[
        "input_tokens"
    ]

    output_tokens = current_usage[
        "output_tokens"
    ]

    total_tokens = current_usage[
        "total_tokens"
    ]

    # --------------------------------------------------------
    # USAGE CARD
    # --------------------------------------------------------

    st.markdown(
        f"""
        <div class="usage-card">

            <div class="usage-header">
                Token Usage
            </div>

            <div class="usage-row">
                <span>Requests</span>
                <span class="usage-number">
                    {requests_count}
                </span>
            </div>

            <div class="usage-row">
                <span>Input tokens</span>
                <span class="usage-number">
                    {input_tokens}
                </span>
            </div>

            <div class="usage-row">
                <span>Output tokens</span>
                <span class="usage-number">
                    {output_tokens}
                </span>
            </div>

            <div class="usage-row">
                <span>Total tokens</span>
                <span class="usage-total">
                    {total_tokens}
                </span>
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    # --------------------------------------------------------
    # LOGOUT
    # --------------------------------------------------------

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
# CREATE CHAT IF NONE EXISTS
# ============================================================

if not st.session_state.current_chat_id:

    create_new_chat()


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    """
    <div class="main-title">
        <span class="main-star">✦</span>
        AI Document Intelligence
    </div>

    <div class="main-subtitle">
        Document & Website Assistant
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SOURCE SECTION
# ============================================================

with st.expander(
    "📎  Add a document or website",
    expanded=not bool(
        st.session_state.document_text
    )
):

    source_option = st.radio(
        "Source",
        [
            "Document",
            "Website URL"
        ],
        horizontal=True,
        label_visibility="collapsed"
    )

    # ========================================================
    # DOCUMENT
    # ========================================================

    if source_option == "Document":

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

                    document_text = (
                        process_document(
                            uploaded_file
                        )
                    )

                    if document_text:

                        st.session_state.document_text = (
                            document_text
                        )

                        st.session_state.document_name = (
                            uploaded_file.name
                        )

                        st.session_state.source_type = (
                            "document"
                        )

                        try:

                            st.session_state.retriever = (
                                create_retriever(
                                    document_text
                                )
                            )

                            st.success(
                                f"Loaded: {uploaded_file.name}"
                            )

                            save_chat()

                        except Exception as error:

                            st.error(
                                f"Could not create search index: {error}"
                            )

    # ========================================================
    # WEBSITE
    # ========================================================

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

                    website_text = read_website(
                        website_url.strip()
                    )

                    if website_text:

                        st.session_state.document_text = (
                            website_text
                        )

                        st.session_state.document_name = (
                            website_url.strip()
                        )

                        st.session_state.source_type = (
                            "website"
                        )

                        try:

                            st.session_state.retriever = (
                                create_retriever(
                                    website_text
                                )
                            )

                            st.success(
                                "Website loaded successfully."
                            )

                            save_chat()

                        except Exception as error:

                            st.error(
                                f"Could not create search index: {error}"
                            )


# ============================================================
# CHAT HISTORY DISPLAY
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
        <div class="welcome">

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
# TOKEN USAGE ABOVE KEYBOARD
# ============================================================

st.markdown(
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
    """,
    unsafe_allow_html=True
)


# ============================================================
# CHAT INPUT
# ============================================================

user_question = st.chat_input(
    "Ask anything about your document..."
)


# ============================================================
# QUESTION PROCESSING
# ============================================================

if user_question:

    # ========================================================
    # SAVE USER MESSAGE
    # ========================================================

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question
        }
    )

    with st.chat_message("user"):

        st.markdown(
            user_question
        )

    # ========================================================
    # CHECK RETRIEVER
    # ========================================================

    if not st.session_state.retriever:

        answer = (
            "Please upload a document or load a website "
            "before asking a question."
        )

        with st.chat_message(
            "assistant"
        ):

            st.markdown(
                answer
            )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        save_chat()

        st.rerun()

    # ========================================================
    # LOAD MODEL + ANSWER
    # ========================================================

    with st.spinner(
        "Thinking..."
    ):

        try:

            tokenizer, generator = (
                load_model()
            )

            # ------------------------------------------------
            # RETRIEVE CONTEXT
            # ------------------------------------------------

            relevant_documents = (
                st.session_state.retriever.invoke(
                    user_question
                )
            )

            context_parts = []

            for document in relevant_documents:

                context_parts.append(
                    document.page_content
                )

            context = "\n\n".join(
                context_parts
            )

            # ------------------------------------------------
            # PROMPT
            # ------------------------------------------------

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

            # ------------------------------------------------
            # INPUT TOKEN COUNT
            # ------------------------------------------------

            input_token_count = count_tokens(
                tokenizer,
                prompt
            )

            # ------------------------------------------------
            # GENERATE ANSWER
            # ------------------------------------------------

            result = generator(
                prompt,
                max_new_tokens=100,
                do_sample=False,
                return_full_text=False,
                pad_token_id=tokenizer.eos_token_id
            )

            answer = result[0][
                "generated_text"
            ].strip()

            # ------------------------------------------------
            # CLEAN ANSWER
            # ------------------------------------------------

            if "ANSWER:" in answer:

                answer = answer.split(
                    "ANSWER:",
                    1
                )[-1].strip()

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
            # OUTPUT TOKEN COUNT
            # ------------------------------------------------

            output_token_count = count_tokens(
                tokenizer,
                answer
            )

            total_token_count = (
                input_token_count
                + output_token_count
            )

            # ------------------------------------------------
            # SAVE TOKEN COUNTS
            # ------------------------------------------------

            st.session_state.last_input_tokens = (
                input_token_count
            )

            st.session_state.last_output_tokens = (
                output_token_count
            )

            st.session_state.last_total_tokens = (
                total_token_count
            )

            # ------------------------------------------------
            # UPDATE USAGE
            # ------------------------------------------------

            update_usage(
                st.session_state.username,
                input_token_count,
                output_token_count
            )

        except Exception as error:

            answer = (
                "Sorry, I couldn't process your question.\n\n"
                f"Error: {error}"
            )

    # ========================================================
    # SHOW ASSISTANT ANSWER
    # ========================================================

    with st.chat_message(
        "assistant"
    ):

        st.markdown(
            answer
        )

    # ========================================================
    # SAVE ASSISTANT MESSAGE
    # ========================================================

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    # ========================================================
    # SAVE CHAT
    # ========================================================

    save_chat()

    # ========================================================
    # REFRESH
    # ========================================================

    st.rerun()

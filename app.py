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
    initial_sidebar_state="expanded"
)


# ============================================================
# FILES
# ============================================================

USERS_FILE = "users.json"
CHATS_FILE = "chats.json"
USAGE_FILE = "usage.json"


# ============================================================
# DARK CLAUDE-STYLE CSS
# ============================================================

st.markdown(
    """
<style>

html, body, [class*="css"] {
    font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI",
                 sans-serif;
}

.stApp {
    background: #1f1e1b;
    color: #f5f5f0;
}

/* Main content */

.main .block-container {
    max-width: 1100px;
    padding-top: 2rem;
    padding-bottom: 8rem;
}

/* Hide Streamlit decoration */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header[data-testid="stHeader"] {
    background: transparent;
}


/* ============================================================
   SIDEBAR
   ============================================================ */

section[data-testid="stSidebar"] {
    background: #171614 !important;
    border-right: 1px solid #35332f;
}

section[data-testid="stSidebar"] > div {
    background: #171614 !important;
}

.sidebar-title {
    font-size: 18px;
    font-weight: 700;
    color: #f5f1e8;
    padding: 8px 0 18px 0;
}

.sidebar-star {
    color: #f0a45d;
    font-size: 21px;
}

.sidebar-section {
    color: #e8e3da;
    font-size: 14px;
    font-weight: 700;
    margin-top: 28px;
    margin-bottom: 10px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}


/* ============================================================
   BUTTONS
   ============================================================ */

.stButton > button {
    width: 100%;
    border-radius: 9px;
    border: 1px solid #4a4741;
    background: #2a2824;
    color: #f5f1e8;
    min-height: 42px;
    font-weight: 500;
}

.stButton > button:hover {
    border-color: #f0a45d;
    color: #ffffff;
    background: #312e29;
}

.new-chat-button button {
    background: #2c2925 !important;
    border: 1px solid #514d46 !important;
}

.new-chat-button button:hover {
    background: #37332d !important;
    border-color: #f0a45d !important;
}


/* ============================================================
   APP HEADER
   ============================================================ */

.app-header {
    text-align: center;
    padding: 5px 0 25px 0;
}

.app-title {
    font-size: 31px;
    font-weight: 750;
    color: #f5f3ed;
    letter-spacing: -0.7px;
}

.app-title-star {
    color: #f0a45d;
}

.app-subtitle {
    color: #aaa59d;
    font-size: 14px;
    margin-top: 6px;
}


/* ============================================================
   LOGIN PAGE
   ============================================================ */

.login-wrapper {
    max-width: 680px;
    margin: 70px auto 0 auto;
}

.login-title {
    text-align: center;
    font-size: 32px;
    font-weight: 750;
    color: #f5f3ed;
    margin-bottom: 8px;
}

.login-subtitle {
    text-align: center;
    color: #aaa59d;
    font-size: 15px;
    margin-bottom: 45px;
}

.auth-label {
    color: #e8e3da;
    font-size: 14px;
    margin-bottom: 7px;
    font-weight: 600;
}


/* ============================================================
   INPUTS
   ============================================================ */

.stTextInput > div > div > input,
.stTextArea textarea {
    background: #292824 !important;
    color: #f5f3ed !important;
    border: 1px solid #4a4741 !important;
    border-radius: 10px !important;
}

.stTextInput > div > div > input:focus,
.stTextArea textarea:focus {
    border-color: #f0a45d !important;
    box-shadow: 0 0 0 1px #f0a45d !important;
}

.stTextInput label,
.stTextArea label {
    color: #ddd8cf !important;
}


/* ============================================================
   LOGIN BUTTON
   ============================================================ */

.login-button button {
    background: #f0a45d !important;
    color: #1f1e1b !important;
    border: none !important;
    font-weight: 700 !important;
}

.login-button button:hover {
    background: #f5b476 !important;
    color: #1f1e1b !important;
}


/* ============================================================
   WELCOME
   ============================================================ */

.welcome-box {
    text-align: center;
    padding: 65px 20px 75px 20px;
}

.welcome-symbol {
    color: #f0a45d;
    font-size: 34px;
    margin-bottom: 15px;
}

.welcome-title {
    color: #f5f3ed;
    font-size: 29px;
    font-weight: 700;
}

.welcome-text {
    color: #aaa59d;
    font-size: 16px;
    margin-top: 10px;
}


/* ============================================================
   SOURCE BOX
   ============================================================ */

.source-box {
    background: #24231f;
    border: 1px solid #494640;
    border-radius: 12px;
    padding: 15px;
    margin-bottom: 20px;
}

.source-title {
    color: #eee9df;
    font-weight: 600;
    font-size: 15px;
}


/* ============================================================
   CHAT MESSAGES
   ============================================================ */

.user-message {
    background: #37342f;
    border: 1px solid #4b4842;
    border-radius: 15px;
    padding: 13px 17px;
    margin: 14px 0 10px auto;
    max-width: 78%;
    color: #f5f3ed;
}

.assistant-message {
    background: transparent;
    padding: 10px 5px 22px 5px;
    color: #e8e3da;
    line-height: 1.65;
    max-width: 90%;
}

.message-label {
    color: #f0a45d;
    font-size: 13px;
    font-weight: 700;
    margin-bottom: 5px;
}


/* ============================================================
   TOKEN BADGE NEAR KEYBOARD
   ============================================================ */

.token-keyboard-info {
    position: fixed;
    right: 105px;
    bottom: 83px;
    z-index: 9999;

    background: #292824;
    border: 1px solid #4d4942;
    border-radius: 8px;

    padding: 5px 10px;

    color: #aaa59d;
    font-size: 11px;

    box-shadow: 0 4px 16px rgba(0,0,0,0.25);
}

.token-keyboard-info strong {
    color: #e8e3da;
}


/* ============================================================
   CHAT INPUT
   ============================================================ */

div[data-testid="stChatInput"] {
    background: #18191f !important;
    border-top: 1px solid #2c2d33 !important;
    padding: 14px 20px 18px 20px !important;
}

div[data-testid="stChatInput"] textarea {
    background: #292824 !important;
    color: #f5f3ed !important;
    border: 1px solid #4b4842 !important;
    border-radius: 15px !important;
}

div[data-testid="stChatInput"] textarea::placeholder {
    color: #8e8980 !important;
}

div[data-testid="stChatInput"] button {
    background: #f0a45d !important;
    color: #1f1e1b !important;
    border-radius: 9px !important;
}

div[data-testid="stChatInput"] button:hover {
    background: #f5b476 !important;
}


/* ============================================================
   USAGE CARD
   ============================================================ */

.usage-card {
    background: #24231f;
    border: 1px solid #3f3c36;
    border-radius: 12px;
    padding: 14px;
}

.usage-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 0;
    color: #aaa59d;
    font-size: 13px;
}

.usage-row strong {
    color: #f0a45d;
    font-weight: 700;
}


/* ============================================================
   CHAT HISTORY
   ============================================================ */

.chat-history-title {
    color: #e7e2d9;
    font-size: 13px;
    padding: 7px 2px;
}


/* ============================================================
   MAX RESPONSE
   ============================================================ */

.max-token-note {
    text-align: center;
    color: #858078;
    font-size: 11px;
    margin-top: 4px;
}


/* ============================================================
   ALERTS
   ============================================================ */

div[data-testid="stAlert"] {
    border-radius: 10px;
}


/* ============================================================
   RADIO
   ============================================================ */

div[data-testid="stRadio"] label {
    color: #ddd8cf !important;
}


/* ============================================================
   FILE UPLOADER
   ============================================================ */

section[data-testid="stFileUploaderDropzone"] {
    background: #292824 !important;
    border: 1px dashed #555149 !important;
}

section[data-testid="stFileUploaderDropzone"] * {
    color: #ddd8cf !important;
}


/* ============================================================
   EXPANDER
   ============================================================ */

details[data-testid="stExpander"] {
    background: #24231f;
    border: 1px solid #46433d;
    border-radius: 11px;
}


/* ============================================================
   SCROLLBAR
   ============================================================ */

::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: #1f1e1b;
}

::-webkit-scrollbar-thumb {
    background: #4a4741;
    border-radius: 10px;
}

::-webkit-scrollbar-thumb:hover {
    background: #625d54;
}

</style>
""",
    unsafe_allow_html=True
)


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
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


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
    "username": None,
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
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# MODEL
# ============================================================

@st.cache_resource
def load_embeddings():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


@st.cache_resource
def load_llm():

    model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name
    )

    text_generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=100,
        do_sample=False,
        return_full_text=False
    )

    return tokenizer, text_generator


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
# USER USAGE
# ============================================================

def get_user_usage(username):

    if username not in usage:
        usage[username] = {
            "requests": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0
        }
        save_json(USAGE_FILE, usage)

    return usage[username]


def update_usage(
    username,
    input_tokens,
    output_tokens
):

    if username not in usage:
        usage[username] = {
            "requests": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0
        }

    total = input_tokens + output_tokens

    usage[username]["requests"] += 1
    usage[username]["input_tokens"] += input_tokens
    usage[username]["output_tokens"] += output_tokens
    usage[username]["total_tokens"] += total

    save_json(USAGE_FILE, usage)


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
# TEXT CLEANING
# ============================================================

def clean_text(text):

    if not text:
        return ""

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# CUSTOM TEXT SPLITTER
# ============================================================

def split_text(text, chunk_size=1000, overlap=200):

    text = clean_text(text)

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

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = end - overlap

    return chunks


# ============================================================
# BUILD RETRIEVER
# ============================================================

def build_retriever(text):

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
# PDF
# ============================================================

def process_pdf(uploaded_file):

    from pypdf import PdfReader

    reader = PdfReader(uploaded_file)

    pages = []

    for page in reader.pages:

        text = page.extract_text()

        if text:
            pages.append(text)

    return "\n".join(pages)


# ============================================================
# TXT
# ============================================================

def process_txt(uploaded_file):

    data = uploaded_file.read()

    try:
        return data.decode("utf-8")

    except UnicodeDecodeError:
        return data.decode(
            "latin-1",
            errors="ignore"
        )


# ============================================================
# DOCX
# ============================================================

def process_docx(uploaded_file):

    document = DocxDocument(uploaded_file)

    paragraphs = []

    for paragraph in document.paragraphs:

        if paragraph.text.strip():
            paragraphs.append(
                paragraph.text
            )

    return "\n".join(paragraphs)


# ============================================================
# XLSX
# ============================================================

def process_xlsx(uploaded_file):

    excel_file = pd.ExcelFile(uploaded_file)

    all_text = []

    for sheet in excel_file.sheet_names:

        df = pd.read_excel(
            uploaded_file,
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
# WEBSITE
# ============================================================

def process_website(url):

    headers = {
        "User-Agent": (
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/131.0 Safari/537.36"
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

    return clean_text(text)


# ============================================================
# RESET CHAT
# ============================================================

def reset_chat():

    st.session_state.current_chat_id = str(
        uuid.uuid4()
    )

    st.session_state.messages = []

    st.session_state.retriever = None

    st.session_state.document_text = ""

    st.session_state.document_name = ""

    st.session_state.source_type = ""

    st.session_state.chat_loaded = False

    st.session_state.last_input_tokens = 0

    st.session_state.last_output_tokens = 0

    st.session_state.last_total_tokens = 0


# ============================================================
# SAVE CHAT
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
                m["content"]
                for m in messages
                if m["role"] == "user"
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
        "total_tokens": sum(
            m.get("total_tokens", 0)
            for m in messages
        ),
        "updated_at": datetime.now().isoformat()
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

    if username not in chats:
        return

    if chat_id not in chats[username]:
        return

    chat = chats[username][chat_id]

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

    if st.session_state.document_text:

        try:

            st.session_state.retriever = (
                build_retriever(
                    st.session_state.document_text
                )
            )

        except Exception as e:

            st.session_state.retriever = None

            st.warning(
                f"Could not rebuild document search: {e}"
            )

    else:

        st.session_state.retriever = None

    st.session_state.chat_loaded = True


# ============================================================
# DELETE CHAT
# ============================================================

def delete_current_chat():

    username = st.session_state.username

    chat_id = st.session_state.current_chat_id

    if (
        username in chats
        and chat_id in chats[username]
    ):

        del chats[username][chat_id]

        save_json(
            CHATS_FILE,
            chats
        )

    reset_chat()


# ============================================================
# LOGIN PAGE
# ============================================================

def authentication_page():

    st.markdown(
        """
<div class="login-wrapper">

    <div class="login-title">
        <span class="app-title-star">✦</span>
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

    with login_tab:

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

        st.markdown(
            '<div class="login-button">',
            unsafe_allow_html=True
        )

        login_clicked = st.button(
            "Login",
            key="login_btn"
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

        if login_clicked:

            if (
                username in users
                and users[username] == password
            ):

                st.session_state.logged_in = True

                st.session_state.username = username

                reset_chat()

                st.rerun()

            else:

                st.error(
                    "Invalid username or password."
                )

    with create_tab:

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

        st.markdown(
            '<div class="login-button">',
            unsafe_allow_html=True
        )

        create_clicked = st.button(
            "Create Account",
            key="create_btn"
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

        if create_clicked:

            if not new_username or not new_password:

                st.warning(
                    "Please enter a username and password."
                )

            elif new_username in users:

                st.error(
                    "Username already exists."
                )

            else:

                users[new_username] = new_password

                save_json(
                    USERS_FILE,
                    users
                )

                usage[new_username] = {
                    "requests": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0
                }

                save_json(
                    USAGE_FILE,
                    usage
                )

                st.success(
                    "Account created successfully. Please login."
                )


# ============================================================
# SHOW AUTH PAGE IF NOT LOGGED IN
# ============================================================

if not st.session_state.logged_in:

    authentication_page()

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
<div class="sidebar-title">
    <span class="sidebar-star">✦</span>
    AI Document Intelligence
</div>
""",
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="new-chat-button">',
        unsafe_allow_html=True
    )

    new_chat_clicked = st.button(
        "+ New Chat",
        key="new_chat_top"
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )

    if new_chat_clicked:

        reset_chat()

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

        for chat_id, chat_data in sorted_chats:

            title = chat_data.get(
                "title",
                "New Chat"
            )

            if st.button(
                title,
                key=f"chat_{chat_id}"
            ):

                load_chat(chat_id)

                st.rerun()

    else:

        st.caption(
            "No conversations yet."
        )

    st.divider()

    st.markdown(
        '<div class="sidebar-section">Usage</div>',
        unsafe_allow_html=True
    )

    user_usage = get_user_usage(
        st.session_state.username
    )

    st.markdown(
        f"""
<div class="usage-card">

    <div class="usage-row">
        <span>Requests</span>
        <strong>{user_usage["requests"]:,}</strong>
    </div>

    <div class="usage-row">
        <span>Input Tokens</span>
        <strong>{user_usage["input_tokens"]:,}</strong>
    </div>

    <div class="usage-row">
        <span>Output Tokens</span>
        <strong>{user_usage["output_tokens"]:,}</strong>
    </div>

    <div class="usage-row">
        <span>Total Tokens</span>
        <strong>{user_usage["total_tokens"]:,}</strong>
    </div>

</div>
""",
        unsafe_allow_html=True
    )

    st.markdown("")

    if st.button(
        "⚙ Settings",
        key="settings_btn"
    ):

        st.info(
            "AI Document Intelligence Settings"
        )

    if st.button(
        "Logout",
        key="logout_btn"
    ):

        st.session_state.logged_in = False

        st.session_state.username = None

        reset_chat()

        st.rerun()


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    """
<div class="app-header">

    <div class="app-title">
        <span class="app-title-star">✦</span>
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
# DELETE CHAT BUTTON
# ============================================================

if st.session_state.messages:

    col1, col2 = st.columns(
        [8, 1]
    )

    with col2:

        if st.button(
            "Delete",
            key="delete_chat"
        ):

            delete_current_chat()

            st.rerun()


# ============================================================
# DOCUMENT / WEBSITE INPUT
# ============================================================

with st.expander(
    "📎  Add a document or website",
    expanded=not bool(
        st.session_state.document_text
    )
):

    source_type = st.radio(
        "Choose source",
        ["Document", "Website URL"],
        horizontal=True,
        key="source_selector"
    )

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

        if uploaded_file is not None:

            if (
                st.session_state.document_name
                != uploaded_file.name
            ):

                try:

                    extension = (
                        uploaded_file.name
                        .lower()
                        .split(".")[-1]
                    )

                    if extension == "pdf":

                        extracted_text = process_pdf(
                            uploaded_file
                        )

                    elif extension == "txt":

                        extracted_text = process_txt(
                            uploaded_file
                        )

                    elif extension == "docx":

                        extracted_text = process_docx(
                            uploaded_file
                        )

                    elif extension == "xlsx":

                        extracted_text = process_xlsx(
                            uploaded_file
                        )

                    else:

                        extracted_text = ""

                    if not extracted_text.strip():

                        st.error(
                            "No readable text was found."
                        )

                    else:

                        with st.spinner(
                            "Processing document..."
                        ):

                            st.session_state.document_text = (
                                extracted_text
                            )

                            st.session_state.document_name = (
                                uploaded_file.name
                            )

                            st.session_state.source_type = (
                                "Document"
                            )

                            st.session_state.retriever = (
                                build_retriever(
                                    extracted_text
                                )
                            )

                        st.success(
                            f"Loaded: {uploaded_file.name}"
                        )

                        save_current_chat()

                except Exception as e:

                    st.error(
                        f"Error processing document: {e}"
                    )

    else:

        website_url = st.text_input(
            "Website URL",
            placeholder="https://example.com",
            key="website_url"
        )

        load_website = st.button(
            "Load Website",
            key="load_website"
        )

        if load_website:

            if not website_url.strip():

                st.warning(
                    "Please enter a website URL."
                )

            else:

                try:

                    with st.spinner(
                        "Reading website..."
                    ):

                        extracted_text = process_website(
                            website_url.strip()
                        )

                        if not extracted_text:

                            raise ValueError(
                                "No readable text found on this website."
                            )

                        st.session_state.document_text = (
                            extracted_text
                        )

                        st.session_state.document_name = (
                            website_url.strip()
                        )

                        st.session_state.source_type = (
                            "Website"
                        )

                        st.session_state.retriever = (
                            build_retriever(
                                extracted_text
                            )
                        )

                    st.success(
                        "Website loaded successfully."
                    )

                    save_current_chat()

                except Exception as e:

                    st.error(
                        f"Could not load website: {e}"
                    )


# ============================================================
# WELCOME MESSAGE
# ============================================================

if not st.session_state.messages:

    greeting = get_greeting()

    st.markdown(
        f"""
<div class="welcome-box">

    <div class="welcome-symbol">
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
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    role = message.get(
        "role",
        ""
    )

    content = message.get(
        "content",
        ""
    )

    if role == "user":

        st.markdown(
            f"""
<div class="user-message">
    {content}
</div>
""",
            unsafe_allow_html=True
        )

    elif role == "assistant":

        st.markdown(
            f"""
<div class="assistant-message">

    <div class="message-label">
        AI Document Intelligence
    </div>

    {content}

</div>
""",
            unsafe_allow_html=True
        )


# ============================================================
# TOKEN INFORMATION NEAR KEYBOARD
# ============================================================

last_input = st.session_state.last_input_tokens
last_output = st.session_state.last_output_tokens
last_total = st.session_state.last_total_tokens

st.markdown(
    f"""
<div class="token-keyboard-info">

    Input: <strong>{last_input:,}</strong>
    &nbsp;•&nbsp;
    Output: <strong>{last_output:,}</strong>
    &nbsp;•&nbsp;
    Total: <strong>{last_total:,}</strong>
    &nbsp;•&nbsp;
    Max: <strong>100</strong>

</div>
""",
    unsafe_allow_html=True
)


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask anything about your document..."
)


# ============================================================
# QUESTION PROCESSING
# ============================================================

if question:

    question = question.strip()

    if not question:
        st.stop()

    # --------------------------------------------------------
    # Check document
    # --------------------------------------------------------

    if not st.session_state.document_text:

        st.warning(
            "Please upload a document or load a website first."
        )

        st.stop()

    # --------------------------------------------------------
    # Add user message
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0
        }
    )

    # --------------------------------------------------------
    # Retrieve relevant context
    # --------------------------------------------------------

    context = ""

    try:

        if st.session_state.retriever:

            docs = (
                st.session_state.retriever.invoke(
                    question
                )
            )

            context = "\n\n".join(
                doc.page_content
                for doc in docs
            )

    except Exception as e:

        st.error(
            f"Search error: {e}"
        )

        st.stop()

    # --------------------------------------------------------
    # Prompt
    # --------------------------------------------------------

    prompt = f"""
You are an AI document assistant.

Answer the user's question using ONLY the provided context.

If the answer is not available in the context, say:

"I could not find that information in the document."

Be clear and concise.

Context:
{context}

User question:
{question}

Answer:
"""

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    try:

        with st.spinner(
            "Thinking..."
        ):

            tokenizer, text_generator = load_llm()

            input_tokens = count_tokens(
                tokenizer,
                prompt
            )

            result = text_generator(
                prompt,
                max_new_tokens=100,
                do_sample=False
            )

            answer = result[0]["generated_text"].strip()

            output_tokens = count_tokens(
                tokenizer,
                answer
            )

            total_tokens = (
                input_tokens
                + output_tokens
            )

    except Exception as e:

        st.error(
            f"AI generation error: {e}"
        )

        st.stop()

    # --------------------------------------------------------
    # Update token state
    # --------------------------------------------------------

    st.session_state.last_input_tokens = (
        input_tokens
    )

    st.session_state.last_output_tokens = (
        output_tokens
    )

    st.session_state.last_total_tokens = (
        total_tokens
    )

    # --------------------------------------------------------
    # Add assistant message
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens
        }
    )

    # --------------------------------------------------------
    # Update usage
    # --------------------------------------------------------

    update_usage(
        st.session_state.username,
        input_tokens,
        output_tokens
    )

    # --------------------------------------------------------
    # Save chat
    # --------------------------------------------------------

    save_current_chat()

    # --------------------------------------------------------
    # Refresh
    # --------------------------------------------------------

    st.rerun()

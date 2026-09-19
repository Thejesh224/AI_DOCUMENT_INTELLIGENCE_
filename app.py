import streamlit as st
import os
import json
import uuid
import requests
import pandas as pd

from datetime import datetime

from bs4 import BeautifulSoup
from docx import Document as DocxDocument

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline
)


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

DEFAULT_MAX_TOKENS = 100


# ============================================================
# DARK CLAUDE-INSPIRED THEME
# ============================================================

st.markdown(
    """
<style>

/* ==========================================================
   GLOBAL
========================================================== */

.stApp {
    background-color: #1f1e1b !important;
    color: #f2efe8 !important;
}

[data-testid="stAppViewContainer"] {
    background-color: #1f1e1b !important;
}

[data-testid="stMain"] {
    background-color: #1f1e1b !important;
}

.main {
    background-color: #1f1e1b !important;
}

.block-container {
    max-width: 920px !important;
    padding-top: 25px !important;
    padding-bottom: 140px !important;
}


/* ==========================================================
   HIDE STREAMLIT DEFAULT ELEMENTS
========================================================== */

#MainMenu {
    visibility: hidden;
}

header {
    visibility: hidden;
}

footer {
    visibility: hidden;
}


/* ==========================================================
   GENERAL TEXT
========================================================== */

.stApp,
.stApp p,
.stApp span,
.stApp label,
.stApp div {
    color: #f2efe8;
}

h1,
h2,
h3,
h4,
h5,
h6 {
    color: #f2efe8 !important;
}


/* ==========================================================
   SIDEBAR
========================================================== */

section[data-testid="stSidebar"] {
    background-color: #171614 !important;
    border-right: 1px solid #34312c !important;
}

section[data-testid="stSidebar"] > div {
    background-color: #171614 !important;
}

section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label {
    color: #ddd8cf !important;
}


/* ==========================================================
   SIDEBAR BUTTONS
========================================================== */

section[data-testid="stSidebar"] button {
    background-color: transparent !important;
    color: #ddd8cf !important;
    border: 1px solid transparent !important;
    border-radius: 9px !important;
    text-align: left !important;
}

section[data-testid="stSidebar"] button:hover {
    background-color: #292722 !important;
    border-color: #3a3731 !important;
}


/* ==========================================================
   MAIN HEADER
========================================================== */

.app-header {
    padding-bottom: 16px;
    border-bottom: 1px solid #36332e;
    margin-bottom: 25px;
}

.app-title {
    font-size: 19px;
    font-weight: 600;
    color: #f4f0e8 !important;
}

.app-subtitle {
    font-size: 12px;
    color: #969087 !important;
    margin-top: 3px;
}


/* ==========================================================
   LOGIN PAGE
========================================================== */

.login-container {
    max-width: 460px;
    margin: 100px auto 0 auto;
}

.login-title {
    text-align: center;
    font-size: 32px;
    font-weight: 650;
    color: #f5f1e9 !important;
    margin-bottom: 8px;
}

.login-subtitle {
    text-align: center;
    font-size: 14px;
    color: #9b958b !important;
    margin-bottom: 35px;
}


/* ==========================================================
   LOGIN TABS
========================================================== */

button[data-baseweb="tab"] {
    color: #9e988e !important;
    background: transparent !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #f3eee5 !important;
}


/* ==========================================================
   INPUTS
========================================================== */

.stTextInput label {
    color: #d9d3ca !important;
}

.stTextInput input {
    background-color: #292824 !important;
    color: #f4f0e8 !important;
    border: 1px solid #48443d !important;
    border-radius: 10px !important;
}

.stTextInput input::placeholder {
    color: #777169 !important;
}

.stTextInput input:focus {
    border-color: #b6aea2 !important;
    box-shadow: none !important;
}


/* ==========================================================
   BUTTONS
========================================================== */

.stButton button {
    background-color: #d97757 !important;

    color: #ffffff !important;

    border: none !important;

    border-radius: 9px !important;

    min-height: 44px !important;

    font-weight: 600 !important;

    transition: all 0.2s ease !important;
}

.stButton button:hover {
    background-color: #e48768 !important;

    color: #ffffff !important;

    border: none !important;
}

.stButton button:active {
    background-color: #c9684b !important;

    color: #ffffff !important;
}


/* ==========================================================
   FILE UPLOADER
========================================================== */

[data-testid="stFileUploader"] {
    background-color: #272521 !important;
    border: 1px solid #3d3933 !important;
    border-radius: 12px !important;
    padding: 10px !important;
}

[data-testid="stFileUploader"] * {
    color: #ddd8cf !important;
}


/* ==========================================================
   EXPANDER
========================================================== */

[data-testid="stExpander"] {
    background-color: #24231f !important;
    border: 1px solid #3b3832 !important;
    border-radius: 12px !important;
}

[data-testid="stExpander"] * {
    color: #ddd8cf !important;
}


/* ==========================================================
   SELECT / RADIO
========================================================== */

div[data-baseweb="select"] {
    background-color: #292824 !important;
    border-radius: 9px !important;
}

div[data-baseweb="select"] * {
    color: #eee9e0 !important;
}

div[data-testid="stRadio"] label {
    color: #ddd8cf !important;
}


/* ==========================================================
   WELCOME SCREEN
========================================================== */

.welcome {
    text-align: center;
    padding-top: 125px;
    padding-bottom: 70px;
}

.welcome-symbol {
    font-size: 34px;
    margin-bottom: 20px;
    color: #eee7dc !important;
}

.welcome-title {
    font-size: 30px;
    font-weight: 600;
    color: #f4f0e8 !important;
    margin-bottom: 8px;
}

.welcome-text {
    font-size: 15px;
    color: #969087 !important;
}


/* ==========================================================
   SOURCE BADGE
========================================================== */

.source-badge {
    display: inline-block;
    background-color: #2b2925;
    border: 1px solid #3b3832;
    color: #b9b1a6 !important;
    border-radius: 20px;
    padding: 5px 11px;
    font-size: 11px;
    margin-bottom: 12px;
}


/* ==========================================================
   USER CHAT
========================================================== */

.user-message {
    background-color: #34312b !important;
    color: #f1ece4 !important;
    padding: 12px 16px;
    border-radius: 18px 18px 5px 18px;
    margin: 18px 0 18px auto;
    max-width: 72%;
    line-height: 1.6;
    font-size: 15px;
}


/* ==========================================================
   AI CHAT
========================================================== */

.ai-message {
    background-color: transparent !important;
    color: #e8e2d9 !important;
    max-width: 85%;
    line-height: 1.75;
    font-size: 15px;
    margin: 25px 0;
}

.ai-label {
    color: #a59e94 !important;
    font-size: 12px;
    font-weight: 600;
    margin-bottom: 6px;
}


/* ==========================================================
   CHAT INPUT
========================================================== */

[data-testid="stChatInput"] {
    background: transparent !important;
}

[data-testid="stChatInput"] > div {
    background-color: #292824 !important;
    border: 1px solid #48443d !important;
    border-radius: 18px !important;
    box-shadow: 0 5px 25px rgba(0, 0, 0, 0.25) !important;
}

[data-testid="stChatInput"] textarea {
    background-color: #292824 !important;
    color: #f4efe7 !important;
    border: none !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: #858077 !important;
}


/* ==========================================================
   TOKEN LIMIT
========================================================== */

.token-info {
    text-align: right;
    color: #817b72 !important;
    font-size: 11px;
    margin-bottom: 5px;
    padding-right: 8px;
}


/* ==========================================================
   USAGE CARD
========================================================== */

.usage-card {
    background-color: #25231f;
    border: 1px solid #37342e;
    border-radius: 10px;
    padding: 12px;
    color: #aaa39a !important;
    font-size: 12px;
    line-height: 1.8;
}


/* ==========================================================
   SETTINGS
========================================================== */

.settings-box {
    background-color: #25231f;
    border: 1px solid #3a3731;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 20px;
}


/* ==========================================================
   DIVIDER
========================================================== */

hr {
    border-color: #36332e !important;
}


/* ==========================================================
   MOBILE
========================================================== */

@media (max-width: 768px) {

    .login-container {
        margin-top: 60px;
    }

    .user-message {
        max-width: 88%;
    }

    .ai-message {
        max-width: 94%;
    }

}

</style>
""",
    unsafe_allow_html=True
)


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

defaults = {
    "logged_in": False,
    "username": None,
    "current_chat_id": None,
    "messages": [],
    "retriever": None,
    "document_text": "",
    "document_name": "",
    "source_type": "",
    "show_settings": False,
    "response_length": DEFAULT_MAX_TOKENS
}

for key, value in defaults.items():

    if key not in st.session_state:

        st.session_state[key] = value


# ============================================================
# TOKEN COUNTER
# ============================================================

def count_tokens(tokenizer, text):

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
# EMBEDDINGS
# ============================================================

@st.cache_resource
def load_embeddings():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# ============================================================
# LANGUAGE MODEL
# ============================================================

@st.cache_resource
def load_llm():

    model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name
    )

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=DEFAULT_MAX_TOKENS,
        do_sample=False,
        return_full_text=False
    )

    return tokenizer, generator


# ============================================================
# GREETING
# ============================================================

def get_greeting():

    hour = datetime.now().hour

    if hour < 12:

        return "Good Morning"

    if hour < 17:

        return "Good Afternoon"

    return "Good Evening"


# ============================================================
# NEW CHAT
# ============================================================

def create_new_chat():

    st.session_state.current_chat_id = None

    st.session_state.messages = []

    st.session_state.retriever = None

    st.session_state.document_text = ""

    st.session_state.document_name = ""

    st.session_state.source_type = ""

    st.rerun()


# ============================================================
# BUILD RETRIEVER
# ============================================================

def build_retriever(text):

    if not text:

        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    documents = [
        Document(
            page_content=text
        )
    ]

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
            "k": 2
        }
    )


# ============================================================
# SAVE CHAT
# ============================================================

def save_current_chat():

    if not st.session_state.messages:

        return

    if not st.session_state.current_chat_id:

        st.session_state.current_chat_id = str(
            uuid.uuid4()
        )

    chat_id = st.session_state.current_chat_id

    title = "New Chat"

    for message in st.session_state.messages:

        if message["role"] == "user":

            title = message["content"].strip()

            if len(title) > 35:

                title = title[:35] + "..."

            break

    total_tokens = sum(
        message.get("tokens", 0)
        for message in st.session_state.messages
    )

    chats[chat_id] = {
        "username":
            st.session_state.username,

        "title":
            title,

        "messages":
            st.session_state.messages,

        "document_text":
            st.session_state.document_text,

        "document_name":
            st.session_state.document_name,

        "source_type":
            st.session_state.source_type,

        "total_tokens":
            total_tokens,

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

    if st.session_state.document_text:

        st.session_state.retriever = (
            build_retriever(
                st.session_state.document_text
            )
        )

    else:

        st.session_state.retriever = None

    st.rerun()


# ============================================================
# DELETE CHAT
# ============================================================

def delete_current_chat():

    chat_id = st.session_state.current_chat_id

    if chat_id and chat_id in chats:

        del chats[chat_id]

        save_json(
            CHATS_FILE,
            chats
        )

    create_new_chat()


# ============================================================
# LOGIN
# ============================================================

def login_user(username, password):

    username = username.strip()

    if (
        username in users
        and users[username]["password"] == password
    ):

        st.session_state.logged_in = True

        st.session_state.username = username

        if username not in usage:

            usage[username] = {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "requests": 0
            }

            save_json(
                USAGE_FILE,
                usage
            )

        st.rerun()

    else:

        st.error(
            "Invalid username or password."
        )


# ============================================================
# SIGNUP
# ============================================================

def signup_user(username, password):

    username = username.strip()

    if not username or not password:

        st.error(
            "Please enter username and password."
        )

        return

    if username in users:

        st.error(
            "Username already exists."
        )

        return

    users[username] = {
        "password": password,
        "created_at":
            datetime.now().isoformat()
    }

    save_json(
        USERS_FILE,
        users
    )

    usage[username] = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "requests": 0
    }

    save_json(
        USAGE_FILE,
        usage
    )

    st.success(
        "Account created successfully. Please login."
    )


# ============================================================
# PROCESS PDF
# ============================================================

def process_pdf(uploaded_file):

    path = "temporary_document.pdf"

    with open(
        path,
        "wb"
    ) as file:

        file.write(
            uploaded_file.getbuffer()
        )

    loader = PyPDFLoader(
        path
    )

    pages = loader.load()

    return "\n\n".join(
        page.page_content
        for page in pages
    )


# ============================================================
# PROCESS TXT
# ============================================================

def process_txt(uploaded_file):

    return uploaded_file.getvalue().decode(
        "utf-8",
        errors="ignore"
    )


# ============================================================
# PROCESS DOCX
# ============================================================

def process_docx(uploaded_file):

    document = DocxDocument(
        uploaded_file
    )

    text = []

    for paragraph in document.paragraphs:

        value = paragraph.text.strip()

        if value:

            text.append(value)

    return "\n".join(text)


# ============================================================
# PROCESS XLSX
# ============================================================

def process_xlsx(uploaded_file):

    excel = pd.ExcelFile(
        uploaded_file
    )

    result = []

    for sheet in excel.sheet_names:

        dataframe = pd.read_excel(
            uploaded_file,
            sheet_name=sheet
        )

        result.append(
            f"Sheet: {sheet}"
        )

        result.append(
            dataframe.to_string(
                index=False
            )
        )

    return "\n\n".join(result)


# ============================================================
# PROCESS WEBSITE
# ============================================================

def process_website(url):

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

    lines = []

    for line in soup.get_text(
        separator="\n"
    ).splitlines():

        line = line.strip()

        if line:

            lines.append(line)

    return "\n".join(lines)


# ============================================================
# LOGIN SCREEN
# ============================================================

if not st.session_state.logged_in:

    st.markdown(
        '<div class="login-container">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="login-title">'
        'AI Document Intelligence'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="login-subtitle">'
        'Ask questions about your documents and websites.'
        '</div>',
        unsafe_allow_html=True
    )

    login_tab, signup_tab = st.tabs(
        [
            "Login",
            "Create Account"
        ]
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

        if st.button(
            "Login",
            use_container_width=True
        ):

            login_user(
                username,
                password
            )

    with signup_tab:

        new_username = st.text_input(
            "Username",
            placeholder="Choose a username",
            key="signup_username"
        )

        new_password = st.text_input(
            "Password",
            placeholder="Create a password",
            type="password",
            key="signup_password"
        )

        if st.button(
            "Create Account",
            use_container_width=True
        ):

            signup_user(
                new_username,
                new_password
            )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "### ✦ AI Document Intelligence"
    )

    if st.button(
        "＋  New Chat",
        use_container_width=True
    ):

        create_new_chat()

    st.divider()

    st.markdown(
        "### Chats"
    )

    user_chats = []

    for chat_id, chat in chats.items():

        if (
            chat.get("username")
            == st.session_state.username
        ):

            user_chats.append(
                (
                    chat_id,
                    chat
                )
            )

    user_chats.sort(
        key=lambda x:
        x[1].get(
            "updated_at",
            ""
        ),
        reverse=True
    )

    if not user_chats:

        st.caption(
            "No conversations yet."
        )

    for chat_id, chat in user_chats:

        title = chat.get(
            "title",
            "New Chat"
        )

        if len(title) > 32:

            title = title[:32] + "..."

        if st.button(
            "▸ " + title,
            key=f"chat_{chat_id}",
            use_container_width=True
        ):

            load_chat(
                chat_id
            )

    st.divider()

    # ========================================================
    # USAGE
    # ========================================================

    user_usage = usage.get(
        st.session_state.username,
        {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "requests": 0
        }
    )

    st.markdown(
        "### Usage"
    )

    st.markdown(
        f"""
        <div class="usage-card">
        Requests: <b>{user_usage.get("requests", 0)}</b><br>
        Input: <b>{user_usage.get("input_tokens", 0)}</b><br>
        Output: <b>{user_usage.get("output_tokens", 0)}</b><br>
        Total: <b>{user_usage.get("total_tokens", 0)}</b>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    # ========================================================
    # SETTINGS
    # ========================================================

    if st.button(
        "⚙ Settings",
        use_container_width=True
    ):

        st.session_state.show_settings = (
            not st.session_state.show_settings
        )

        st.rerun()

    # ========================================================
    # LOGOUT
    # ========================================================

    if st.button(
        "Logout",
        use_container_width=True
    ):

        st.session_state.clear()

        st.rerun()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="app-header">

        <div class="app-title">
            ✦ AI Document Intelligence
        </div>

        <div class="app-subtitle">
            Document & Website Assistant
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SETTINGS
# ============================================================

if st.session_state.show_settings:

    st.markdown(
        "### ⚙ Chat Settings"
    )

    st.markdown(
        '<div class="settings-box">',
        unsafe_allow_html=True
    )

    st.session_state.response_length = st.selectbox(
        "Maximum response tokens",
        [50, 100, 150, 200],
        index=1
    )

    st.caption(
        "This controls the maximum length of the AI response."
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
# DOCUMENT / WEBSITE
# ============================================================

with st.expander(
    "📎 Add a document or website"
):

    source = st.radio(
        "Source",
        [
            "Document",
            "Website URL"
        ],
        horizontal=True
    )

    # ========================================================
    # DOCUMENT
    # ========================================================

    if source == "Document":

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
                uploaded_file.name
                != st.session_state.document_name
            ):

                with st.spinner(
                    "Reading document..."
                ):

                    try:

                        extension = (
                            uploaded_file.name
                            .split(".")[-1]
                            .lower()
                        )

                        if extension == "pdf":

                            text = process_pdf(
                                uploaded_file
                            )

                        elif extension == "txt":

                            text = process_txt(
                                uploaded_file
                            )

                        elif extension == "docx":

                            text = process_docx(
                                uploaded_file
                            )

                        elif extension == "xlsx":

                            text = process_xlsx(
                                uploaded_file
                            )

                        else:

                            st.error(
                                "Unsupported file."
                            )

                            st.stop()

                        if not text.strip():

                            st.error(
                                "No readable text found."
                            )

                        else:

                            st.session_state.document_text = text

                            st.session_state.document_name = (
                                uploaded_file.name
                            )

                            st.session_state.source_type = (
                                "Document"
                            )

                            st.session_state.retriever = (
                                build_retriever(
                                    text
                                )
                            )

                            st.success(
                                f"✓ {uploaded_file.name} is ready."
                            )

                    except Exception as error:

                        st.error(
                            f"Error: {error}"
                        )

    # ========================================================
    # WEBSITE
    # ========================================================

    else:

        url = st.text_input(
            "Website URL",
            placeholder="https://example.com"
        )

        if st.button(
            "Load Website",
            use_container_width=True
        ):

            if not url.strip():

                st.warning(
                    "Please enter a URL."
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
                                "No readable text found."
                            )

                        else:

                            st.session_state.document_text = text

                            st.session_state.document_name = (
                                url.strip()
                            )

                            st.session_state.source_type = (
                                "Website"
                            )

                            st.session_state.retriever = (
                                build_retriever(
                                    text
                                )
                            )

                            st.success(
                                "✓ Website is ready."
                            )

                    except Exception as error:

                        st.error(
                            f"Could not load website: {error}"
                        )


# ============================================================
# SOURCE
# ============================================================

if st.session_state.document_text:

    st.markdown(
        f"""
        <div class="source-badge">
            📄 {st.session_state.document_name}
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# WELCOME
# ============================================================

if not st.session_state.messages:

    greeting = get_greeting()

    st.markdown(
        f"""
        <div class="welcome">

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
# DISPLAY CHAT
# ============================================================

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


# ============================================================
# TOKEN INFORMATION
# ============================================================

st.markdown(
    f"""
    <div class="token-info">
        Max response: {st.session_state.response_length} tokens
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

    prompt = prompt.strip()

    if not prompt:

        st.stop()

    # ========================================================
    # USER MESSAGE
    # ========================================================

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
            "tokens": 0
        }
    )

    # ========================================================
    # AI
    # ========================================================

    with st.spinner(
        "Thinking..."
    ):

        try:

            tokenizer, generator = load_llm()

            input_tokens = count_tokens(
                tokenizer,
                prompt
            )

            # =================================================
            # RETRIEVE CONTEXT
            # =================================================

            context = ""

            if st.session_state.retriever:

                try:

                    documents = (
                        st.session_state.retriever
                        .invoke(prompt)
                    )

                    context = "\n\n".join(
                        doc.page_content
                        for doc in documents
                    )

                except Exception:

                    context = ""

            # =================================================
            # RAG PROMPT
            # =================================================

            if context:

                final_prompt = f"""
You are an AI document assistant.

Use ONLY the context below to answer.

Do not invent information.

If the answer is not available, say:

"I could not find that information in the document."

Context:
{context}

Question:
{prompt}

Answer:
"""

            else:

                final_prompt = f"""
You are a helpful AI assistant.

Answer the question clearly and briefly.

Question:
{prompt}

Answer:
"""

            # =================================================
            # GENERATE
            # =================================================

            result = generator(
                final_prompt,
                max_new_tokens=st.session_state.response_length,
                do_sample=False,
                return_full_text=False
            )

            answer = (
                result[0]
                .get(
                    "generated_text",
                    ""
                )
                .strip()
            )

            if not answer:

                answer = (
                    "I could not generate an answer."
                )

            output_tokens = count_tokens(
                tokenizer,
                answer
            )

            total_tokens = (
                input_tokens
                + output_tokens
            )

        except Exception as error:

            input_tokens = 0

            output_tokens = 0

            total_tokens = 0

            answer = (
                f"Sorry, I encountered an error: {error}"
            )

    # ========================================================
    # UPDATE USER TOKEN COUNT
    # ========================================================

    username = st.session_state.username

    if username not in usage:

        usage[username] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "requests": 0
        }

    usage[username]["input_tokens"] += (
        input_tokens
    )

    usage[username]["output_tokens"] += (
        output_tokens
    )

    usage[username]["total_tokens"] += (
        total_tokens
    )

    usage[username]["requests"] += 1

    save_json(
        USAGE_FILE,
        usage
    )

    # ========================================================
    # SAVE AI MESSAGE
    # ========================================================

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "tokens": output_tokens
        }
    )

    # ========================================================
    # SAVE CHAT
    # ========================================================

    save_current_chat()

    st.rerun()


# ============================================================
# CHAT ACTIONS
# ============================================================

if st.session_state.current_chat_id:

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "🗑 Delete Chat",
            use_container_width=True
        ):

            delete_current_chat()

    with col2:

        if st.button(
            "＋ New Chat",
            use_container_width=True
        ):

            create_new_chat()

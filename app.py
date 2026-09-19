import streamlit as st
import os
import json
import uuid
import html
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
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Document Intelligence",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# FILE NAMES
# ============================================================

USERS_FILE = "users.json"
CHATS_FILE = "chats.json"
USAGE_FILE = "usage.json"

DEFAULT_MAX_TOKENS = 100


# ============================================================
# LIGHT CLAUDE-STYLE DESIGN
# ============================================================

st.markdown(
    """
<style>

/* ==========================================================
   GLOBAL
========================================================== */

.stApp {
    background-color: #f7f7f5 !important;
    color: #292925 !important;
}

.main {
    background-color: #f7f7f5 !important;
}

.block-container {
    max-width: 1050px !important;
    padding-top: 25px !important;
    padding-bottom: 150px !important;
}


/* ==========================================================
   HIDE STREAMLIT DEFAULTS
========================================================== */

#MainMenu {
    visibility: hidden !important;
}

header {
    visibility: hidden !important;
}

footer {
    visibility: hidden !important;
}


/* ==========================================================
   ALL TEXT
========================================================== */

.stApp p,
.stApp label,
.stApp span {
    color: #292925;
}

h1,
h2,
h3,
h4,
h5,
h6 {
    color: #292925 !important;
}


/* ==========================================================
   SIDEBAR
========================================================== */

section[data-testid="stSidebar"] {
    background-color: #eeeeea !important;
    border-right: 1px solid #ddddD6 !important;
}

section[data-testid="stSidebar"] > div {
    background-color: #eeeeea !important;
}

section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label {
    color: #292925 !important;
}


/* Sidebar buttons */

section[data-testid="stSidebar"] button {
    background-color: transparent !important;
    color: #292925 !important;
    border: none !important;
    border-radius: 9px !important;
}

section[data-testid="stSidebar"] button:hover {
    background-color: #e0e0da !important;
}


/* ==========================================================
   MAIN HEADER
========================================================== */

.app-header {
    display: flex;
    justify-content: space-between;
    align-items: center;

    padding: 5px 0 18px 0;

    border-bottom: 1px solid #deded7;

    margin-bottom: 25px;
}

.app-header-left {
    display: flex;
    align-items: center;
    gap: 10px;
}

.app-logo {
    width: 36px;
    height: 36px;

    border-radius: 10px;

    background-color: #292925;
    color: #ffffff !important;

    display: flex;
    align-items: center;
    justify-content: center;

    font-size: 19px;
    font-weight: 600;
}

.app-title {
    font-size: 18px;
    font-weight: 600;
    color: #292925 !important;
}

.app-status {
    font-size: 12px;
    color: #777770 !important;
}


/* ==========================================================
   LOGIN PAGE
========================================================== */

.login-wrapper {
    max-width: 520px;

    margin: 65px auto 30px auto;

    text-align: center;
}

.login-logo {
    width: 70px;
    height: 70px;

    margin: 0 auto 20px auto;

    border-radius: 18px;

    background-color: #292925;
    color: #ffffff !important;

    display: flex;
    align-items: center;
    justify-content: center;

    font-size: 34px;
    font-weight: 600;
}

.login-title {
    color: #292925 !important;

    font-size: 32px;
    font-weight: 700;

    margin-bottom: 10px;
}

.login-subtitle {
    color: #66665f !important;

    font-size: 16px;

    margin-bottom: 35px;
}


/* ==========================================================
   TABS
========================================================== */

button[data-baseweb="tab"] {
    color: #66665f !important;
    background-color: transparent !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #292925 !important;
}


/* ==========================================================
   TEXT INPUTS
========================================================== */

.stTextInput label {
    color: #292925 !important;
}

.stTextInput input {
    background-color: #ffffff !important;

    color: #292925 !important;

    border: 1px solid #cfcfc7 !important;

    border-radius: 10px !important;
}

.stTextInput input::placeholder {
    color: #999990 !important;
}

.stTextInput input:focus {
    border-color: #8f8f87 !important;

    box-shadow: none !important;
}


/* ==========================================================
   BUTTONS
========================================================== */

.stButton button {
    background-color: #292925 !important;

    color: #ffffff !important;

    border: none !important;

    border-radius: 10px !important;

    min-height: 42px !important;

    font-weight: 500 !important;
}

.stButton button:hover {
    background-color: #44443d !important;

    color: #ffffff !important;
}


/* ==========================================================
   FILE UPLOADER
========================================================== */

[data-testid="stFileUploader"] {
    background-color: #ffffff !important;

    border: 1px solid #deded7 !important;

    border-radius: 12px !important;

    padding: 10px !important;
}

[data-testid="stFileUploader"] * {
    color: #292925 !important;
}


/* ==========================================================
   EXPANDER
========================================================== */

[data-testid="stExpander"] {
    background-color: #ffffff !important;

    border: 1px solid #deded7 !important;

    border-radius: 12px !important;
}

[data-testid="stExpander"] * {
    color: #292925 !important;
}


/* ==========================================================
   RADIO BUTTON
========================================================== */

div[data-testid="stRadio"] label {
    color: #292925 !important;
}


/* ==========================================================
   SELECT BOX
========================================================== */

div[data-baseweb="select"] {
    background-color: #ffffff !important;

    border-radius: 9px !important;
}

div[data-baseweb="select"] * {
    color: #292925 !important;
}


/* ==========================================================
   WELCOME SCREEN
========================================================== */

.welcome {
    text-align: center;

    padding: 70px 20px 45px 20px;
}

.welcome-logo {
    width: 60px;
    height: 60px;

    margin: 0 auto 20px auto;

    border-radius: 16px;

    background-color: #292925;
    color: #ffffff !important;

    display: flex;
    align-items: center;
    justify-content: center;

    font-size: 28px;
}

.welcome-title {
    color: #292925 !important;

    font-size: 30px;

    font-weight: 650;

    margin-bottom: 10px;
}

.welcome-text {
    color: #777770 !important;

    font-size: 16px;
}


/* ==========================================================
   USER MESSAGE
========================================================== */

.user-message-row {
    display: flex;

    justify-content: flex-end;

    margin: 22px 0;
}

.user-message {
    max-width: 72%;

    background-color: #e7e7e1;

    color: #292925 !important;

    padding: 12px 17px;

    border-radius: 18px 18px 5px 18px;

    line-height: 1.6;

    font-size: 15px;
}


/* ==========================================================
   AI MESSAGE
========================================================== */

.ai-message-row {
    display: flex;

    justify-content: flex-start;

    margin: 26px 0;
}

.ai-message {
    max-width: 82%;

    color: #30302c !important;

    line-height: 1.7;

    font-size: 15px;
}

.ai-label {
    color: #777770 !important;

    font-size: 12px;

    font-weight: 600;

    margin-bottom: 7px;
}


/* ==========================================================
   SOURCE BADGE
========================================================== */

.source-badge {
    display: inline-block;

    background-color: #e9e9e3;

    color: #686861 !important;

    border-radius: 20px;

    padding: 5px 11px;

    font-size: 11px;

    margin-bottom: 15px;
}


/* ==========================================================
   USAGE CARD
========================================================== */

.usage-card {
    background-color: #e4e4df;

    border-radius: 11px;

    padding: 12px;

    margin-top: 8px;

    font-size: 12px;

    line-height: 1.8;

    color: #66665f !important;
}


/* ==========================================================
   SETTINGS
========================================================== */

.settings-card {
    background-color: #ffffff;

    border: 1px solid #deded7;

    border-radius: 14px;

    padding: 18px;

    margin-bottom: 20px;
}

.settings-title {
    color: #292925 !important;

    font-size: 16px;

    font-weight: 600;

    margin-bottom: 15px;
}


/* ==========================================================
   CHAT INPUT
========================================================== */

[data-testid="stChatInput"] {
    background-color: transparent !important;
}

[data-testid="stChatInput"] > div {
    background-color: #ffffff !important;

    border: 1px solid #cfcfc7 !important;

    border-radius: 17px !important;

    box-shadow: 0 5px 25px rgba(0, 0, 0, 0.07) !important;
}

[data-testid="stChatInput"] textarea {
    background-color: #ffffff !important;

    color: #292925 !important;

    border: none !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: #999990 !important;
}


/* ==========================================================
   TOKEN INFORMATION
========================================================== */

.composer-info {
    position: fixed;

    bottom: 87px;

    left: 50%;

    transform: translateX(-50%);

    width: min(850px, 82%);

    text-align: right;

    font-size: 11px;

    color: #888880 !important;

    z-index: 998;

    pointer-events: none;
}


/* ==========================================================
   DIVIDER
========================================================== */

hr {
    border-color: #deded7 !important;
}


/* ==========================================================
   MOBILE
========================================================== */

@media (max-width: 768px) {

    .block-container {
        padding-left: 15px !important;
        padding-right: 15px !important;
    }

    .user-message {
        max-width: 88%;
    }

    .ai-message {
        max-width: 94%;
    }

    .composer-info {
        width: 90%;
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

session_defaults = {

    "logged_in": False,

    "username": None,

    "current_chat_id": None,

    "messages": [],

    "retriever": None,

    "document_text": "",

    "document_name": "",

    "source_type": "",

    "show_settings": False,

    "response_length": DEFAULT_MAX_TOKENS,

    "theme": "Light",

    "chat_loaded": False
}


for key, value in session_defaults.items():

    if key not in st.session_state:

        st.session_state[key] = value


# ============================================================
# TOKEN COUNT
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

    model_name = (
        "HuggingFaceTB/SmolLM2-360M-Instruct"
    )

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

    st.session_state.chat_loaded = False

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

    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 2
        }
    )

    return retriever


# ============================================================
# SAVE CURRENT CHAT
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

    st.session_state.chat_loaded = True

    if st.session_state.document_text:

        st.session_state.retriever = build_retriever(
            st.session_state.document_text
        )

    else:

        st.session_state.retriever = None

    st.rerun()


# ============================================================
# DELETE CURRENT CHAT
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
# LOGIN FUNCTION
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
# SIGNUP FUNCTION
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
# PDF PROCESSING
# ============================================================

def process_pdf(uploaded_file):

    temporary_path = "temporary_document.pdf"

    with open(
        temporary_path,
        "wb"
    ) as file:

        file.write(
            uploaded_file.getbuffer()
        )

    loader = PyPDFLoader(
        temporary_path
    )

    pages = loader.load()

    text = "\n\n".join(
        page.page_content
        for page in pages
    )

    return text


# ============================================================
# TXT PROCESSING
# ============================================================

def process_txt(uploaded_file):

    return uploaded_file.getvalue().decode(
        "utf-8",
        errors="ignore"
    )


# ============================================================
# DOCX PROCESSING
# ============================================================

def process_docx(uploaded_file):

    document = DocxDocument(
        uploaded_file
    )

    paragraphs = []

    for paragraph in document.paragraphs:

        text = paragraph.text.strip()

        if text:

            paragraphs.append(
                text
            )

    return "\n".join(
        paragraphs
    )


# ============================================================
# XLSX PROCESSING
# ============================================================

def process_xlsx(uploaded_file):

    excel = pd.ExcelFile(
        uploaded_file
    )

    all_text = []

    for sheet in excel.sheet_names:

        dataframe = pd.read_excel(
            uploaded_file,
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

    return "\n\n".join(
        all_text
    )


# ============================================================
# WEBSITE PROCESSING
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

            lines.append(
                line
            )

    return "\n".join(
        lines
    )


# ============================================================
# LOGIN SCREEN
# ============================================================

if not st.session_state.logged_in:

    st.markdown(
        """
<div class="login-wrapper">

    <div class="login-logo">
        ✦
    </div>

    <div class="login-title">
        AI Document Intelligence
    </div>

    <div class="login-subtitle">
        Ask questions about your documents and websites.
    </div>

</div>
        """,
        unsafe_allow_html=True
    )

    login_tab, signup_tab = st.tabs(
        [
            "Login",
            "Create Account"
        ]
    )

    # --------------------------------------------------------
    # LOGIN
    # --------------------------------------------------------

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

        st.write("")

        if st.button(
            "Login",
            use_container_width=True
        ):

            login_user(
                username,
                password
            )

    # --------------------------------------------------------
    # SIGNUP
    # --------------------------------------------------------

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

        st.write("")

        if st.button(
            "Create Account",
            use_container_width=True
        ):

            signup_user(
                new_username,
                new_password
            )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
<div style="
display:flex;
align-items:center;
gap:10px;
margin-bottom:18px;
">

    <div style="
    width:34px;
    height:34px;
    border-radius:10px;
    background:#292925;
    color:#ffffff !important;
    display:flex;
    align-items:center;
    justify-content:center;
    font-size:18px;
    ">
        ✦
    </div>

    <div style="
    font-size:17px;
    font-weight:600;
    color:#292925 !important;
    ">
        AI Document Intelligence
    </div>

</div>
        """,
        unsafe_allow_html=True
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
        key=lambda item:
        item[1].get(
            "updated_at",
            ""
        ),
        reverse=True
    )

    if not user_chats:

        st.caption(
            "Your conversations will appear here."
        )

    else:

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

    # --------------------------------------------------------
    # USAGE
    # --------------------------------------------------------

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

    Requests:
    <b>{user_usage.get("requests", 0)}</b>

    <br>

    Input tokens:
    <b>{user_usage.get("input_tokens", 0)}</b>

    <br>

    Output tokens:
    <b>{user_usage.get("output_tokens", 0)}</b>

    <br>

    Total tokens:
    <b>{user_usage.get("total_tokens", 0)}</b>

</div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    # --------------------------------------------------------
    # SETTINGS
    # --------------------------------------------------------

    if st.button(
        "⚙  Settings",
        use_container_width=True
    ):

        st.session_state.show_settings = (
            not st.session_state.show_settings
        )

        st.rerun()

    # --------------------------------------------------------
    # LOGOUT
    # --------------------------------------------------------

    if st.button(
        "Logout",
        use_container_width=True
    ):

        st.session_state.clear()

        st.rerun()


# ============================================================
# MAIN HEADER
# ============================================================

username_display = html.escape(
    st.session_state.username
)

st.markdown(
    f"""
<div class="app-header">

    <div class="app-header-left">

        <div class="app-logo">
            ✦
        </div>

        <div>

            <div class="app-title">
                AI Document Intelligence
            </div>

            <div class="app-status">
                ● Ready
            </div>

        </div>

    </div>

    <div class="app-status">
        {username_display}
    </div>

</div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SETTINGS PANEL
# ============================================================

if st.session_state.show_settings:

    st.markdown(
        """
<div class="settings-card">

    <div class="settings-title">
        ⚙ Chat Settings
    </div>

</div>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        st.selectbox(
            "Theme",
            ["Light"],
            key="theme_select"
        )

    with col2:

        st.session_state.response_length = st.selectbox(
            "Maximum response tokens",
            [50, 100, 150, 200],
            index=1
        )

    st.caption(
        "100 tokens is recommended for faster responses."
    )


# ============================================================
# DOCUMENT / WEBSITE INPUT
# ============================================================

with st.expander(
    "📎  Add a document or website",
    expanded=not bool(
        st.session_state.document_text
    )
):

    source = st.radio(
        "Choose source",
        [
            "Document",
            "Website URL"
        ],
        horizontal=True
    )

    # --------------------------------------------------------
    # DOCUMENT
    # --------------------------------------------------------

    if source == "Document":

        uploaded_file = st.file_uploader(
            "Upload your document",
            type=[
                "pdf",
                "txt",
                "docx",
                "xlsx"
            ]
        )

        if uploaded_file is not None:

            if (
                uploaded_file.name
                != st.session_state.document_name
            ):

                with st.spinner(
                    "Reading your document..."
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
                                "Unsupported file type."
                            )

                            st.stop()

                        if not text.strip():

                            st.error(
                                "The document contains no readable text."
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
                            f"Error reading document: {error}"
                        )

    # --------------------------------------------------------
    # WEBSITE
    # --------------------------------------------------------

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
                                "No readable text was found."
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
# SOURCE STATUS
# ============================================================

if st.session_state.document_text:

    source_name = html.escape(
        st.session_state.document_name
    )

    st.markdown(
        f"""
<span class="source-badge">
    📄 {source_name}
</span>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# WELCOME SCREEN
# ============================================================

if not st.session_state.messages:

    greeting = get_greeting()

    username = html.escape(
        st.session_state.username
    )

    st.markdown(
        f"""
<div class="welcome">

    <div class="welcome-logo">
        ✦
    </div>

    <div class="welcome-title">
        {greeting}, {username}
    </div>

    <div class="welcome-text">
        What would you like to know?
    </div>

</div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# DISPLAY CHAT MESSAGES
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

    safe_content = (
        html.escape(content)
        .replace(
            "\n",
            "<br>"
        )
    )

    if role == "user":

        st.markdown(
            f"""
<div class="user-message-row">

    <div class="user-message">
        {safe_content}
    </div>

</div>
            """,
            unsafe_allow_html=True
        )

    elif role == "assistant":

        st.markdown(
            f"""
<div class="ai-message-row">

    <div class="ai-message">

        <div class="ai-label">
            ✦ AI
        </div>

        <div>
            {safe_content}
        </div>

    </div>

</div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# TOKEN INFORMATION NEAR INPUT
# ============================================================

st.markdown(
    f"""
<div class="composer-info">
    Maximum response: {st.session_state.response_length} tokens
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
# PROCESS USER QUESTION
# ============================================================

if prompt:

    prompt = prompt.strip()

    if not prompt:

        st.stop()

    with st.spinner(
        "Thinking..."
    ):

        # ----------------------------------------------------
        # LOAD MODEL
        # ----------------------------------------------------

        try:

            tokenizer, generator = load_llm()

        except Exception as error:

            st.error(
                f"Could not load AI model: {error}"
            )

            st.stop()

        # ----------------------------------------------------
        # INPUT TOKENS
        # ----------------------------------------------------

        input_tokens = count_tokens(
            tokenizer,
            prompt
        )

        # ----------------------------------------------------
        # DOCUMENT CONTEXT
        # ----------------------------------------------------

        context = ""

        if st.session_state.retriever:

            try:

                documents = (
                    st.session_state.retriever
                    .invoke(prompt)
                )

                context = "\n\n".join(
                    document.page_content
                    for document in documents
                )

            except Exception:

                context = ""

        # ----------------------------------------------------
        # CREATE PROMPT
        # ----------------------------------------------------

        if context:

            final_prompt = f"""
You are an AI document assistant.

Answer the user's question using ONLY the
information in the context.

Do not invent information.

If the answer is not available in the context,
say:

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

Answer the user's question clearly and briefly.

Question:
{prompt}

Answer:
"""

        # ----------------------------------------------------
        # GENERATE
        # ----------------------------------------------------

        try:

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

        except Exception as error:

            answer = (
                f"Sorry, I encountered an error: {error}"
            )

        # ----------------------------------------------------
        # OUTPUT TOKENS
        # ----------------------------------------------------

        output_tokens = count_tokens(
            tokenizer,
            answer
        )

        total_tokens = (
            input_tokens
            + output_tokens
        )

    # ========================================================
    # SAVE USER MESSAGE
    # ========================================================

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
            "tokens": input_tokens
        }
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
    # UPDATE USAGE
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
    # SAVE CHAT
    # ========================================================

    save_current_chat()

    # ========================================================
    # REFRESH
    # ========================================================

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

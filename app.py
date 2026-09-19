import os
import json
import hashlib
import requests
from datetime import datetime

import streamlit as st
import pandas as pd

from bs4 import BeautifulSoup
from docx import Document as DocxDocument

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import PyPDFLoader
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
# DARK CLAUDE-STYLE CSS
# ============================================================

st.markdown(
    """
<style>

/* =========================================================
   GLOBAL
   ========================================================= */

html, body, [class*="css"] {
    font-family:
        Inter,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;
}

.stApp {
    background: #1f1e1b;
    color: #f5f5f0;
}

/* Main area */

.main .block-container {
    max-width: 1100px;
    padding-top: 30px;
    padding-bottom: 120px;
}

/* Remove Streamlit decoration */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header[data-testid="stHeader"] {
    background: transparent;
}


/* =========================================================
   SIDEBAR
   ========================================================= */

section[data-testid="stSidebar"] {
    background: #171614 !important;
    border-right: 1px solid #34312c;
}

section[data-testid="stSidebar"] > div {
    background: #171614 !important;
}

section[data-testid="stSidebar"] * {
    color: #f1eee7;
}

.sidebar-title {
    font-size: 18px;
    font-weight: 700;
    padding: 8px 0 20px 0;
    color: #f4f1ea;
}

.sidebar-brand {
    color: #e7a15a;
    font-size: 18px;
    font-weight: 700;
}


/* =========================================================
   HEADER
   ========================================================= */

.app-header {
    text-align: center;
    padding: 18px 0 10px 0;
}

.app-title {
    font-size: 28px;
    font-weight: 700;
    color: #f5f3ed;
    letter-spacing: -0.5px;
}

.app-subtitle {
    margin-top: 7px;
    color: #aaa69e;
    font-size: 14px;
}


/* =========================================================
   WELCOME
   ========================================================= */

.welcome {
    text-align: center;
    padding: 75px 20px 45px 20px;
}

.welcome-symbol {
    font-size: 34px;
    color: #e7a15a;
    margin-bottom: 18px;
}

.welcome-title {
    font-size: 29px;
    font-weight: 600;
    color: #f5f2eb;
}

.welcome-text {
    margin-top: 12px;
    color: #aaa69e;
    font-size: 16px;
}


/* =========================================================
   SOURCE AREA
   ========================================================= */

div[data-testid="stExpander"] {
    background: #22211e !important;
    border: 1px solid #45413b !important;
    border-radius: 12px !important;
}

div[data-testid="stExpander"] summary {
    color: #eeeae2 !important;
}

.source-badge {
    display: inline-block;
    margin: 10px 0 15px 0;
    padding: 7px 12px;
    border-radius: 8px;
    background: #292721;
    border: 1px solid #46423a;
    color: #d8d2c8;
    font-size: 13px;
}


/* =========================================================
   CHAT MESSAGES
   ========================================================= */

div[data-testid="stChatMessage"] {
    background: transparent !important;
}

div[data-testid="stChatMessage"] p {
    color: #eeeae3;
    line-height: 1.65;
}

div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
    color: #eeeae3;
}


/* User message */

div[data-testid="stChatMessage"]:has(
    div[data-testid="chatAvatarIcon-user"]
) {
    background: #292722 !important;
    border-radius: 14px;
    padding: 5px;
}


/* =========================================================
   CHAT INPUT
   ========================================================= */

div[data-testid="stChatInput"] {
    background: #252420 !important;
    border: 1px solid #514c44 !important;
    border-radius: 18px !important;
}

div[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: #f5f2eb !important;
    font-size: 15px !important;
}

div[data-testid="stChatInput"] textarea::placeholder {
    color: #8f8b84 !important;
}

div[data-testid="stChatInput"] button {
    background: #e7a15a !important;
    color: #1f1e1b !important;
    border-radius: 10px !important;
}

div[data-testid="stChatInput"] button:hover {
    background: #f0b477 !important;
}


/* =========================================================
   TOKEN LIMIT NEAR CHAT BOX
   ========================================================= */

.token-limit-bar {
    position: fixed;
    right: 35px;
    bottom: 82px;
    z-index: 999;
    padding: 5px 11px;
    border-radius: 8px;
    background: #2b2925;
    border: 1px solid #45413a;
    color: #a9a49c;
    font-size: 12px;
}


/* =========================================================
   BUTTONS
   ========================================================= */

.stButton > button {
    background: #292722 !important;
    color: #eeeae5 !important;
    border: 1px solid #49453e !important;
    border-radius: 9px !important;
    min-height: 40px;
}

.stButton > button:hover {
    background: #35322c !important;
    border-color: #e7a15a !important;
    color: #ffffff !important;
}


/* Orange primary button */

.primary-button button {
    background: #e7a15a !important;
    color: #211d18 !important;
    border: none !important;
    font-weight: 700 !important;
}

.primary-button button:hover {
    background: #f0b477 !important;
}


/* =========================================================
   INPUTS
   ========================================================= */

.stTextInput input,
.stTextArea textarea {
    background: #252420 !important;
    color: #f3f0e9 !important;
    border: 1px solid #4b4740 !important;
    border-radius: 9px !important;
}

.stTextInput input:focus,
.stTextArea textarea:focus {
    border-color: #e7a15a !important;
    box-shadow: 0 0 0 1px #e7a15a !important;
}


/* =========================================================
   SELECTBOX / RADIO / FILE UPLOADER
   ========================================================= */

div[data-baseweb="select"] > div {
    background: #252420 !important;
    border-color: #4b4740 !important;
    color: #eeeae4 !important;
}

div[data-testid="stFileUploader"] {
    background: #252420 !important;
    border: 1px dashed #514d45 !important;
    border-radius: 10px !important;
}

div[data-testid="stFileUploader"] * {
    color: #ddd8d0 !important;
}


/* =========================================================
   USAGE CARD
   ========================================================= */

.usage-card {
    margin-top: 8px;
    padding: 14px;
    background: #24221f;
    border: 1px solid #3f3c36;
    border-radius: 11px;
    line-height: 1.9;
    font-size: 13px;
    color: #aaa69f;
}

.usage-card b {
    color: #eeeae3;
}


/* =========================================================
   CHAT HISTORY
   ========================================================= */

.chat-history-title {
    color: #8f8b84;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 25px;
    margin-bottom: 8px;
}


/* =========================================================
   ALERTS
   ========================================================= */

div[data-testid="stAlert"] {
    border-radius: 10px !important;
}


/* =========================================================
   LOGIN
   ========================================================= */

.login-container {
    max-width: 620px;
    margin: 80px auto 0 auto;
}

.login-logo {
    text-align: center;
    font-size: 36px;
    color: #e7a15a;
    margin-bottom: 12px;
}

.login-title {
    text-align: center;
    font-size: 30px;
    font-weight: 700;
    color: #f5f2eb;
}

.login-subtitle {
    text-align: center;
    color: #aaa69e;
    margin-top: 10px;
    margin-bottom: 45px;
}


/* =========================================================
   SETTINGS
   ========================================================= */

.settings-box {
    padding: 15px;
    background: #25231f;
    border: 1px solid #3e3a34;
    border-radius: 10px;
}


/* =========================================================
   MOBILE
   ========================================================= */

@media (max-width: 768px) {

    .main .block-container {
        padding-left: 15px;
        padding-right: 15px;
    }

    .app-title {
        font-size: 23px;
    }

    .welcome {
        padding-top: 45px;
    }

    .welcome-title {
        font-size: 24px;
    }

    .token-limit-bar {
        right: 15px;
        bottom: 82px;
    }
}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# FILE PATHS
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
# PASSWORD HELPERS
# ============================================================

def hash_password(password):
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


# ============================================================
# USER FUNCTIONS
# ============================================================

def create_user(username, password):

    users = load_json(USERS_FILE, {})

    username = username.strip()

    if not username or not password:
        return False, "Username and password are required."

    if username in users:
        return False, "Username already exists."

    users[username] = {
        "password": hash_password(password),
        "created_at": datetime.now().isoformat()
    }

    save_json(USERS_FILE, users)

    return True, "Account created successfully."


def verify_user(username, password):

    users = load_json(USERS_FILE, {})

    if username not in users:
        return False

    stored_password = users[username].get("password", "")

    # New hashed password
    if stored_password == hash_password(password):
        return True

    # Compatibility with older plaintext users.json
    if stored_password == password:
        return True

    return False


# ============================================================
# USAGE FUNCTIONS
# ============================================================

def get_user_usage(username):

    usage = load_json(USAGE_FILE, {})

    return usage.get(
        username,
        {
            "requests": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0
        }
    )


def update_user_usage(
    username,
    input_tokens,
    output_tokens
):

    usage = load_json(USAGE_FILE, {})

    if username not in usage:
        usage[username] = {
            "requests": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0
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

def load_all_chats():

    return load_json(CHATS_FILE, {})


def save_all_chats(chats):

    save_json(CHATS_FILE, chats)


def get_user_chats(username):

    chats = load_all_chats()

    if username not in chats:
        chats[username] = {}

    return chats[username]


def create_chat(username):

    chats = load_all_chats()

    if username not in chats:
        chats[username] = {}

    chat_id = datetime.now().strftime(
        "%Y%m%d%H%M%S%f"
    )

    chats[username][chat_id] = {
        "title": "New Chat",
        "messages": [],
        "document_name": "",
        "document_text": "",
        "source_type": "",
        "total_tokens": 0,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat()
    }

    save_all_chats(chats)

    return chat_id


def save_chat(
    username,
    chat_id,
    messages,
    document_name,
    document_text,
    source_type,
    total_tokens
):

    chats = load_all_chats()

    if username not in chats:
        chats[username] = {}

    if chat_id not in chats[username]:
        chats[username][chat_id] = {}

    title = "New Chat"

    for message in messages:

        if message["role"] == "user":

            title = message["content"][:45]

            if len(message["content"]) > 45:
                title += "..."

            break

    chats[username][chat_id].update(
        {
            "title": title,
            "messages": messages,
            "document_name": document_name,
            "document_text": document_text,
            "source_type": source_type,
            "total_tokens": total_tokens,
            "updated_at": datetime.now().isoformat()
        }
    )

    save_all_chats(chats)


def delete_chat(username, chat_id):

    chats = load_all_chats()

    if (
        username in chats
        and chat_id in chats[username]
    ):

        del chats[username][chat_id]

        save_all_chats(chats)


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
    "chat_total_tokens": 0,
    "response_length": 100,
    "show_settings": False
}

for key, value in defaults.items():

    if key not in st.session_state:
        st.session_state[key] = value


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
# TOKEN COUNT
# ============================================================

def count_tokens(tokenizer, text):

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

@st.cache_resource(show_spinner=False)
def load_embeddings():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# ============================================================
# LOCAL LLM
# ============================================================

@st.cache_resource(show_spinner=False)
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
        max_new_tokens=100,
        do_sample=False,
        return_full_text=False,
        pad_token_id=tokenizer.eos_token_id
    )

    return generator, tokenizer


# ============================================================
# DOCUMENT PROCESSING
# ============================================================

def process_uploaded_file(uploaded_file):

    filename = uploaded_file.name.lower()

    documents = []

    # --------------------------------------------------------
    # PDF
    # --------------------------------------------------------

    if filename.endswith(".pdf"):

        temp_path = "temp_uploaded.pdf"

        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        loader = PyPDFLoader(temp_path)

        documents = loader.load()

        try:
            os.remove(temp_path)
        except Exception:
            pass


    # --------------------------------------------------------
    # TXT
    # --------------------------------------------------------

    elif filename.endswith(".txt"):

        text = uploaded_file.getvalue().decode(
            "utf-8",
            errors="ignore"
        )

        documents = [
            Document(page_content=text)
        ]


    # --------------------------------------------------------
    # DOCX
    # --------------------------------------------------------

    elif filename.endswith(".docx"):

        doc = DocxDocument(uploaded_file)

        paragraphs = [
            p.text
            for p in doc.paragraphs
            if p.text.strip()
        ]

        text = "\n".join(paragraphs)

        documents = [
            Document(page_content=text)
        ]


    # --------------------------------------------------------
    # XLSX
    # --------------------------------------------------------

    elif filename.endswith(".xlsx"):

        excel = pd.ExcelFile(uploaded_file)

        for sheet in excel.sheet_names:

            df = pd.read_excel(
                uploaded_file,
                sheet_name=sheet
            )

            text = df.to_string(
                index=False
            )

            documents.append(
                Document(
                    page_content=(
                        f"Sheet: {sheet}\n\n{text}"
                    )
                )
            )

    else:

        raise ValueError(
            "Unsupported file type."
        )

    return documents


# ============================================================
# WEBSITE PROCESSING
# ============================================================

def process_website(url):

    response = requests.get(
        url,
        headers={
            "User-Agent": "Mozilla/5.0"
        },
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
        separator="\n"
    )

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    clean_text = "\n".join(lines)

    if not clean_text.strip():

        raise ValueError(
            "Could not extract readable text from this website."
        )

    return [
        Document(page_content=clean_text)
    ]


# ============================================================
# BUILD VECTOR STORE
# ============================================================

def build_retriever(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(
        documents
    )

    if not chunks:

        raise ValueError(
            "No readable content was found."
        )

    embeddings = load_embeddings()

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    # Only retrieve 2 chunks for faster response
    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 2
        }
    )

    return retriever


# ============================================================
# REBUILD RETRIEVER FROM SAVED CHAT
# ============================================================

def rebuild_retriever_from_text(text):

    if not text.strip():
        return None

    documents = [
        Document(
            page_content=text
        )
    ]

    return build_retriever(
        documents
    )


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(
    question,
    retriever,
    generator,
    tokenizer
):

    docs = retriever.invoke(
        question
    )

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    prompt = f"""
You are an AI document assistant.

Answer the user's question using ONLY the provided context.

If the answer is not available in the context, say:
"I could not find that information in the uploaded document."

Keep the answer clear, direct and concise.

Context:
{context}

Question:
{question}

Answer:
"""

    input_tokens = count_tokens(
        tokenizer,
        prompt
    )

    result = generator(
        prompt,
        max_new_tokens=100,
        do_sample=False,
        return_full_text=False
    )

    answer = result[0]["generated_text"].strip()

    output_tokens = count_tokens(
        tokenizer,
        answer
    )

    return (
        answer,
        input_tokens,
        output_tokens
    )


# ============================================================
# LOGIN PAGE
# ============================================================

def show_login():

    st.html(
        """
        <div class="login-container">

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
        """
    )

    tab_login, tab_create = st.tabs(
        [
            "Login",
            "Create Account"
        ]
    )

    # ========================================================
    # LOGIN
    # ========================================================

    with tab_login:

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

            if verify_user(
                username,
                password
            ):

                st.session_state.logged_in = True
                st.session_state.username = username

                chat_id = create_chat(
                    username
                )

                st.session_state.current_chat_id = chat_id
                st.session_state.messages = []
                st.session_state.retriever = None
                st.session_state.document_text = ""
                st.session_state.document_name = ""
                st.session_state.source_type = ""
                st.session_state.chat_total_tokens = 0

                st.rerun()

            else:

                st.error(
                    "Invalid username or password."
                )


    # ========================================================
    # CREATE ACCOUNT
    # ========================================================

    with tab_create:

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

            success, message = create_user(
                new_username,
                new_password
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
# SHOW LOGIN IF NOT LOGGED IN
# ============================================================

if not st.session_state.logged_in:

    show_login()

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-title">
            ✦ <span class="sidebar-brand">
            AI Document Intelligence
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # NEW CHAT
    # --------------------------------------------------------

    if st.button(
        "＋ New Chat",
        use_container_width=True
    ):

        chat_id = create_chat(
            st.session_state.username
        )

        st.session_state.current_chat_id = chat_id

        st.session_state.messages = []

        st.session_state.retriever = None

        st.session_state.document_text = ""

        st.session_state.document_name = ""

        st.session_state.source_type = ""

        st.session_state.chat_total_tokens = 0

        # Clear uploader / URL state
        for key in [
            "uploaded_file",
            "website_url"
        ]:

            if key in st.session_state:
                del st.session_state[key]

        st.rerun()


    st.markdown(
        '<div class="chat-history-title">Chats</div>',
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # CHAT HISTORY
    # --------------------------------------------------------

    user_chats = get_user_chats(
        st.session_state.username
    )

    if user_chats:

        sorted_chats = sorted(
            user_chats.items(),
            key=lambda item: item[1].get(
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

            if len(title) > 30:
                title = title[:30] + "..."

            if st.button(
                title,
                key=f"chat_{chat_id}",
                use_container_width=True
            ):

                st.session_state.current_chat_id = chat_id

                st.session_state.messages = chat.get(
                    "messages",
                    []
                )

                st.session_state.document_name = chat.get(
                    "document_name",
                    ""
                )

                st.session_state.document_text = chat.get(
                    "document_text",
                    ""
                )

                st.session_state.source_type = chat.get(
                    "source_type",
                    ""
                )

                st.session_state.chat_total_tokens = chat.get(
                    "total_tokens",
                    0
                )

                if st.session_state.document_text:

                    try:

                        st.session_state.retriever = (
                            rebuild_retriever_from_text(
                                st.session_state.document_text
                            )
                        )

                    except Exception:

                        st.session_state.retriever = None

                else:

                    st.session_state.retriever = None

                st.rerun()

    else:

        st.caption(
            "No conversations yet."
        )


    st.divider()


    # --------------------------------------------------------
    # USAGE
    # --------------------------------------------------------

    st.markdown(
        "### Usage"
    )

    usage = get_user_usage(
        st.session_state.username
    )

    st.markdown(
        f"""
        <div class="usage-card">

            Requests:
            <b>{usage.get("requests", 0)}</b>
            <br>

            Input:
            <b>{usage.get("input_tokens", 0)}</b>
            <br>

            Output:
            <b>{usage.get("output_tokens", 0)}</b>
            <br>

            Total:
            <b>{usage.get("total_tokens", 0)}</b>

        </div>
        """,
        unsafe_allow_html=True
    )


    st.write("")


    # --------------------------------------------------------
    # SETTINGS
    # --------------------------------------------------------

    if st.button(
        "⚙ Settings",
        use_container_width=True
    ):

        st.session_state.show_settings = (
            not st.session_state.show_settings
        )

        st.rerun()


    # --------------------------------------------------------
    # DELETE CURRENT CHAT
    # --------------------------------------------------------

    if st.session_state.current_chat_id:

        if st.button(
            "🗑 Delete Chat",
            use_container_width=True
        ):

            delete_chat(
                st.session_state.username,
                st.session_state.current_chat_id
            )

            new_chat_id = create_chat(
                st.session_state.username
            )

            st.session_state.current_chat_id = (
                new_chat_id
            )

            st.session_state.messages = []

            st.session_state.retriever = None

            st.session_state.document_text = ""

            st.session_state.document_name = ""

            st.session_state.source_type = ""

            st.session_state.chat_total_tokens = 0

            st.rerun()


    # --------------------------------------------------------
    # LOGOUT
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

st.html(
    """
    <div class="app-header">

        <div class="app-title">
            ✦ AI Document Intelligence
        </div>

        <div class="app-subtitle">
            Document & Website Assistant
        </div>

    </div>
    """
)


# ============================================================
# SETTINGS PANEL
# ============================================================

if st.session_state.show_settings:

    with st.expander(
        "⚙ Settings",
        expanded=True
    ):

        st.markdown(
            "### Response Settings"
        )

        response_length = st.slider(
            "Maximum response tokens",
            min_value=50,
            max_value=200,
            value=st.session_state.response_length,
            step=10
        )

        st.session_state.response_length = (
            response_length
        )

        st.caption(
            "Lower values give shorter and usually faster responses."
        )


# ============================================================
# DOCUMENT / WEBSITE UPLOAD AREA
# ============================================================

with st.expander(
    "📎 Add a document or website",
    expanded=(
        st.session_state.retriever is None
    )
):

    source_type = st.radio(
        "Choose source",
        [
            "Document",
            "Website URL"
        ],
        horizontal=True
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
            key="uploaded_file"
        )

        if uploaded_file is not None:

            if st.button(
                "Process Document",
                use_container_width=True
            ):

                with st.spinner(
                    "Reading and indexing document..."
                ):

                    try:

                        documents = process_uploaded_file(
                            uploaded_file
                        )

                        full_text = "\n\n".join(
                            doc.page_content
                            for doc in documents
                        )

                        retriever = build_retriever(
                            documents
                        )

                        st.session_state.retriever = (
                            retriever
                        )

                        st.session_state.document_text = (
                            full_text
                        )

                        st.session_state.document_name = (
                            uploaded_file.name
                        )

                        st.session_state.source_type = (
                            "Document"
                        )

                        st.success(
                            "Document processed successfully."
                        )

                        save_chat(
                            st.session_state.username,
                            st.session_state.current_chat_id,
                            st.session_state.messages,
                            st.session_state.document_name,
                            st.session_state.document_text,
                            st.session_state.source_type,
                            st.session_state.chat_total_tokens
                        )

                        st.rerun()

                    except Exception as e:

                        st.error(
                            f"Error processing document: {e}"
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
            "Process Website",
            use_container_width=True
        ):

            if not website_url.strip():

                st.warning(
                    "Please enter a website URL."
                )

            else:

                with st.spinner(
                    "Reading and indexing website..."
                ):

                    try:

                        documents = process_website(
                            website_url.strip()
                        )

                        full_text = "\n\n".join(
                            doc.page_content
                            for doc in documents
                        )

                        retriever = build_retriever(
                            documents
                        )

                        st.session_state.retriever = (
                            retriever
                        )

                        st.session_state.document_text = (
                            full_text
                        )

                        st.session_state.document_name = (
                            website_url.strip()
                        )

                        st.session_state.source_type = (
                            "Website"
                        )

                        st.success(
                            "Website processed successfully."
                        )

                        save_chat(
                            st.session_state.username,
                            st.session_state.current_chat_id,
                            st.session_state.messages,
                            st.session_state.document_name,
                            st.session_state.document_text,
                            st.session_state.source_type,
                            st.session_state.chat_total_tokens
                        )

                        st.rerun()

                    except Exception as e:

                        st.error(
                            f"Error processing website: {e}"
                        )


# ============================================================
# SOURCE BADGE
# ============================================================

if st.session_state.document_name:

    icon = (
        "🌐"
        if st.session_state.source_type == "Website"
        else "📄"
    )

    st.html(
        f"""
        <div class="source-badge">
            {icon} {st.session_state.document_name}
        </div>
        """
    )


# ============================================================
# WELCOME MESSAGE
# ============================================================

if not st.session_state.messages:

    greeting = get_greeting()

    st.html(
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
        """
    )


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# TOKEN LIMIT NEAR CHAT BOX
# ============================================================

st.html(
    f"""
    <div class="token-limit-bar">
        Max response: {st.session_state.response_length} tokens
    </div>
    """
)


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask anything about your document..."
)


# ============================================================
# HANDLE QUESTION
# ============================================================

if question:

    question = question.strip()

    if not question:
        st.stop()


    # --------------------------------------------------------
    # USER MESSAGE
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):

        st.markdown(
            question
        )


    # --------------------------------------------------------
    # CHECK DOCUMENT
    # --------------------------------------------------------

    if st.session_state.retriever is None:

        answer = (
            "Please upload a document or process "
            "a website first."
        )

        input_tokens = 0
        output_tokens = 0

    else:

        with st.chat_message("assistant"):

            with st.spinner(
                "Thinking..."
            ):

                try:

                    generator, tokenizer = (
                        load_llm()
                    )

                    answer, input_tokens, output_tokens = (
                        generate_answer(
                            question,
                            st.session_state.retriever,
                            generator,
                            tokenizer
                        )
                    )

                except Exception as e:

                    answer = (
                        f"Sorry, I couldn't generate "
                        f"the answer. Error: {e}"
                    )

                    input_tokens = 0
                    output_tokens = 0

            st.markdown(
                answer
            )


    # --------------------------------------------------------
    # SAVE ASSISTANT MESSAGE
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )


    # --------------------------------------------------------
    # UPDATE TOKEN USAGE
    # --------------------------------------------------------

    total_tokens = (
        input_tokens +
        output_tokens
    )

    st.session_state.chat_total_tokens += (
        total_tokens
    )

    update_user_usage(
        st.session_state.username,
        input_tokens,
        output_tokens
    )


    # --------------------------------------------------------
    # SAVE CHAT
    # --------------------------------------------------------

    save_chat(
        st.session_state.username,
        st.session_state.current_chat_id,
        st.session_state.messages,
        st.session_state.document_name,
        st.session_state.document_text,
        st.session_state.source_type,
        st.session_state.chat_total_tokens
    )


    # --------------------------------------------------------
    # RERUN TO UPDATE SIDEBAR USAGE
    # --------------------------------------------------------

    st.rerun()

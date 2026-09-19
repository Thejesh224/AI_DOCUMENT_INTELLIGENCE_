import streamlit as st
import os
import json
import uuid
import re
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
# CUSTOM CSS - CLAUDE STYLE
# ============================================================

st.markdown(
    """
<style>

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

/* Main background */

.stApp {
    background: #f7f7f5;
}

/* Sidebar */

section[data-testid="stSidebar"] {
    background: #f0f0ec;
    border-right: 1px solid #deded8;
}

section[data-testid="stSidebar"] > div {
    padding-top: 1rem;
}

/* Sidebar buttons */

section[data-testid="stSidebar"] button {
    border-radius: 10px;
    border: none;
    background: transparent;
    text-align: left;
}

section[data-testid="stSidebar"] button:hover {
    background: #e5e5df;
}

/* Main content width */

.block-container {
    max-width: 1050px;
    padding-top: 1.5rem;
    padding-bottom: 150px;
}

/* Header */

.chat-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0 20px 0;
    border-bottom: 1px solid #e2e2dc;
    margin-bottom: 20px;
}

.chat-title {
    font-size: 19px;
    font-weight: 600;
    color: #2d2d2a;
}

.online {
    font-size: 13px;
    color: #777770;
}

/* Welcome */

.welcome {
    text-align: center;
    padding: 80px 20px 40px 20px;
}

.welcome-icon {
    font-size: 40px;
    margin-bottom: 15px;
}

.welcome h1 {
    font-size: 31px;
    font-weight: 600;
    color: #292925;
    margin-bottom: 10px;
}

.welcome p {
    color: #777770;
    font-size: 16px;
}

/* User message */

.user-message {
    display: flex;
    justify-content: flex-end;
    margin: 22px 0;
}

.user-bubble {
    max-width: 75%;
    background: #e8e8e3;
    padding: 12px 17px;
    border-radius: 18px;
    color: #292925;
    line-height: 1.55;
}

/* Assistant */

.assistant-message {
    margin: 25px 0;
}

.assistant-name {
    font-size: 13px;
    font-weight: 600;
    color: #6d6d66;
    margin-bottom: 7px;
}

.assistant-content {
    color: #30302c;
    line-height: 1.65;
    font-size: 15px;
}

/* Input area */

.chat-input-wrapper {
    position: fixed;
    bottom: 18px;
    left: 50%;
    transform: translateX(-50%);
    width: min(850px, 85%);
    z-index: 999;
}

/* Token badge */

.token-badge {
    position: absolute;
    right: 55px;
    bottom: 14px;
    font-size: 11px;
    color: #888880;
    background: transparent;
    z-index: 1000;
}

/* Settings card */

.settings-card {
    background: #ffffff;
    border: 1px solid #deded8;
    border-radius: 14px;
    padding: 18px;
    margin-top: 12px;
}

.settings-title {
    font-size: 15px;
    font-weight: 600;
    margin-bottom: 12px;
}

/* Source badge */

.source-badge {
    display: inline-block;
    padding: 4px 9px;
    border-radius: 20px;
    background: #e9e9e3;
    color: #686861;
    font-size: 11px;
    margin-bottom: 15px;
}

/* Usage */

.usage-box {
    background: #e7e7e2;
    border-radius: 10px;
    padding: 10px;
    margin-top: 10px;
    font-size: 12px;
    color: #66665f;
}

/* Hide default file uploader border */

[data-testid="stFileUploader"] {
    border-radius: 10px;
}

/* Buttons */

.stButton button {
    border-radius: 9px;
}

/* Select boxes */

.stSelectbox div[data-baseweb="select"] {
    border-radius: 9px;
}

/* Text input */

.stTextInput input {
    border-radius: 12px;
}

/* Mobile */

@media (max-width: 768px) {

    .block-container {
        padding-left: 15px;
        padding-right: 15px;
    }

    .chat-input-wrapper {
        width: 94%;
    }

    .user-bubble {
        max-width: 90%;
    }

}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# FILES
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
    "show_settings": False,
    "theme": "Light",
    "response_length": 100,
    "input_token_count": 0
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# AUTH
# ============================================================

def login_user(username, password):

    if username in users and users[username]["password"] == password:

        st.session_state.logged_in = True
        st.session_state.username = username

        if username not in usage:
            usage[username] = {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "requests": 0
            }

            save_json(USAGE_FILE, usage)

        st.rerun()

    else:
        st.error("Invalid username or password.")


def signup_user(username, password):

    if not username or not password:
        st.error("Please enter username and password.")
        return

    if username in users:
        st.error("Username already exists.")
        return

    users[username] = {
        "password": password,
        "created_at": datetime.now().isoformat()
    }

    save_json(USERS_FILE, users)

    usage[username] = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "requests": 0
    }

    save_json(USAGE_FILE, usage)

    st.success("Account created. Please login.")


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
# LOAD EMBEDDINGS
# ============================================================

@st.cache_resource
def load_embeddings():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# ============================================================
# LOAD LLM
# ============================================================

@st.cache_resource
def load_llm():

    model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name
    )

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=100,
        do_sample=False,
        return_full_text=False
    )

    return tokenizer, generator


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
    st.session_state.input_token_count = 0

    st.rerun()


# ============================================================
# SAVE CHAT
# ============================================================

def save_current_chat():

    username = st.session_state.username

    if not st.session_state.messages:
        return

    if not st.session_state.current_chat_id:
        chat_id = str(uuid.uuid4())
        st.session_state.current_chat_id = chat_id
    else:
        chat_id = st.session_state.current_chat_id

    first_user_message = "New Chat"

    for message in st.session_state.messages:

        if message["role"] == "user":
            first_user_message = message["content"][:35]
            break

    chats[chat_id] = {
        "username": username,
        "title": first_user_message,
        "messages": st.session_state.messages,
        "document_text": st.session_state.document_text,
        "document_name": st.session_state.document_name,
        "source_type": st.session_state.source_type,
        "total_tokens": sum(
            m.get("tokens", 0)
            for m in st.session_state.messages
        ),
        "updated_at": datetime.now().isoformat()
    }

    save_json(CHATS_FILE, chats)


# ============================================================
# LOAD CHAT
# ============================================================

def load_chat(chat_id):

    chat = chats.get(chat_id)

    if not chat:
        return

    st.session_state.current_chat_id = chat_id
    st.session_state.messages = chat.get("messages", [])
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

        build_retriever(
            st.session_state.document_text
        )

    st.rerun()


# ============================================================
# BUILD RETRIEVER
# ============================================================

def build_retriever(text):

    if not text:
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = [
        Document(page_content=text)
    ]

    chunks = splitter.split_documents(docs)

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
# PDF
# ============================================================

def process_pdf(uploaded_file):

    temp_path = "temp_uploaded.pdf"

    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    loader = PyPDFLoader(temp_path)

    pages = loader.load()

    text = "\n\n".join(
        page.page_content
        for page in pages
    )

    return text


# ============================================================
# TXT
# ============================================================

def process_txt(uploaded_file):

    return uploaded_file.getvalue().decode(
        "utf-8",
        errors="ignore"
    )


# ============================================================
# DOCX
# ============================================================

def process_docx(uploaded_file):

    document = DocxDocument(
        uploaded_file
    )

    text = "\n".join(
        paragraph.text
        for paragraph in document.paragraphs
    )

    return text


# ============================================================
# XLSX
# ============================================================

def process_xlsx(uploaded_file):

    excel = pd.ExcelFile(
        uploaded_file
    )

    all_text = []

    for sheet in excel.sheet_names:

        df = pd.read_excel(
            uploaded_file,
            sheet_name=sheet
        )

        all_text.append(
            f"Sheet: {sheet}\n"
        )

        all_text.append(
            df.to_string(index=False)
        )

    return "\n\n".join(all_text)


# ============================================================
# WEBSITE
# ============================================================

def process_website(url):

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
        separator="\n"
    )

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    return "\n".join(lines)


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
# LOGIN SCREEN
# ============================================================

if not st.session_state.logged_in:

    st.markdown(
        """
        <div style="text-align:center; padding:90px 20px 30px;">
            <div style="font-size:45px;">✦</div>
            <h1>AI Document Intelligence</h1>
            <p style="color:#777;">
                Ask questions about your documents and websites.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    tab1, tab2 = st.tabs(
        ["Login", "Create Account"]
    )

    with tab1:

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
            use_container_width=True
        ):

            login_user(
                username,
                password
            )

    with tab2:

        new_username = st.text_input(
            "Username",
            key="signup_username"
        )

        new_password = st.text_input(
            "Password",
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

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="font-size:20px;font-weight:600;margin-bottom:15px;">
            ✦ AI Document Intelligence
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
        "**Chats**"
    )

    user_chats = [
        (chat_id, chat)
        for chat_id, chat in chats.items()
        if chat.get("username") ==
        st.session_state.username
    ]

    user_chats.sort(
        key=lambda x: x[1].get(
            "updated_at",
            ""
        ),
        reverse=True
    )

    if not user_chats:

        st.caption(
            "No previous chats"
        )

    else:

        for chat_id, chat in user_chats:

            title = chat.get(
                "title",
                "New Chat"
            )

            if len(title) > 30:
                title = title[:30] + "..."

            if st.button(
                "▸ " + title,
                key=f"chat_{chat_id}",
                use_container_width=True
            ):

                load_chat(chat_id)

    st.divider()

    st.markdown(
        f"**👤 {st.session_state.username}**"
    )

    user_usage = usage.get(
        st.session_state.username,
        {}
    )

    st.markdown(
        f"""
        <div class="usage-box">
            <b>Usage</b><br><br>
            Requests: {user_usage.get("requests", 0)}<br>
            Input tokens: {user_usage.get("input_tokens", 0)}<br>
            Output tokens: {user_usage.get("output_tokens", 0)}<br>
            Total tokens: {user_usage.get("total_tokens", 0)}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "⚙ Settings",
            use_container_width=True
        ):

            st.session_state.show_settings = (
                not st.session_state.show_settings
            )

            st.rerun()

    with col2:

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
    f"""
    <div class="chat-header">
        <div>
            <div class="chat-title">
                AI Document Intelligence
            </div>
            <div class="online">
                ● Ready
            </div>
        </div>
        <div class="online">
            {st.session_state.username}
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

        st.session_state.theme = st.selectbox(
            "Theme",
            [
                "Light",
                "Dark"
            ],
            index=0 if
            st.session_state.theme == "Light"
            else 1
        )

    with col2:

        st.session_state.response_length = st.selectbox(
            "Response length",
            [
                50,
                100,
                150,
                200
            ],
            index=1
        )

    st.caption(
        "Token count is based on the HuggingFace model tokenizer."
    )


# ============================================================
# DOCUMENT / WEBSITE SOURCE
# ============================================================

with st.expander(
    "📎 Add document or website",
    expanded=not bool(
        st.session_state.document_text
    )
):

    source_type = st.radio(
        "Source",
        [
            "Document",
            "Website URL"
        ],
        horizontal=True
    )

    if source_type == "Document":

        uploaded_file = st.file_uploader(
            "Upload PDF, TXT, DOCX or XLSX",
            type=[
                "pdf",
                "txt",
                "docx",
                "xlsx"
            ]
        )

        if uploaded_file is not None:

            if (
                uploaded_file.name !=
                st.session_state.document_name
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
                                "Unsupported file."
                            )

                            st.stop()

                        st.session_state.document_text = text
                        st.session_state.document_name = uploaded_file.name
                        st.session_state.source_type = "Document"

                        st.session_state.retriever = build_retriever(
                            text
                        )

                        st.success(
                            f"✓ {uploaded_file.name} is ready."
                        )

                    except Exception as e:

                        st.error(
                            f"Error reading document: {e}"
                        )

    else:

        url = st.text_input(
            "Website URL",
            placeholder="https://example.com"
        )

        if st.button(
            "Load Website"
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

                        if not text:

                            st.error(
                                "No readable text found."
                            )

                        else:

                            st.session_state.document_text = text
                            st.session_state.document_name = url.strip()
                            st.session_state.source_type = "Website"

                            st.session_state.retriever = build_retriever(
                                text
                            )

                            st.success(
                                "✓ Website is ready."
                            )

                    except Exception as e:

                        st.error(
                            f"Could not load website: {e}"
                        )


# ============================================================
# SOURCE STATUS
# ============================================================

if st.session_state.document_text:

    st.markdown(
        f"""
        <span class="source-badge">
            📄 {st.session_state.document_name}
        </span>
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

            <div class="welcome-icon">
                ✦
            </div>

            <h1>
                {greeting}, {st.session_state.username}
            </h1>

            <p>
                What would you like to know?
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# DISPLAY MESSAGES
# ============================================================

for message in st.session_state.messages:

    role = message["role"]
    content = message["content"]

    if role == "user":

        st.markdown(
            f"""
            <div class="user-message">
                <div class="user-bubble">
                    {content}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f"""
            <div class="assistant-message">

                <div class="assistant-name">
                    ✦ AI
                </div>

                <div class="assistant-content">
                    {content}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# CHAT INPUT
# ============================================================

st.markdown(
    '<div class="chat-input-wrapper">',
    unsafe_allow_html=True
)

prompt = st.chat_input(
    "Ask anything about your document..."
)

st.markdown(
    f"""
    <div class="token-badge">
        {st.session_state.input_token_count} tokens
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# PROCESS QUESTION
# ============================================================

if prompt:

    # --------------------------------------------
    # Load model
    # --------------------------------------------

    with st.spinner(
        "Thinking..."
    ):

        try:

            tokenizer, generator = load_llm()

        except Exception as e:

            st.error(
                f"Model loading error: {e}"
            )

            st.stop()

        # ----------------------------------------
        # Input tokens
        # ----------------------------------------

        input_tokens = count_tokens(
            tokenizer,
            prompt
        )

        st.session_state.input_token_count = (
            input_tokens
        )

        # ----------------------------------------
        # Retrieve context
        # ----------------------------------------

        context = ""

        if st.session_state.retriever:

            try:

                docs = (
                    st.session_state.retriever
                    .invoke(prompt)
                )

                context = "\n\n".join(
                    doc.page_content
                    for doc in docs
                )

            except Exception:

                context = ""

        # ----------------------------------------
        # Prompt
        # ----------------------------------------

        if context:

            final_prompt = f"""
You are a helpful document assistant.

Answer the user's question using ONLY the
information in the provided context.

If the answer is not present in the context,
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

Question:
{prompt}

Answer:
"""

        # ----------------------------------------
        # Generate answer
        # ----------------------------------------

        try:

            result = generator(
                final_prompt,
                max_new_tokens=st.session_state.response_length,
                do_sample=False
            )

            answer = result[0]["generated_text"].strip()

            if not answer:

                answer = (
                    "I could not generate an answer."
                )

        except Exception as e:

            answer = (
                f"Sorry, I encountered an error: {e}"
            )

        # ----------------------------------------
        # Output tokens
        # ----------------------------------------

        output_tokens = count_tokens(
            tokenizer,
            answer
        )

        total_tokens = (
            input_tokens +
            output_tokens
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
    # UPDATE USER USAGE
    # ========================================================

    username = st.session_state.username

    if username not in usage:

        usage[username] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "requests": 0
        }

    usage[username]["input_tokens"] += input_tokens

    usage[username]["output_tokens"] += output_tokens

    usage[username]["total_tokens"] += total_tokens

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
    # DISPLAY ANSWER
    # ========================================================

    st.markdown(
        f"""
        <div class="assistant-message">

            <div class="assistant-name">
                ✦ AI
            </div>

            <div class="assistant-content">
                {answer}
            </div>

        </div>

        <div style="
            text-align:right;
            color:#888880;
            font-size:11px;
            margin-bottom:25px;
        ">
            {input_tokens} input
            ·
            {output_tokens} output
            ·
            {total_tokens} total tokens
        </div>
        """,
        unsafe_allow_html=True
    )

    st.rerun()


# ============================================================
# CURRENT CHAT ACTIONS
# ============================================================

if st.session_state.current_chat_id:

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "🗑 Delete this chat",
            use_container_width=True
        ):

            delete_current_chat()

    with col2:

        if st.button(
            "＋ Start new chat",
            use_container_width=True
        ):

            create_new_chat()

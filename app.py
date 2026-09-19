# ============================================================
# AI DOCUMENT INTELLIGENCE SYSTEM
# ============================================================
# Features:
# - Login / Signup
# - Multiple chat history
# - New Chat
# - Good Morning / Afternoon / Evening greeting
# - PDF / TXT / DOCX / XLSX upload
# - Website URL input
# - RAG
# - FAISS semantic search
# - HuggingFace embeddings
# - HuggingFace local LLM
# - Token usage tracking
# - Per-chat token usage
# - Per-user token usage
# ============================================================

import os
import json
import uuid
import tempfile

import streamlit as st
import pandas as pd
import requests

from bs4 import BeautifulSoup
from datetime import datetime
from dotenv import load_dotenv


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Document Intelligence System",
    page_icon="📄",
    layout="wide"
)

load_dotenv()


# ============================================================
# LANGCHAIN IMPORTS
# ============================================================

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


# ============================================================
# HUGGINGFACE
# ============================================================

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline
)


# ============================================================
# FILES
# ============================================================

USER_FILE = "users.json"
CHAT_FILE = "chats.json"
USAGE_FILE = "usage.json"


# ============================================================
# USER STORAGE
# ============================================================

def load_users():

    if not os.path.exists(USER_FILE):

        data = {
            "users": {},
            "count": 0
        }

        with open(USER_FILE, "w") as f:
            json.dump(data, f, indent=2)

        return data

    try:

        with open(USER_FILE, "r") as f:
            data = json.load(f)

    except Exception:

        data = {
            "users": {},
            "count": 0
        }

    data.setdefault("users", {})
    data.setdefault("count", 0)

    return data


def save_users(data):

    with open(USER_FILE, "w") as f:
        json.dump(data, f, indent=2)


# ============================================================
# CHAT STORAGE
# ============================================================

def load_chats():

    if not os.path.exists(CHAT_FILE):
        return {}

    try:

        with open(CHAT_FILE, "r") as f:
            return json.load(f)

    except Exception:

        return {}


def save_chats(data):

    with open(CHAT_FILE, "w") as f:
        json.dump(data, f, indent=2)


# ============================================================
# TOKEN USAGE STORAGE
# ============================================================

def load_usage():

    if not os.path.exists(USAGE_FILE):
        return {}

    try:

        with open(USAGE_FILE, "r") as f:
            return json.load(f)

    except Exception:

        return {}


def save_usage(data):

    with open(USAGE_FILE, "w") as f:
        json.dump(data, f, indent=2)


def initialize_user_usage(username):

    usage = load_usage()

    if username not in usage:

        usage[username] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "requests": 0
        }

        save_usage(usage)


def update_user_usage(
    username,
    input_tokens,
    output_tokens
):

    usage = load_usage()

    if username not in usage:

        usage[username] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "requests": 0
        }

    usage[username]["input_tokens"] += input_tokens

    usage[username]["output_tokens"] += output_tokens

    usage[username]["total_tokens"] += (
        input_tokens + output_tokens
    )

    usage[username]["requests"] += 1

    save_usage(usage)


# ============================================================
# SESSION STATE
# ============================================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "document_text" not in st.session_state:
    st.session_state.document_text = ""

if "document_name" not in st.session_state:
    st.session_state.document_name = ""

if "source_type" not in st.session_state:
    st.session_state.source_type = ""

if "chat_loaded" not in st.session_state:
    st.session_state.chat_loaded = False

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

if "url_key" not in st.session_state:
    st.session_state.url_key = 0


# ============================================================
# EMBEDDINGS
# ============================================================

@st.cache_resource
def load_embeddings():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# ============================================================
# AI MODEL
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
        max_new_tokens=200,
        do_sample=False,
        return_full_text=False,
        pad_token_id=tokenizer.eos_token_id
    )

    return generator, tokenizer


# ============================================================
# TOKEN COUNT
# ============================================================

def count_tokens(
    tokenizer,
    text
):

    try:

        tokens = tokenizer.encode(
            text,
            add_special_tokens=True
        )

        return len(tokens)

    except Exception:

        return 0


# ============================================================
# AUTHENTICATION
# ============================================================

def authentication_page():

    st.title(
        "🔐 AI Document Intelligence"
    )

    st.write(
        "Login or create an account to use the system."
    )

    option = st.radio(
        "Choose an option",
        ["Login", "Create Account"]
    )

    data = load_users()


    # ========================================================
    # LOGIN
    # ========================================================

    if option == "Login":

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

            if (
                username in data["users"]
                and data["users"][username] == password
            ):

                st.session_state.logged_in = True

                st.session_state.username = username

                initialize_user_usage(
                    username
                )

                st.session_state.current_chat_id = None

                st.session_state.messages = []

                st.session_state.retriever = None

                st.session_state.document_text = ""

                st.session_state.document_name = ""

                st.session_state.source_type = ""

                st.session_state.chat_loaded = False

                st.session_state.uploader_key += 1

                st.session_state.url_key += 1

                st.success(
                    f"Welcome {username}!"
                )

                st.rerun()

            else:

                st.error(
                    "Invalid username or password."
                )


    # ========================================================
    # CREATE ACCOUNT
    # ========================================================

    else:

        new_username = st.text_input(
            "Create Username",
            key="new_username"
        )

        new_password = st.text_input(
            "Create Password",
            type="password",
            key="new_password"
        )

        if st.button(
            "Create Account",
            use_container_width=True
        ):

            if not new_username or not new_password:

                st.warning(
                    "Please fill all fields."
                )

            elif new_username in data["users"]:

                st.warning(
                    "Username already exists."
                )

            else:

                data["users"][new_username] = (
                    new_password
                )

                data["count"] += 1

                save_users(data)

                initialize_user_usage(
                    new_username
                )

                st.success(
                    "Account created successfully. Please login."
                )


# ============================================================
# PROTECT APP
# ============================================================

if not st.session_state.logged_in:

    authentication_page()

    st.stop()


# ============================================================
# MAIN TITLE
# ============================================================

st.title(
    "📄 AI Document Intelligence System"
)

st.caption(
    "Upload documents or website links and ask questions using RAG and semantic search."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "💬 Chat History"
)

st.sidebar.write(
    f"👤 {st.session_state.username}"
)


# ============================================================
# NEW CHAT
# ============================================================

if st.sidebar.button(
    "➕ New Chat",
    use_container_width=True
):

    st.session_state.current_chat_id = str(
        uuid.uuid4()
    )

    st.session_state.messages = []

    st.session_state.retriever = None

    st.session_state.document_text = ""

    st.session_state.document_name = ""

    st.session_state.source_type = ""

    st.session_state.chat_loaded = False

    # Reset file uploader
    st.session_state.uploader_key += 1

    # Reset URL input
    st.session_state.url_key += 1

    st.rerun()


# ============================================================
# CHAT HISTORY
# ============================================================

all_chats = load_chats()

user_chats = []

for chat_id, chat in all_chats.items():

    if chat.get(
        "username"
    ) == st.session_state.username:

        user_chats.append(
            (chat_id, chat)
        )


user_chats.sort(
    key=lambda x: x[1].get(
        "updated_at",
        ""
    ),
    reverse=True
)


# ============================================================
# DISPLAY HISTORY
# ============================================================

for chat_id, chat in user_chats:

    title = chat.get(
        "title",
        "New Chat"
    )

    if len(title) > 35:

        title = title[:35] + "..."

    if st.sidebar.button(
        f"💬 {title}",
        key=f"chat_{chat_id}",
        use_container_width=True
    ):

        selected_chat = all_chats.get(
            chat_id
        )

        if selected_chat:

            st.session_state.current_chat_id = (
                chat_id
            )

            st.session_state.messages = (
                selected_chat.get(
                    "messages",
                    []
                )
            )

            st.session_state.document_text = (
                selected_chat.get(
                    "document_text",
                    ""
                )
            )

            st.session_state.document_name = (
                selected_chat.get(
                    "document_name",
                    ""
                )
            )

            st.session_state.source_type = (
                selected_chat.get(
                    "source_type",
                    ""
                )
            )

            st.session_state.chat_loaded = True

            st.session_state.uploader_key += 1

            st.session_state.url_key += 1


            # Rebuild retriever
            if st.session_state.document_text:

                try:

                    embeddings = load_embeddings()

                    document = Document(
                        page_content=(
                            st.session_state.document_text
                        )
                    )

                    splitter = (
                        RecursiveCharacterTextSplitter(
                            chunk_size=1000,
                            chunk_overlap=200
                        )
                    )

                    chunks = (
                        splitter.split_documents(
                            [document]
                        )
                    )

                    vectorstore = (
                        FAISS.from_documents(
                            chunks,
                            embeddings
                        )
                    )

                    st.session_state.retriever = (
                        vectorstore.as_retriever(
                            search_kwargs={
                                "k": 4
                            }
                        )
                    )

                except Exception as e:

                    st.error(
                        "Could not rebuild document search."
                    )

                    st.exception(e)

            else:

                st.session_state.retriever = None

            st.rerun()


# ============================================================
# TOKEN USAGE SIDEBAR
# ============================================================

usage_data = load_usage()

current_usage = usage_data.get(
    st.session_state.username,
    {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "requests": 0
    }
)

st.sidebar.divider()

st.sidebar.subheader(
    "📊 Token Usage"
)

st.sidebar.write(
    f"Input: **{current_usage['input_tokens']:,}**"
)

st.sidebar.write(
    f"Output: **{current_usage['output_tokens']:,}**"
)

st.sidebar.write(
    f"Total: **{current_usage['total_tokens']:,}**"
)

st.sidebar.write(
    f"Requests: **{current_usage['requests']}**"
)


# ============================================================
# DELETE CURRENT CHAT
# ============================================================

if (
    st.session_state.current_chat_id
    and st.session_state.current_chat_id in all_chats
):

    if st.sidebar.button(
        "🗑️ Delete Current Chat",
        use_container_width=True
    ):

        del all_chats[
            st.session_state.current_chat_id
        ]

        save_chats(
            all_chats
        )

        st.session_state.current_chat_id = None

        st.session_state.messages = []

        st.session_state.retriever = None

        st.session_state.document_text = ""

        st.session_state.document_name = ""

        st.session_state.source_type = ""

        st.session_state.uploader_key += 1

        st.session_state.url_key += 1

        st.rerun()


# ============================================================
# LOGOUT
# ============================================================

if st.sidebar.button(
    "🚪 Logout",
    use_container_width=True
):

    st.session_state.logged_in = False

    st.session_state.username = ""

    st.session_state.current_chat_id = None

    st.session_state.messages = []

    st.session_state.retriever = None

    st.session_state.document_text = ""

    st.session_state.document_name = ""

    st.session_state.source_type = ""

    st.session_state.uploader_key += 1

    st.session_state.url_key += 1

    st.rerun()


# ============================================================
# REGISTERED USERS
# ============================================================

user_data = load_users()

st.sidebar.write(
    f"👥 Registered Users: "
    f"{user_data.get('count', 0)}"
)


# ============================================================
# GREETING
# ============================================================

if (
    not st.session_state.messages
    and not st.session_state.document_name
):

    current_hour = datetime.now().hour

    if current_hour < 12:

        greeting = "Good Morning"

    elif current_hour < 17:

        greeting = "Good Afternoon"

    else:

        greeting = "Good Evening"


    st.markdown(
        f"""
        ### 👋 {greeting}, {st.session_state.username}!

        Welcome to a new chat.

        📄 Upload a document or 🔗 enter a website link
        and ask me anything about it.
        """
    )


# ============================================================
# SOURCE TYPE
# ============================================================

st.subheader(
    "📚 Choose Your Source"
)

source_option = st.radio(
    "Select source",
    [
        "📄 Upload Document",
        "🔗 Website Link"
    ],
    horizontal=True
)


# ============================================================
# DOCUMENT UPLOAD
# ============================================================

if source_option == "📄 Upload Document":

    st.subheader(
        "📁 Upload Document"
    )

    uploaded_file = st.file_uploader(
        "Upload PDF, TXT, DOCX or XLSX",
        type=[
            "pdf",
            "txt",
            "docx",
            "xlsx"
        ],
        key=(
            f"document_uploader_"
            f"{st.session_state.uploader_key}"
        )
    )

    # ========================================================
    # PROCESS FILE
    # ========================================================

    if uploaded_file:

        if (
            st.session_state.document_name
            != uploaded_file.name
            or st.session_state.source_type
            != "file"
        ):

            with st.spinner(
                "Processing document..."
            ):

                try:

                    file_type = (
                        uploaded_file.name
                        .split(".")[-1]
                        .lower()
                    )


                    # =================================================
                    # PDF
                    # =================================================

                    if file_type == "pdf":

                        with tempfile.NamedTemporaryFile(
                            delete=False,
                            suffix=".pdf"
                        ) as temp_file:

                            temp_file.write(
                                uploaded_file.read()
                            )

                            pdf_path = temp_file.name

                        loader = PyPDFLoader(
                            pdf_path
                        )

                        docs = loader.load()

                        os.remove(
                            pdf_path
                        )


                    # =================================================
                    # TXT
                    # =================================================

                    elif file_type == "txt":

                        text = (
                            uploaded_file
                            .read()
                            .decode(
                                "utf-8",
                                errors="ignore"
                            )
                        )

                        docs = [
                            Document(
                                page_content=text
                            )
                        ]


                    # =================================================
                    # DOCX
                    # =================================================

                    elif file_type == "docx":

                        from docx import Document as DocxDocument

                        docx_file = (
                            DocxDocument(
                                uploaded_file
                            )
                        )

                        text = "\n".join(
                            paragraph.text
                            for paragraph
                            in docx_file.paragraphs
                        )

                        docs = [
                            Document(
                                page_content=text
                            )
                        ]


                    # =================================================
                    # XLSX
                    # =================================================

                    elif file_type == "xlsx":

                        df = pd.read_excel(
                            uploaded_file
                        )

                        text = df.to_string(
                            index=False
                        )

                        docs = [
                            Document(
                                page_content=text
                            )
                        ]


                    else:

                        st.error(
                            "Unsupported file type."
                        )

                        st.stop()


                    # =================================================
                    # COMBINE TEXT
                    # =================================================

                    document_text = "\n\n".join(
                        doc.page_content
                        for doc in docs
                    )

                    st.session_state.document_text = (
                        document_text
                    )

                    st.session_state.document_name = (
                        uploaded_file.name
                    )

                    st.session_state.source_type = (
                        "file"
                    )


                    # =================================================
                    # SPLIT
                    # =================================================

                    splitter = (
                        RecursiveCharacterTextSplitter(
                            chunk_size=1000,
                            chunk_overlap=200
                        )
                    )

                    chunks = (
                        splitter.split_documents(
                            docs
                        )
                    )


                    # =================================================
                    # EMBEDDINGS
                    # =================================================

                    embeddings = (
                        load_embeddings()
                    )


                    # =================================================
                    # FAISS
                    # =================================================

                    vectorstore = (
                        FAISS.from_documents(
                            chunks,
                            embeddings
                        )
                    )


                    st.session_state.retriever = (
                        vectorstore.as_retriever(
                            search_kwargs={
                                "k": 4
                            }
                        )
                    )


                    st.session_state.messages = []


                    st.success(
                        f"✅ {uploaded_file.name} "
                        f"processed successfully!"
                    )


                except Exception as e:

                    st.error(
                        "Error while processing the document."
                    )

                    st.exception(e)


# ============================================================
# WEBSITE URL
# ============================================================

else:

    st.subheader(
        "🔗 Website Link"
    )

    website_url = st.text_input(
        "Paste a website URL",
        placeholder="https://example.com/article",
        key=f"website_url_{st.session_state.url_key}"
    )

    load_url = st.button(
        "🌐 Load Website",
        use_container_width=True
    )


    # ========================================================
    # LOAD WEBSITE
    # ========================================================

    if load_url:

        if not website_url:

            st.warning(
                "Please enter a website URL."
            )

        elif not (
            website_url.startswith("http://")
            or website_url.startswith("https://")
        ):

            st.error(
                "Please enter a valid URL starting with http:// or https://"
            )

        else:

            with st.spinner(
                "Reading website..."
            ):

                try:

                    response = requests.get(
                        website_url,
                        timeout=20,
                        headers={
                            "User-Agent":
                            "Mozilla/5.0"
                        }
                    )

                    response.raise_for_status()


                    # =================================================
                    # EXTRACT WEBSITE TEXT
                    # =================================================

                    soup = BeautifulSoup(
                        response.text,
                        "html.parser"
                    )


                    # Remove unnecessary elements
                    for element in soup(
                        [
                            "script",
                            "style",
                            "noscript",
                            "header",
                            "footer",
                            "nav"
                        ]
                    ):

                        element.decompose()


                    text = soup.get_text(
                        separator="\n"
                    )

                    # Clean empty lines
                    lines = [
                        line.strip()
                        for line in text.splitlines()
                        if line.strip()
                    ]

                    website_text = "\n".join(
                        lines
                    )


                    if len(website_text) < 50:

                        st.error(
                            "Could not extract enough text from this website."
                        )

                        st.stop()


                    # =================================================
                    # CREATE DOCUMENT
                    # =================================================

                    docs = [
                        Document(
                            page_content=website_text,
                            metadata={
                                "source": website_url
                            }
                        )
                    ]


                    # =================================================
                    # SAVE SOURCE
                    # =================================================

                    st.session_state.document_text = (
                        website_text
                    )

                    st.session_state.document_name = (
                        website_url
                    )

                    st.session_state.source_type = (
                        "url"
                    )


                    # =================================================
                    # SPLIT WEBSITE
                    # =================================================

                    splitter = (
                        RecursiveCharacterTextSplitter(
                            chunk_size=1000,
                            chunk_overlap=200
                        )
                    )

                    chunks = (
                        splitter.split_documents(
                            docs
                        )
                    )


                    # =================================================
                    # EMBEDDINGS
                    # =================================================

                    embeddings = (
                        load_embeddings()
                    )


                    # =================================================
                    # FAISS
                    # =================================================

                    vectorstore = (
                        FAISS.from_documents(
                            chunks,
                            embeddings
                        )
                    )


                    st.session_state.retriever = (
                        vectorstore.as_retriever(
                            search_kwargs={
                                "k": 4
                            }
                        )
                    )


                    st.session_state.messages = []


                    st.success(
                        "✅ Website loaded successfully!"
                    )


                except requests.exceptions.RequestException as e:

                    st.error(
                        "Could not access this website."
                    )

                    st.write(
                        str(e)
                    )

                except Exception as e:

                    st.error(
                        "Error while processing website."
                    )

                    st.exception(e)


# ============================================================
# CURRENT SOURCE
# ============================================================

if st.session_state.document_name:

    if st.session_state.source_type == "url":

        st.info(
            f"🔗 Current Website: "
            f"{st.session_state.document_name}"
        )

    else:

        st.info(
            f"📄 Current Document: "
            f"{st.session_state.document_name}"
        )


# ============================================================
# CHAT
# ============================================================

st.divider()

st.subheader(
    "💬 Ask Questions"
)


# ============================================================
# DISPLAY OLD MESSAGES
# ============================================================

for message in st.session_state.messages:

    role = message.get(
        "role",
        "assistant"
    )

    content = message.get(
        "content",
        ""
    )

    with st.chat_message(
        role
    ):

        st.write(
            content
        )

        # Show token information if available
        if role == "assistant":

            if "input_tokens" in message:

                st.caption(
                    f"📊 Input: "
                    f"{message['input_tokens']:,} tokens | "
                    f"Output: "
                    f"{message['output_tokens']:,} tokens | "
                    f"Total: "
                    f"{message['total_tokens']:,} tokens"
                )


# ============================================================
# CHAT INPUT
# ============================================================

if st.session_state.retriever:

    question = st.chat_input(
        "Ask a question about your document or website..."
    )

    if question:

        # ====================================================
        # USER MESSAGE
        # ====================================================

        with st.chat_message(
            "user"
        ):

            st.write(
                question
            )


        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )


        # ====================================================
        # SEARCH
        # ====================================================

        with st.spinner(
            "Searching..."
        ):

            try:

                docs = (
                    st.session_state
                    .retriever
                    .invoke(
                        question
                    )
                )

                context = "\n\n".join(
                    doc.page_content
                    for doc in docs
                )

            except Exception as e:

                st.error(
                    "Error while searching."
                )

                st.exception(e)

                st.stop()


        # ====================================================
        # RAG PROMPT
        # ====================================================

        prompt = f"""
You are an AI document question-answering assistant.

Answer the user's question using ONLY the information
provided in the context below.

Do not make up information.

If the answer is not available in the context,
say exactly:

"I could not find that information in the uploaded document."

Keep the answer clear and easy to understand.

CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""


        # ====================================================
        # LOAD MODEL
        # ====================================================

        with st.spinner(
            "AI is generating the answer..."
        ):

            try:

                llm, tokenizer = load_llm()


                # =================================================
                # COUNT INPUT TOKENS
                # =================================================

                input_tokens = count_tokens(
                    tokenizer,
                    prompt
                )


                # =================================================
                # GENERATE
                # =================================================

                result = llm(
                    prompt
                )


                answer = result[0][
                    "generated_text"
                ].strip()


                if not answer:

                    answer = (
                        "I could not generate an answer."
                    )


                # =================================================
                # COUNT OUTPUT TOKENS
                # =================================================

                output_tokens = count_tokens(
                    tokenizer,
                    answer
                )


                total_tokens = (
                    input_tokens
                    + output_tokens
                )


            except Exception as e:

                answer = (
                    "The AI model could not be loaded. "
                    "Please try again."
                )

                input_tokens = 0
                output_tokens = 0
                total_tokens = 0

                st.error(
                    "AI model error."
                )

                st.exception(e)


        # ====================================================
        # SHOW ANSWER
        # ====================================================

        with st.chat_message(
            "assistant"
        ):

            st.write(
                answer
            )

            st.caption(
                f"📊 Input: "
                f"{input_tokens:,} tokens | "
                f"Output: "
                f"{output_tokens:,} tokens | "
                f"Total: "
                f"{total_tokens:,} tokens"
            )


        # ====================================================
        # SAVE ASSISTANT MESSAGE
        # ====================================================

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens
            }
        )


        # ====================================================
        # UPDATE USER TOKEN USAGE
        # ====================================================

        update_user_usage(
            st.session_state.username,
            input_tokens,
            output_tokens
        )


        # ====================================================
        # SAVE CHAT
        # ====================================================

        all_chats = load_chats()


        if not st.session_state.current_chat_id:

            st.session_state.current_chat_id = str(
                uuid.uuid4()
            )


        chat_id = (
            st.session_state.current_chat_id
        )


        # ====================================================
        # CHAT TITLE
        # ====================================================

        title = question.strip()

        if len(title) > 45:

            title = title[:45] + "..."


        # ====================================================
        # CHAT TOTAL TOKEN USAGE
        # ====================================================

        existing_chat = all_chats.get(
            chat_id,
            {}
        )

        previous_chat_tokens = existing_chat.get(
            "total_tokens",
            0
        )

        chat_total_tokens = (
            previous_chat_tokens
            + total_tokens
        )


        # ====================================================
        # UPDATE EXISTING CHAT
        # ====================================================

        if chat_id in all_chats:

            all_chats[chat_id][
                "messages"
            ] = st.session_state.messages

            all_chats[chat_id][
                "document_text"
            ] = st.session_state.document_text

            all_chats[chat_id][
                "document_name"
            ] = st.session_state.document_name

            all_chats[chat_id][
                "source_type"
            ] = st.session_state.source_type

            all_chats[chat_id][
                "total_tokens"
            ] = chat_total_tokens

            all_chats[chat_id][
                "updated_at"
            ] = datetime.now().isoformat()


        # ====================================================
        # CREATE NEW CHAT
        # ====================================================

        else:

            all_chats[chat_id] = {

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


        save_chats(
            all_chats
        )


else:

    st.info(
        "📁 Upload a document or 🔗 load a website first."
    )

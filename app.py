# ============================================================
# AI DOCUMENT INTELLIGENCE SYSTEM
# RAG + FAISS + HuggingFace + Chat History
# PDF + TXT + DOCX + XLSX
# ============================================================

import os
import json
import uuid
import tempfile

import streamlit as st
import pandas as pd

from dotenv import load_dotenv
from langchain_core.documents import Document

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
# IMPORTS
# ============================================================

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


# ============================================================
# FILES
# ============================================================

USER_FILE = "users.json"
CHAT_FILE = "chats.json"


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
    "chat_loaded": False
}

for key, value in defaults.items():

    if key not in st.session_state:
        st.session_state[key] = value


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

    return generator


# ============================================================
# AUTHENTICATION
# ============================================================

def authentication_page():

    st.title("🔐 AI Document Intelligence")

    st.write(
        "Login or create an account to use the document intelligence system."
    )

    option = st.radio(
        "Choose an option",
        ["Login", "Create Account"]
    )

    data = load_users()

    # --------------------------------------------------------
    # LOGIN
    # --------------------------------------------------------

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

                st.session_state.current_chat_id = None
                st.session_state.messages = []
                st.session_state.retriever = None
                st.session_state.document_text = ""
                st.session_state.document_name = ""

                st.success("Login successful!")

                st.rerun()

            else:

                st.error(
                    "Invalid username or password."
                )

    # --------------------------------------------------------
    # CREATE ACCOUNT
    # --------------------------------------------------------

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

                data["users"][new_username] = new_password
                data["count"] += 1

                save_users(data)

                st.success(
                    "Account created successfully. Please login."
                )


# ============================================================
# STOP IF NOT LOGGED IN
# ============================================================

if not st.session_state.logged_in:

    authentication_page()

    st.stop()


# ============================================================
# MAIN APP
# ============================================================

st.title("📄 AI Document Intelligence System")

st.caption(
    "Upload documents and ask questions using RAG and semantic search."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("💬 Chat History")

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

    st.session_state.chat_loaded = False

    st.rerun()


# ============================================================
# LOAD USER CHATS
# ============================================================

all_chats = load_chats()

user_chats = []

for chat_id, chat in all_chats.items():

    if chat.get("username") == st.session_state.username:

        user_chats.append(
            (chat_id, chat)
        )


# Sort newest first
user_chats.sort(
    key=lambda x: x[1].get("updated_at", ""),
    reverse=True
)


# ============================================================
# DISPLAY CHAT HISTORY
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

            st.session_state.current_chat_id = chat_id

            st.session_state.messages = selected_chat.get(
                "messages",
                []
            )

            st.session_state.document_text = selected_chat.get(
                "document_text",
                ""
            )

            st.session_state.document_name = selected_chat.get(
                "document_name",
                ""
            )

            st.session_state.chat_loaded = True

            # Rebuild retriever
            if st.session_state.document_text:

                try:

                    embeddings = load_embeddings()

                    document = Document(
                        page_content=st.session_state.document_text
                    )

                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200
                    )

                    chunks = splitter.split_documents(
                        [document]
                    )

                    vectorstore = FAISS.from_documents(
                        chunks,
                        embeddings
                    )

                    st.session_state.retriever = (
                        vectorstore.as_retriever(
                            search_kwargs={"k": 4}
                        )
                    )

                except Exception as e:

                    st.error(
                        "Could not rebuild document search."
                    )

                    st.exception(e)

            st.rerun()


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

        save_chats(all_chats)

        st.session_state.current_chat_id = None
        st.session_state.messages = []
        st.session_state.retriever = None
        st.session_state.document_text = ""
        st.session_state.document_name = ""

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

    st.rerun()


# ============================================================
# USER COUNT
# ============================================================

user_data = load_users()

st.sidebar.divider()

st.sidebar.write(
    f"👥 Registered Users: {user_data.get('count', 0)}"
)


# ============================================================
# FILE UPLOAD
# ============================================================

st.subheader("📁 Upload Document")

uploaded_file = st.file_uploader(
    "Upload PDF, TXT, DOCX or XLSX",
    type=[
        "pdf",
        "txt",
        "docx",
        "xlsx"
    ]
)


# ============================================================
# PROCESS DOCUMENT
# ============================================================

if uploaded_file:

    if (
        st.session_state.document_name
        != uploaded_file.name
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

                # ------------------------------------------------
                # PDF
                # ------------------------------------------------

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

                # ------------------------------------------------
                # TXT
                # ------------------------------------------------

                elif file_type == "txt":

                    text = uploaded_file.read().decode(
                        "utf-8",
                        errors="ignore"
                    )

                    docs = [
                        Document(
                            page_content=text
                        )
                    ]

                # ------------------------------------------------
                # DOCX
                # ------------------------------------------------

                elif file_type == "docx":

                    from docx import Document as DocxDocument

                    docx_file = DocxDocument(
                        uploaded_file
                    )

                    text = "\n".join(
                        paragraph.text
                        for paragraph in docx_file.paragraphs
                    )

                    docs = [
                        Document(
                            page_content=text
                        )
                    ]

                # ------------------------------------------------
                # XLSX
                # ------------------------------------------------

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

                # ------------------------------------------------
                # SAVE DOCUMENT TEXT
                # ------------------------------------------------

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

                # ------------------------------------------------
                # SPLIT DOCUMENT
                # ------------------------------------------------

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )

                chunks = splitter.split_documents(
                    docs
                )

                # ------------------------------------------------
                # EMBEDDINGS
                # ------------------------------------------------

                embeddings = load_embeddings()

                # ------------------------------------------------
                # FAISS
                # ------------------------------------------------

                vectorstore = FAISS.from_documents(
                    chunks,
                    embeddings
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
                    f"✅ {uploaded_file.name} processed successfully!"
                )

            except Exception as e:

                st.error(
                    "Error while processing the document."
                )

                st.exception(e)


# ============================================================
# DOCUMENT STATUS
# ============================================================

if st.session_state.document_name:

    st.info(
        f"📄 Current Document: "
        f"{st.session_state.document_name}"
    )


# ============================================================
# CHAT
# ============================================================

st.divider()

st.subheader("💬 Ask Questions")


# ============================================================
# DISPLAY PREVIOUS MESSAGES
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

    with st.chat_message(role):

        st.write(content)


# ============================================================
# CHAT INPUT
# ============================================================

if st.session_state.retriever:

    question = st.chat_input(
        "Ask a question about your document..."
    )

    if question:

        # --------------------------------------------------------
        # SHOW USER MESSAGE
        # --------------------------------------------------------

        with st.chat_message("user"):

            st.write(question)

        # --------------------------------------------------------
        # SAVE USER MESSAGE
        # --------------------------------------------------------

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        # --------------------------------------------------------
        # SEARCH DOCUMENT
        # --------------------------------------------------------

        with st.spinner(
            "Searching the document..."
        ):

            try:

                docs = (
                    st.session_state
                    .retriever
                    .invoke(question)
                )

                context = "\n\n".join(
                    doc.page_content
                    for doc in docs
                )

            except Exception as e:

                st.error(
                    "Error while searching the document."
                )

                st.exception(e)

                st.stop()

        # --------------------------------------------------------
        # CREATE RAG PROMPT
        # --------------------------------------------------------

        prompt = f"""
You are a document question-answering assistant.

Answer the user's question using ONLY the information
provided in the document context below.

If the answer is not available in the document,
say:

"I could not find that information in the uploaded document."

Do not make up information.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""

        # --------------------------------------------------------
        # LOAD AI MODEL
        # --------------------------------------------------------

        with st.spinner(
            "AI is generating the answer..."
        ):

            try:

                llm = load_llm()

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

            except Exception as e:

                answer = (
                    "The AI model could not be loaded. "
                    "Please try again."
                )

                st.error(
                    "AI model error"
                )

                st.exception(e)

        # --------------------------------------------------------
        # SHOW ANSWER
        # --------------------------------------------------------

        with st.chat_message(
            "assistant"
        ):

            st.write(answer)

        # --------------------------------------------------------
        # SAVE ASSISTANT MESSAGE
        # --------------------------------------------------------

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        # ========================================================
        # SAVE CHAT
        # ========================================================

        all_chats = load_chats()

        # Create chat if necessary
        if not st.session_state.current_chat_id:

            st.session_state.current_chat_id = str(
                uuid.uuid4()
            )

        chat_id = (
            st.session_state.current_chat_id
        )

        # Create title from first question
        title = question.strip()

        if len(title) > 45:

            title = title[:45] + "..."

        # Existing chat
        if chat_id in all_chats:

            all_chats[chat_id]["messages"] = (
                st.session_state.messages
            )

            all_chats[chat_id]["document_text"] = (
                st.session_state.document_text
            )

            all_chats[chat_id]["document_name"] = (
                st.session_state.document_name
            )

            all_chats[chat_id]["updated_at"] = (
                str(__import__("datetime").datetime.now())
            )

        # New chat
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

                "updated_at":
                    str(__import__("datetime").datetime.now())
            }

        save_chats(
            all_chats
        )

else:

    st.info(
        "📁 Upload a document first to start chatting."
    )

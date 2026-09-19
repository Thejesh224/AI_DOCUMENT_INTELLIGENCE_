# ============================================================
# AI DOCUMENT INTELLIGENCE SYSTEM
# LOGIN + MULTI-FILE RAG + CHATGPT STYLE HISTORY
# ============================================================

import os
import json
import uuid
import tempfile
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

# ============================================================
# PAGE CONFIG
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
# TRANSFORMERS
# ============================================================

from transformers import pipeline


# ============================================================
# FILES
# ============================================================

USER_FILE = "users.json"
CHAT_FILE = "chats.json"


# ============================================================
# USER FUNCTIONS
# ============================================================

def load_users():

    if not os.path.exists(USER_FILE):

        data = {
            "users": {},
            "count": 0
        }

        with open(USER_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

        return data

    try:

        with open(USER_FILE, "r", encoding="utf-8") as f:
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

    with open(USER_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


# ============================================================
# CHAT HISTORY FUNCTIONS
# ============================================================

def load_chats():

    if not os.path.exists(CHAT_FILE):

        data = {
            "users": {}
        }

        with open(CHAT_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

        return data

    try:

        with open(CHAT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

    except Exception:

        data = {
            "users": {}
        }

    data.setdefault("users", {})

    return data


def save_chats(data):

    with open(CHAT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def create_chat(username):

    chats = load_chats()

    if username not in chats["users"]:
        chats["users"][username] = []

    chat_id = str(uuid.uuid4())

    chat = {
        "id": chat_id,
        "title": "New Chat",
        "created_at": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "updated_at": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "messages": [],
        "documents": []
    }

    chats["users"][username].insert(0, chat)

    save_chats(chats)

    return chat_id


def get_user_chats(username):

    chats = load_chats()

    return chats["users"].get(username, [])


def get_chat(username, chat_id):

    user_chats = get_user_chats(username)

    for chat in user_chats:

        if chat["id"] == chat_id:
            return chat

    return None


def update_chat(
    username,
    chat_id,
    messages=None,
    documents=None,
    title=None
):

    chats = load_chats()

    user_chats = chats["users"].get(username, [])

    for chat in user_chats:

        if chat["id"] == chat_id:

            if messages is not None:
                chat["messages"] = messages

            if documents is not None:
                chat["documents"] = documents

            if title is not None:
                chat["title"] = title

            chat["updated_at"] = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            break

    # Put recently updated chat first
    user_chats.sort(
        key=lambda x: x.get("updated_at", ""),
        reverse=True
    )

    save_chats(chats)


def delete_chat(username, chat_id):

    chats = load_chats()

    if username not in chats["users"]:
        return

    chats["users"][username] = [
        chat
        for chat in chats["users"][username]
        if chat["id"] != chat_id
    ]

    save_chats(chats)


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

if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False


# ============================================================
# CACHED EMBEDDINGS
# ============================================================

@st.cache_resource
def load_embeddings():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# ============================================================
# CACHED LLM
# ============================================================

@st.cache_resource
def load_llm():

    return pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        max_new_tokens=150,
        do_sample=False
    )


# ============================================================
# LOAD MODELS
# ============================================================

try:

    embeddings = load_embeddings()

except Exception as e:

    st.error(
        "Could not load the embedding model."
    )

    st.exception(e)

    st.stop()


try:

    llm = load_llm()

except Exception as e:

    st.error(
        "Could not load the AI model."
    )

    st.exception(e)

    st.stop()


# ============================================================
# BUILD RETRIEVER FROM SAVED DOCUMENTS
# ============================================================

def build_retriever(documents):

    if not documents:
        return None

    all_documents = []

    for item in documents:

        all_documents.append(
            Document(
                page_content=item["page_content"],
                metadata=item.get("metadata", {})
            )
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(
        all_documents
    )

    if not chunks:
        return None

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    return vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )


# ============================================================
# AUTH PAGE
# ============================================================

def show_auth_page():

    st.title("🔐 AI Document Intelligence")

    st.write(
        "Login to access your document conversations."
    )

    option = st.radio(
        "Choose an option",
        ["Login", "Create Account"],
        horizontal=True
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
            type="primary",
            use_container_width=True
        ):

            if (
                username in data["users"]
                and data["users"][username] == password
            ):

                st.session_state.logged_in = True
                st.session_state.username = username

                # --------------------------------------------
                # Create first chat if user has none
                # --------------------------------------------

                user_chats = get_user_chats(username)

                if not user_chats:

                    chat_id = create_chat(username)

                else:

                    chat_id = user_chats[0]["id"]

                st.session_state.current_chat_id = chat_id

                # --------------------------------------------
                # Load current chat
                # --------------------------------------------

                chat = get_chat(
                    username,
                    chat_id
                )

                st.session_state.messages = (
                    chat.get("messages", [])
                    if chat
                    else []
                )

                st.session_state.retriever = (
                    build_retriever(
                        chat.get("documents", [])
                    )
                    if chat
                    else None
                )

                st.session_state.documents_loaded = (
                    st.session_state.retriever is not None
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
            type="primary",
            use_container_width=True
        ):

            new_username = new_username.strip()

            if not new_username or not new_password:

                st.warning(
                    "Please fill in all fields."
                )

            elif new_username in data["users"]:

                st.warning(
                    "Username already exists."
                )

            else:

                data["users"][new_username] = new_password

                data["count"] += 1

                save_users(data)

                # Create first chat for user
                create_chat(new_username)

                st.success(
                    "Account created successfully. Please login."
                )


# ============================================================
# STOP IF NOT LOGGED IN
# ============================================================

if not st.session_state.logged_in:

    show_auth_page()

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("💬 AI Chats")

st.sidebar.caption(
    f"👤 {st.session_state.username}"
)


# ============================================================
# NEW CHAT
# ============================================================

if st.sidebar.button(
    "➕ New Chat",
    use_container_width=True
):

    chat_id = create_chat(
        st.session_state.username
    )

    st.session_state.current_chat_id = chat_id
    st.session_state.messages = []
    st.session_state.retriever = None
    st.session_state.documents_loaded = False

    st.rerun()


st.sidebar.divider()


# ============================================================
# CHAT HISTORY
# ============================================================

user_chats = get_user_chats(
    st.session_state.username
)


if user_chats:

    st.sidebar.subheader("History")

    for chat in user_chats:

        title = chat.get(
            "title",
            "New Chat"
        )

        if len(title) > 32:
            title = title[:32] + "..."

        is_current = (
            chat["id"]
            == st.session_state.current_chat_id
        )

        button_text = (
            "🟢 " + title
            if is_current
            else "💬 " + title
        )

        if st.sidebar.button(
            button_text,
            key="history_" + chat["id"],
            use_container_width=True
        ):

            st.session_state.current_chat_id = (
                chat["id"]
            )

            st.session_state.messages = (
                chat.get("messages", [])
            )

            st.session_state.retriever = (
                build_retriever(
                    chat.get("documents", [])
                )
            )

            st.session_state.documents_loaded = (
                st.session_state.retriever is not None
            )

            st.rerun()


# ============================================================
# DELETE CHAT
# ============================================================

st.sidebar.divider()

if st.sidebar.button(
    "🗑️ Delete Current Chat",
    use_container_width=True
):

    current_id = st.session_state.current_chat_id

    if current_id:

        delete_chat(
            st.session_state.username,
            current_id
        )

    # --------------------------------------------
    # Create replacement chat
    # --------------------------------------------

    remaining = get_user_chats(
        st.session_state.username
    )

    if remaining:

        new_current = remaining[0]

        st.session_state.current_chat_id = (
            new_current["id"]
        )

        st.session_state.messages = (
            new_current.get("messages", [])
        )

        st.session_state.retriever = (
            build_retriever(
                new_current.get("documents", [])
            )
        )

        st.session_state.documents_loaded = (
            st.session_state.retriever is not None
        )

    else:

        new_id = create_chat(
            st.session_state.username
        )

        st.session_state.current_chat_id = new_id
        st.session_state.messages = []
        st.session_state.retriever = None
        st.session_state.documents_loaded = False

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
    st.session_state.documents_loaded = False

    st.rerun()


# ============================================================
# MAIN PAGE
# ============================================================

st.title(
    "📄 AI Document Intelligence System"
)


# ============================================================
# CURRENT CHAT
# ============================================================

current_chat = get_chat(
    st.session_state.username,
    st.session_state.current_chat_id
)

if current_chat:

    if current_chat["title"] != "New Chat":

        st.caption(
            "💬 " + current_chat["title"]
        )


# ============================================================
# FILE UPLOAD
# ============================================================

st.subheader("📂 Upload Documents")

uploaded_files = st.file_uploader(
    "Upload PDF, TXT, DOCX or Excel files",
    type=[
        "pdf",
        "txt",
        "docx",
        "xlsx"
    ],
    accept_multiple_files=True
)


# ============================================================
# PROCESS DOCUMENTS
# ============================================================

if uploaded_files:

    current_chat = get_chat(
        st.session_state.username,
        st.session_state.current_chat_id
    )

    existing_documents = (
        current_chat.get("documents", [])
        if current_chat
        else []
    )

    # Only process new upload when current chat
    # doesn't already contain documents

    if not existing_documents:

        all_docs = []

        with st.spinner(
            "Processing your documents..."
        ):

            for uploaded_file in uploaded_files:

                filename = uploaded_file.name

                extension = (
                    filename
                    .split(".")[-1]
                    .lower()
                )

                # ==========================================
                # PDF
                # ==========================================

                if extension == "pdf":

                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=".pdf"
                    ) as temp_file:

                        temp_file.write(
                            uploaded_file.getvalue()
                        )

                        pdf_path = temp_file.name

                    try:

                        loader = PyPDFLoader(
                            pdf_path
                        )

                        docs = loader.load()

                    finally:

                        if os.path.exists(pdf_path):
                            os.remove(pdf_path)

                # ==========================================
                # TXT
                # ==========================================

                elif extension == "txt":

                    text = uploaded_file.getvalue().decode(
                        "utf-8",
                        errors="ignore"
                    )

                    docs = [
                        Document(
                            page_content=text,
                            metadata={}
                        )
                    ]

                # ==========================================
                # DOCX
                # ==========================================

                elif extension == "docx":

                    from docx import Document as WordDocument

                    word_document = WordDocument(
                        uploaded_file
                    )

                    paragraphs = [
                        paragraph.text
                        for paragraph in word_document.paragraphs
                    ]

                    text = "\n".join(
                        paragraphs
                    )

                    docs = [
                        Document(
                            page_content=text,
                            metadata={}
                        )
                    ]

                # ==========================================
                # XLSX
                # ==========================================

                elif extension == "xlsx":

                    import pandas as pd

                    dataframe = pd.read_excel(
                        uploaded_file
                    )

                    text = dataframe.to_string(
                        index=False
                    )

                    docs = [
                        Document(
                            page_content=text,
                            metadata={}
                        )
                    ]

                else:

                    continue

                # ==========================================
                # ADD SOURCE
                # ==========================================

                for doc in docs:

                    doc.metadata["source"] = filename

                all_docs.extend(docs)

        # ==================================================
        # CHECK DOCUMENTS
        # ==================================================

        if not all_docs:

            st.error(
                "No readable content was found."
            )

        else:

            # ==============================================
            # SPLIT DOCUMENTS
            # ==============================================

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=100
            )

            chunks = splitter.split_documents(
                all_docs
            )

            # ==============================================
            # CREATE VECTOR DATABASE
            # ==============================================

            with st.spinner(
                "Creating document search index..."
            ):

                vectorstore = FAISS.from_documents(
                    chunks,
                    embeddings
                )

                st.session_state.retriever = (
                    vectorstore.as_retriever(
                        search_kwargs={"k": 3}
                    )
                )

            st.session_state.documents_loaded = True

            # ==============================================
            # SAVE ORIGINAL DOCUMENTS
            # ==============================================

            saved_documents = []

            for doc in all_docs:

                saved_documents.append(
                    {
                        "page_content": doc.page_content,
                        "metadata": doc.metadata
                    }
                )

            update_chat(
                st.session_state.username,
                st.session_state.current_chat_id,
                documents=saved_documents
            )

            st.success(
                "✅ Documents processed successfully!"
            )


# ============================================================
# DISPLAY DOCUMENT STATUS
# ============================================================

if st.session_state.retriever:

    st.success(
        "📚 Documents are ready. You can ask questions below."
    )

else:

    st.info(
        "Upload a document to start asking questions."
    )


# ============================================================
# CHAT SECTION
# ============================================================

st.divider()

st.subheader("💬 Chat")


# ============================================================
# DISPLAY HISTORY
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
        "Ask something about your document..."
    )

    if question:

        question = question.strip()

        if not question:

            st.warning(
                "Please enter a question."
            )

            st.stop()

        # ==============================================
        # DISPLAY USER QUESTION
        # ==============================================

        with st.chat_message("user"):

            st.write(question)

        # ==============================================
        # SEARCH DOCUMENT
        # ==============================================

        with st.spinner(
            "Searching your document..."
        ):

            try:

                documents = (
                    st.session_state.retriever.invoke(
                        question
                    )
                )

            except Exception as e:

                st.error(
                    "There was a problem searching the document."
                )

                st.exception(e)

                st.stop()

        # ==============================================
        # CREATE CONTEXT
        # ==============================================

        context_parts = []

        for document in documents:

            source = document.metadata.get(
                "source",
                "Document"
            )

            context_parts.append(
                f"Source: {source}\n"
                f"{document.page_content}"
            )

        context = "\n\n".join(
            context_parts
        )

        # Limit context size
        context = context[:4000]

        # ==============================================
        # PROMPT
        # ==============================================

        ai_prompt = f"""
You are an AI document assistant.

Answer the user's question using ONLY the information
provided in the document context.

If the answer cannot be found in the context,
say:

"I could not find that information in the uploaded document."

Do not make up information.

Document Context:
{context}

User Question:
{question}

Answer:
"""

        # ==============================================
        # GENERATE ANSWER
        # ==============================================

        with st.spinner(
            "Generating answer..."
        ):

            try:

                result = llm(
                    ai_prompt
                )

                if isinstance(result, list):

                    answer = result[0].get(
                        "generated_text",
                        ""
                    )

                else:

                    answer = str(result)

                answer = answer.strip()

            except Exception as e:

                st.error(
                    "The AI model could not generate a response."
                )

                st.exception(e)

                st.stop()

        # ==============================================
        # DISPLAY ANSWER
        # ==============================================

        with st.chat_message("assistant"):

            st.write(answer)

        # ==============================================
        # SAVE MESSAGES IN SESSION
        # ==============================================

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        # ==============================================
        # GET CURRENT CHAT
        # ==============================================

        current_chat = get_chat(
            st.session_state.username,
            st.session_state.current_chat_id
        )

        # ==============================================
        # AUTOMATIC TITLE
        # ==============================================

        title = None

        if (
            current_chat
            and current_chat.get("title")
            == "New Chat"
        ):

            title = question

            if len(title) > 40:

                title = title[:40] + "..."

        # ==============================================
        # SAVE CHAT
        # ==============================================

        update_chat(
            st.session_state.username,
            st.session_state.current_chat_id,
            messages=st.session_state.messages,
            title=title
        )

        # ==============================================
        # REFRESH SIDEBAR
        # ==============================================

        st.rerun()

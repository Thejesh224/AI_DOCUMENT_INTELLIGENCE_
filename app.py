# ============================================================
# AI DOCUMENT INTELLIGENCE SYSTEM
# LOGIN + MULTI-FILE + RAG + CHATGPT-STYLE CHAT HISTORY
# ============================================================

import tempfile
import streamlit as st
import json
import os
import uuid
from datetime import datetime

from dotenv import load_dotenv

st.set_page_config(
    page_title="AI Document Intelligence System",
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
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document

from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline


# ============================================================
# FILE STORAGE
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
            json.dump(data, f)

        return data

    with open(USER_FILE, "r") as f:

        try:
            data = json.load(f)

        except:
            data = {
                "users": {},
                "count": 0
            }

    data.setdefault("users", {})
    data.setdefault("count", 0)

    return data


def save_users(data):

    with open(USER_FILE, "w") as f:
        json.dump(data, f, indent=4)


# ============================================================
# CHAT STORAGE
# ============================================================

def load_chats():

    if not os.path.exists(CHAT_FILE):

        data = {
            "users": {}
        }

        with open(CHAT_FILE, "w") as f:
            json.dump(data, f, indent=4)

        return data

    with open(CHAT_FILE, "r") as f:

        try:
            data = json.load(f)

        except:
            data = {
                "users": {}
            }

    data.setdefault("users", {})

    return data


def save_chats(data):

    with open(CHAT_FILE, "w") as f:
        json.dump(data, f, indent=4)


# ============================================================
# CREATE NEW CHAT
# ============================================================

def create_new_chat(username):

    chats = load_chats()

    chat_id = str(uuid.uuid4())

    new_chat = {
        "id": chat_id,
        "title": "New Chat",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "messages": [],
        "documents": []
    }

    if username not in chats["users"]:
        chats["users"][username] = []

    chats["users"][username].insert(0, new_chat)

    save_chats(chats)

    return chat_id


# ============================================================
# GET USER CHATS
# ============================================================

def get_user_chats(username):

    chats = load_chats()

    return chats["users"].get(username, [])


# ============================================================
# GET CURRENT CHAT
# ============================================================

def get_chat(username, chat_id):

    user_chats = get_user_chats(username)

    for chat in user_chats:

        if chat["id"] == chat_id:
            return chat

    return None


# ============================================================
# UPDATE CHAT
# ============================================================

def update_chat(username, chat_id, messages=None, documents=None, title=None):

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

    save_chats(chats)


# ============================================================
# DELETE CHAT
# ============================================================

def delete_chat(username, chat_id):

    chats = load_chats()

    user_chats = chats["users"].get(username, [])

    chats["users"][username] = [
        chat for chat in user_chats
        if chat["id"] != chat_id
    ]

    save_chats(chats)


# ============================================================
# REBUILD RETRIEVER FROM SAVED DOCUMENTS
# ============================================================

def rebuild_retriever(documents):

    if not documents:
        return None

    all_docs = []

    for item in documents:

        doc = Document(
            page_content=item["page_content"],
            metadata=item.get("metadata", {})
        )

        all_docs.append(doc)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(all_docs)

    vectorstore = FAISS.from_documents(
        chunks,
        st.session_state.embeddings
    )

    return vectorstore.as_retriever(
        search_kwargs={"k": 2}
    )


# ============================================================
# SESSION STATE
# ============================================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

if "user_count" not in st.session_state:
    st.session_state.user_count = load_users()["count"]

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None

if "embeddings" not in st.session_state:

    st.session_state.embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# ============================================================
# AUTH PAGE
# ============================================================

def auth_page():

    st.title("🔐 Authentication")

    choice = st.radio(
        "Select Option",
        ["Login", "Create Account"]
    )

    data = load_users()

    # --------------------------------------------------------
    # LOGIN
    # --------------------------------------------------------

    if choice == "Login":

        username = st.text_input("Username")
        password = st.text_input(
            "Password",
            type="password"
        )

        if st.button("Login"):

            if (
                username in data["users"]
                and data["users"][username] == password
            ):

                st.session_state.logged_in = True
                st.session_state.username = username

                data["count"] += 1

                save_users(data)

                st.session_state.user_count = data["count"]

                # ------------------------------------------------
                # LOAD USER CHATS
                # ------------------------------------------------

                user_chats = get_user_chats(username)

                if user_chats:

                    latest_chat = user_chats[0]

                    st.session_state.current_chat_id = latest_chat["id"]

                    st.session_state.messages = []

                    for msg in latest_chat["messages"]:

                        if msg["role"] == "user":
                            st.session_state.messages.append(
                                HumanMessage(
                                    content=msg["content"]
                                )
                            )

                        else:
                            st.session_state.messages.append(
                                AIMessage(
                                    content=msg["content"]
                                )
                            )

                    st.session_state.retriever = rebuild_retriever(
                        latest_chat.get("documents", [])
                    )

                else:

                    chat_id = create_new_chat(username)

                    st.session_state.current_chat_id = chat_id
                    st.session_state.messages = []
                    st.session_state.retriever = None

                st.rerun()

            else:

                st.error("Invalid credentials")

    # --------------------------------------------------------
    # CREATE ACCOUNT
    # --------------------------------------------------------

    if choice == "Create Account":

        new_user = st.text_input("New Username")
        new_pass = st.text_input(
            "New Password",
            type="password"
        )

        if st.button("Create Account"):

            if new_user in data["users"]:

                st.warning("Username exists")

            elif new_user == "" or new_pass == "":

                st.warning("Fill all fields")

            else:

                data["users"][new_user] = new_pass

                save_users(data)

                st.success(
                    "Account created. Please login."
                )


# ============================================================
# PROTECT APP
# ============================================================

if not st.session_state.logged_in:

    auth_page()

    st.stop()


# ============================================================
# MAIN UI
# ============================================================

st.title("📄 AI Document Intelligence System")


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.success(
    f"👤 {st.session_state.username}"
)

st.sidebar.success(
    f"👥 Users: {st.session_state.user_count}"
)

st.sidebar.divider()

st.sidebar.subheader("💬 Chat History")


# ============================================================
# NEW CHAT BUTTON
# ============================================================

if st.sidebar.button(
    "➕ New Chat",
    use_container_width=True
):

    new_chat_id = create_new_chat(
        st.session_state.username
    )

    st.session_state.current_chat_id = new_chat_id
    st.session_state.messages = []
    st.session_state.retriever = None

    st.rerun()


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

user_chats = get_user_chats(
    st.session_state.username
)


for chat in user_chats:

    chat_title = chat.get(
        "title",
        "New Chat"
    )

    # Keep title short in sidebar
    if len(chat_title) > 30:
        chat_title = chat_title[:30] + "..."

    if st.sidebar.button(
        f"💬 {chat_title}",
        key=f"chat_{chat['id']}",
        use_container_width=True
    ):

        st.session_state.current_chat_id = chat["id"]

        st.session_state.messages = []

        # Load messages
        for msg in chat["messages"]:

            if msg["role"] == "user":

                st.session_state.messages.append(
                    HumanMessage(
                        content=msg["content"]
                    )
                )

            else:

                st.session_state.messages.append(
                    AIMessage(
                        content=msg["content"]
                    )
                )

        # Rebuild RAG retriever
        st.session_state.retriever = rebuild_retriever(
            chat.get("documents", [])
        )

        st.rerun()


# ============================================================
# DELETE CURRENT CHAT
# ============================================================

if st.session_state.current_chat_id:

    if st.sidebar.button(
        "🗑️ Delete Current Chat",
        use_container_width=True
    ):

        delete_chat(
            st.session_state.username,
            st.session_state.current_chat_id
        )

        remaining_chats = get_user_chats(
            st.session_state.username
        )

        if remaining_chats:

            latest_chat = remaining_chats[0]

            st.session_state.current_chat_id = latest_chat["id"]

            st.session_state.messages = []

            for msg in latest_chat["messages"]:

                if msg["role"] == "user":

                    st.session_state.messages.append(
                        HumanMessage(
                            content=msg["content"]
                        )
                    )

                else:

                    st.session_state.messages.append(
                        AIMessage(
                            content=msg["content"]
                        )
                    )

            st.session_state.retriever = rebuild_retriever(
                latest_chat.get("documents", [])
            )

        else:

            new_chat_id = create_new_chat(
                st.session_state.username
            )

            st.session_state.current_chat_id = new_chat_id
            st.session_state.messages = []
            st.session_state.retriever = None

        st.rerun()


st.sidebar.divider()


# ============================================================
# LOGOUT
# ============================================================

if st.sidebar.button(
    "🚪 Logout",
    use_container_width=True
):

    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.messages = []
    st.session_state.retriever = None
    st.session_state.current_chat_id = None

    st.rerun()


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
            f"💬 {current_chat['title']}"
        )


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_files = st.file_uploader(
    "Upload your files",
    type=["pdf", "txt", "docx", "xlsx"],
    accept_multiple_files=True
)


# ============================================================
# PROCESS FILES
# ============================================================

if uploaded_files:

    # Only process if current chat doesn't already have documents
    current_chat = get_chat(
        st.session_state.username,
        st.session_state.current_chat_id
    )

    existing_documents = current_chat.get(
        "documents",
        []
    ) if current_chat else []

    if not existing_documents:

        all_docs = []

        with st.spinner(
            "Processing files..."
        ):

            for uploaded_file in uploaded_files:

                file_type = uploaded_file.name.split(
                    "."
                )[-1].lower()

                if file_type not in [
                    "pdf",
                    "txt",
                    "docx",
                    "xlsx"
                ]:
                    continue

                # ----------------------------------------------
                # PDF
                # ----------------------------------------------

                if file_type == "pdf":

                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=".pdf"
                    ) as tmp:

                        tmp.write(
                            uploaded_file.read()
                        )

                        pdf_path = tmp.name

                    loader = PyPDFLoader(
                        pdf_path
                    )

                    docs = loader.load()

                # ----------------------------------------------
                # TXT
                # ----------------------------------------------

                elif file_type == "txt":

                    text = uploaded_file.read().decode(
                        "utf-8"
                    )

                    docs = [
                        Document(
                            page_content=text
                        )
                    ]

                # ----------------------------------------------
                # DOCX
                # ----------------------------------------------

                elif file_type == "docx":

                    from docx import Document as DocxDocument

                    doc = DocxDocument(
                        uploaded_file
                    )

                    text = "\n".join(
                        [
                            p.text
                            for p in doc.paragraphs
                        ]
                    )

                    docs = [
                        Document(
                            page_content=text
                        )
                    ]

                # ----------------------------------------------
                # XLSX
                # ----------------------------------------------

                elif file_type == "xlsx":

                    import pandas as pd

                    df = pd.read_excel(
                        uploaded_file
                    )

                    text = df.to_string()

                    docs = [
                        Document(
                            page_content=text
                        )
                    ]

                # ----------------------------------------------
                # METADATA
                # ----------------------------------------------

                for doc in docs:

                    doc.metadata[
                        "source"
                    ] = uploaded_file.name

                all_docs.extend(docs)

            # ----------------------------------------------
            # SPLIT DOCUMENTS
            # ----------------------------------------------

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=100
            )

            chunks = splitter.split_documents(
                all_docs
            )

            # ----------------------------------------------
            # CREATE VECTORSTORE
            # ----------------------------------------------

            vectorstore = FAISS.from_documents(
                chunks,
                st.session_state.embeddings
            )

            st.session_state.retriever = (
                vectorstore.as_retriever(
                    search_kwargs={"k": 2}
                )
            )

            # ----------------------------------------------
            # SAVE ORIGINAL DOCUMENTS
            # ----------------------------------------------

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
                "Files processed successfully!"
            )


# ============================================================
# LLM
# ============================================================

@st.cache_resource
def load_llm():

    pipe = pipeline(
        "text-generation",
        model="gpt2",
        max_new_tokens=150,
        do_sample=False,
        pad_token_id=50256
    )

    return HuggingFacePipeline(
        pipeline=pipe
    )


llm = load_llm()


# ============================================================
# PROMPT
# ============================================================

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer only using the given context."
        ),
        (
            "human",
            "{question}\n\nContext:\n{context}"
        )
    ]
)


# ============================================================
# CHAT UI
# ============================================================

st.divider()

st.subheader("💬 Chat")


# ============================================================
# DISPLAY PREVIOUS MESSAGES
# ============================================================

for msg in st.session_state.messages:

    role = (
        "user"
        if isinstance(msg, HumanMessage)
        else "assistant"
    )

    with st.chat_message(role):

        st.write(
            msg.content
        )


# ============================================================
# CHAT INPUT
# ============================================================

if st.session_state.retriever:

    question = st.chat_input(
        "Ask your question..."
    )

    if question:

        # ----------------------------------------------------
        # DISPLAY USER QUESTION
        # ----------------------------------------------------

        with st.chat_message("user"):

            st.write(question)

        # ----------------------------------------------------
        # GENERATE ANSWER
        # ----------------------------------------------------

        with st.spinner(
            "Thinking..."
        ):

            docs = st.session_state.retriever.invoke(
                question
            )

            context = "\n\n".join(
                [
                    doc.page_content
                    for doc in docs
                ]
            )[:1500]

            response = llm.invoke(
                prompt.format_messages(
                    question=question,
                    context=context
                )
            )

            answer = response

        # ----------------------------------------------------
        # DISPLAY ANSWER
        # ----------------------------------------------------

        with st.chat_message("assistant"):

            st.write(answer)

        # ----------------------------------------------------
        # SAVE TO SESSION
        # ----------------------------------------------------

        st.session_state.messages.append(
            HumanMessage(
                content=question
            )
        )

        st.session_state.messages.append(
            AIMessage(
                content=answer
            )
        )

        # ====================================================
        # SAVE TO DATABASE/JSON
        # ====================================================

        saved_messages = []

        for msg in st.session_state.messages:

            if isinstance(
                msg,
                HumanMessage
            ):

                saved_messages.append(
                    {
                        "role": "user",
                        "content": msg.content
                    }
                )

            else:

                saved_messages.append(
                    {
                        "role": "assistant",
                        "content": msg.content
                    }
                )

        # ----------------------------------------------------
        # CREATE CHAT TITLE
        # ----------------------------------------------------

        current_chat = get_chat(
            st.session_state.username,
            st.session_state.current_chat_id
        )

        if (
            current_chat
            and current_chat["title"] == "New Chat"
        ):

            title = question.strip()

            if len(title) > 45:

                title = title[:45] + "..."

        else:

            title = None

        # ----------------------------------------------------
        # SAVE EVERYTHING
        # ----------------------------------------------------

        update_chat(
            st.session_state.username,
            st.session_state.current_chat_id,
            messages=saved_messages,
            title=title
        )

        # ----------------------------------------------------
        # RERUN
        # ----------------------------------------------------

        st.rerun()

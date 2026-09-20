# ============================================================
# AI DOCUMENT INTELLIGENCE
# ChatGPT-style Recent Chats + Separate Knowledge per Chat
# ============================================================

import os
import re
import json
import uuid
import hashlib
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from urllib.parse import urlparse

import streamlit as st
import requests
import pandas as pd

from PIL import Image

from pypdf import PdfReader
from docx import Document

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from google import genai
from google.genai import types


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Document Intelligence",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CONSTANTS
# ============================================================

APP_NAME = "AI Document Intelligence"

# Gemini model
# You can change this from Streamlit Secrets using:
# GEMINI_MODEL = "gemini-3.8-flash"
GEMINI_MODEL = st.secrets.get(
    "GEMINI_MODEL",
    os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
)

MAX_OUTPUT_TOKENS = 4096

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150

RETRIEVER_K = 5

BASE_DIR = Path(".")
USERS_FILE = BASE_DIR / "users.json"
CHATS_FILE = BASE_DIR / "chats.json"

CHAT_STORAGE = BASE_DIR / "chat_storage"

CHAT_STORAGE.mkdir(exist_ok=True)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

    /* ------------------------------
       MAIN APP
    ------------------------------ */

    .stApp {
        background: #191919;
        color: #f5f5f5;
    }

    .main .block-container {
        max-width: 1200px;
        padding-top: 1.5rem;
        padding-bottom: 7rem;
    }


    /* ------------------------------
       SIDEBAR
    ------------------------------ */

    section[data-testid="stSidebar"] {
        background: #111111;
        border-right: 1px solid #2d2d2d;
    }

    section[data-testid="stSidebar"] * {
        color: #f5f5f5;
    }

    section[data-testid="stSidebar"] button {
        border-radius: 8px;
    }


    /* ------------------------------
       CHAT TITLE
    ------------------------------ */

    .app-title {
        font-size: 30px;
        font-weight: 700;
        margin-bottom: 3px;
    }

    .app-subtitle {
        color: #a8a8a8;
        font-size: 14px;
        margin-bottom: 20px;
    }


    /* ------------------------------
       RECENT CHAT
    ------------------------------ */

    .recent-heading {
        color: #8f8f8f;
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        margin-top: 18px;
        margin-bottom: 6px;
        letter-spacing: 0.6px;
    }


    /* ------------------------------
       CHAT BUBBLES
    ------------------------------ */

    [data-testid="stChatMessage"] {
        border-radius: 12px;
    }

    [data-testid="stChatMessageContent"] {
        font-size: 15px;
        line-height: 1.65;
    }


    /* ------------------------------
       INPUT
    ------------------------------ */

    [data-testid="stChatInput"] {
        background: #242424;
    }


    /* ------------------------------
       CARDS
    ------------------------------ */

    .info-card {
        background: #222222;
        border: 1px solid #343434;
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 12px;
    }

    .small-muted {
        color: #999999;
        font-size: 12px;
    }


    /* ------------------------------
       FILE ITEMS
    ------------------------------ */

    .file-item {
        background: #202020;
        border: 1px solid #303030;
        border-radius: 8px;
        padding: 8px 10px;
        margin-bottom: 6px;
        font-size: 13px;
    }


    /* ------------------------------
       DIVIDER
    ------------------------------ */

    .divider {
        height: 1px;
        background: #303030;
        margin: 12px 0;
    }

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def now_iso():
    return datetime.now().isoformat()


def safe_username(username):
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", username)


def safe_filename(filename):
    filename = Path(filename).name
    return re.sub(r"[^a-zA-Z0-9_. -]", "_", filename)


def file_hash(data):
    return hashlib.sha256(data).hexdigest()


def get_chat_folder(username, chat_id):
    return (
        CHAT_STORAGE
        / safe_username(username)
        / chat_id
    )


def get_documents_folder(username, chat_id):
    folder = get_chat_folder(username, chat_id) / "documents"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def get_images_folder(username, chat_id):
    folder = get_chat_folder(username, chat_id) / "images"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def get_knowledge_file(username, chat_id):
    folder = get_chat_folder(username, chat_id)
    folder.mkdir(parents=True, exist_ok=True)
    return folder / "knowledge.json"


# ============================================================
# JSON STORAGE
# ============================================================

def load_json(path, default):
    try:
        if not path.exists():
            return default

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    except Exception:
        return default


def save_json(path, data):
    temp_path = path.with_suffix(".tmp")

    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False
        )

    temp_path.replace(path)


# ============================================================
# USERS
# ============================================================

def load_users():
    return load_json(USERS_FILE, {})


def save_users(users):
    save_json(USERS_FILE, users)


def create_user(username, password):
    users = load_users()

    if username in users:
        return False, "Username already exists."

    users[username] = {
        "password": password,
        "created_at": now_iso()
    }

    save_users(users)

    return True, "Account created successfully."


def login_user(username, password):
    users = load_users()

    if username not in users:
        return False

    if users[username]["password"] != password:
        return False

    return True


# ============================================================
# CHAT STORAGE
# ============================================================

def load_chats():
    return load_json(CHATS_FILE, {})


def save_chats(chats):
    save_json(CHATS_FILE, chats)


def create_chat(username, title="New Chat"):
    chats = load_chats()

    chat_id = uuid.uuid4().hex

    if username not in chats:
        chats[username] = {}

    chats[username][chat_id] = {
        "id": chat_id,
        "title": title,
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "messages": [],
        "documents": [],
        "images": [],
        "website_urls": []
    }

    save_chats(chats)

    get_chat_folder(username, chat_id).mkdir(
        parents=True,
        exist_ok=True
    )

    get_documents_folder(username, chat_id)
    get_images_folder(username, chat_id)

    return chat_id


def get_user_chats(username):
    chats = load_chats()

    user_chats = chats.get(username, {})

    return user_chats


def get_chat(username, chat_id):
    chats = load_chats()

    return (
        chats
        .get(username, {})
        .get(chat_id)
    )


def update_chat(username, chat_id, chat_data):
    chats = load_chats()

    if username not in chats:
        chats[username] = {}

    chats[username][chat_id] = chat_data

    save_chats(chats)


def delete_chat(username, chat_id):
    chats = load_chats()

    if username in chats:
        chats[username].pop(chat_id, None)

    save_chats(chats)

    folder = get_chat_folder(username, chat_id)

    if folder.exists():
        shutil.rmtree(folder, ignore_errors=True)


def rename_chat(username, chat_id, new_title):
    chat = get_chat(username, chat_id)

    if not chat:
        return

    chat["title"] = new_title.strip()[:80]
    chat["updated_at"] = now_iso()

    update_chat(
        username,
        chat_id,
        chat
    )


# ============================================================
# CHAT TITLE
# ============================================================

def generate_chat_title(question):
    question = question.strip()

    if not question:
        return "New Chat"

    clean = re.sub(
        r"\s+",
        " ",
        question
    )

    clean = clean.replace("\n", " ")

    if len(clean) > 45:
        clean = clean[:45].rstrip() + "..."

    return clean


# ============================================================
# RECENT CHAT GROUPING
# ============================================================

def parse_datetime(value):
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return datetime.now()


def group_chats_by_date(chats):
    now = datetime.now()

    today = now.date()
    yesterday = today - timedelta(days=1)
    seven_days_ago = today - timedelta(days=7)

    groups = {
        "Today": [],
        "Yesterday": [],
        "Previous 7 days": [],
        "Older": []
    }

    for chat in chats:
        dt = parse_datetime(
            chat.get(
                "updated_at",
                chat.get("created_at", now_iso())
            )
        )

        date_value = dt.date()

        if date_value == today:
            groups["Today"].append(chat)

        elif date_value == yesterday:
            groups["Yesterday"].append(chat)

        elif date_value >= seven_days_ago:
            groups["Previous 7 days"].append(chat)

        else:
            groups["Older"].append(chat)

    for group in groups:
        groups[group].sort(
            key=lambda x: x.get(
                "updated_at",
                ""
            ),
            reverse=True
        )

    return groups


# ============================================================
# TEXT SPLITTER
# ============================================================

def split_text(
    text,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP
):

    if not text:
        return []

    text = text.strip()

    if len(text) <= chunk_size:
        return [text]

    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:

        end = start + chunk_size

        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk.strip())

        next_start = end - overlap

        if next_start <= start:
            next_start = end

        start = next_start

    return chunks


# ============================================================
# DOCUMENT EXTRACTION
# ============================================================

def extract_pdf(file_path):

    text_parts = []

    reader = PdfReader(str(file_path))

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        try:
            page_text = page.extract_text() or ""

            if page_text.strip():
                text_parts.append(
                    f"[Page {page_number}]\n{page_text}"
                )

        except Exception:
            continue

    return "\n\n".join(text_parts)


def extract_docx(file_path):

    document = Document(str(file_path))

    parts = []

    for paragraph in document.paragraphs:

        text = paragraph.text.strip()

        if text:
            parts.append(text)

    return "\n".join(parts)


def extract_xlsx(file_path):

    parts = []

    excel_file = pd.ExcelFile(file_path)

    for sheet_name in excel_file.sheet_names:

        df = pd.read_excel(
            file_path,
            sheet_name=sheet_name
        )

        parts.append(
            f"Sheet: {sheet_name}\n"
            + df.to_string(index=False)
        )

    return "\n\n".join(parts)


def extract_txt(file_path):

    with open(
        file_path,
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as f:

        return f.read()


def extract_document(file_path):

    suffix = Path(file_path).suffix.lower()

    if suffix == ".pdf":
        return extract_pdf(file_path)

    if suffix == ".docx":
        return extract_docx(file_path)

    if suffix in [".xlsx", ".xls"]:
        return extract_xlsx(file_path)

    if suffix in [".txt", ".md", ".csv"]:
        return extract_txt(file_path)

    return ""


# ============================================================
# WEBSITE EXTRACTION
# ============================================================

def extract_website(url):

    try:

        response = requests.get(
            url,
            timeout=15,
            headers={
                "User-Agent":
                "Mozilla/5.0"
            }
        )

        response.raise_for_status()

        html = response.text

        # Remove scripts and styles
        html = re.sub(
            r"<script.*?</script>",
            " ",
            html,
            flags=re.DOTALL | re.IGNORECASE
        )

        html = re.sub(
            r"<style.*?</style>",
            " ",
            html,
            flags=re.DOTALL | re.IGNORECASE
        )

        text = re.sub(
            r"<[^>]+>",
            " ",
            html
        )

        text = re.sub(
            r"\s+",
            " ",
            text
        )

        return text.strip()

    except Exception as e:
        raise Exception(
            f"Could not load website: {e}"
        )


# ============================================================
# EMBEDDINGS
# ============================================================

@st.cache_resource
def get_embeddings():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# ============================================================
# KNOWLEDGE MANIFEST
# ============================================================

def load_knowledge_manifest(
    username,
    chat_id
):

    knowledge_file = get_knowledge_file(
        username,
        chat_id
    )

    return load_json(
        knowledge_file,
        {
            "documents": [],
            "websites": []
        }
    )


def save_knowledge_manifest(
    username,
    chat_id,
    manifest
):

    knowledge_file = get_knowledge_file(
        username,
        chat_id
    )

    save_json(
        knowledge_file,
        manifest
    )


# ============================================================
# BUILD CHAT VECTORSTORE
# ============================================================

def build_chat_vectorstore(
    username,
    chat_id
):

    manifest = load_knowledge_manifest(
        username,
        chat_id
    )

    documents = []

    # ------------------------------
    # DOCUMENTS
    # ------------------------------

    for item in manifest.get(
        "documents",
        []
    ):

        path = Path(
            item.get("path", "")
        )

        if not path.exists():
            continue

        try:

            text = extract_document(path)

            if not text.strip():
                continue

            chunks = split_text(text)

            for index, chunk in enumerate(chunks):

                documents.append(
                    {
                        "text": chunk,
                        "source": item.get(
                            "name",
                            path.name
                        ),
                        "type": "document",
                        "chunk": index
                    }
                )

        except Exception:
            continue


    # ------------------------------
    # WEBSITES
    # ------------------------------

    for item in manifest.get(
        "websites",
        []
    ):

        text = item.get("text", "")

        if not text.strip():
            continue

        chunks = split_text(text)

        for index, chunk in enumerate(chunks):

            documents.append(
                {
                    "text": chunk,
                    "source": item.get(
                        "url",
                        "Website"
                    ),
                    "type": "website",
                    "chunk": index
                }
            )


    # ------------------------------
    # NOTHING FOUND
    # ------------------------------

    if not documents:
        return None


    # ------------------------------
    # CREATE VECTORSTORE
    # ------------------------------

    texts = [
        item["text"]
        for item in documents
    ]

    metadatas = [
        {
            "source": item["source"],
            "type": item["type"],
            "chunk": item["chunk"]
        }
        for item in documents
    ]

    embeddings = get_embeddings()

    vectorstore = FAISS.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas
    )

    return vectorstore


# ============================================================
# LOAD CURRENT CHAT KNOWLEDGE
# ============================================================

def load_current_chat_knowledge():

    username = st.session_state.get(
        "username"
    )

    chat_id = st.session_state.get(
        "chat_id"
    )

    if not username or not chat_id:
        return

    try:

        vectorstore = build_chat_vectorstore(
            username,
            chat_id
        )

        st.session_state.active_vectorstore = (
            vectorstore
        )

    except Exception as e:

        st.session_state.active_vectorstore = None

        st.warning(
            f"Could not build document knowledge: {e}"
        )


# ============================================================
# SEARCH CURRENT CHAT DOCUMENTS
# ============================================================

def get_document_context(question):

    vectorstore = st.session_state.get(
        "active_vectorstore"
    )

    if vectorstore is None:
        return "", []

    try:

        results = vectorstore.similarity_search(
            question,
            k=RETRIEVER_K
        )

    except Exception:
        return "", []


    context_parts = []
    sources = []

    for result in results:

        text = result.page_content

        metadata = result.metadata or {}

        source = metadata.get(
            "source",
            "Unknown"
        )

        context_parts.append(
            f"SOURCE: {source}\n{text}"
        )

        if source not in sources:
            sources.append(source)

    context = "\n\n---\n\n".join(
        context_parts
    )

    return context, sources


# ============================================================
# LOAD CURRENT CHAT IMAGES
# ============================================================

def load_current_chat_images():

    username = st.session_state.get(
        "username"
    )

    chat_id = st.session_state.get(
        "chat_id"
    )

    if not username or not chat_id:
        return []

    images_folder = get_images_folder(
        username,
        chat_id
    )

    image_files = []

    for path in images_folder.iterdir():

        if path.suffix.lower() in [
            ".png",
            ".jpg",
            ".jpeg",
            ".webp"
        ]:

            image_files.append(path)

    return image_files


# ============================================================
# GEMINI CLIENT
# ============================================================

def get_gemini_client():

    api_key = st.secrets.get(
        "GEMINI_API_KEY",
        os.getenv("GEMINI_API_KEY", "")
    )

    if not api_key:
        return None

    return genai.Client(
        api_key=api_key
    )


# ============================================================
# GENERATE GEMINI RESPONSE
# ============================================================

def generate_ai_response(
    question,
    chat_history,
    document_context,
    sources,
    image_paths
):

    client = get_gemini_client()

    if client is None:

        return (
            "⚠️ Gemini API key is not configured.\n\n"
            "Add `GEMINI_API_KEY` to your Streamlit Secrets."
        )


    # ========================================================
    # SYSTEM INSTRUCTIONS
    # ========================================================

    system_instruction = """
You are the AI assistant inside an AI Document Intelligence application.

IMPORTANT CHAT ISOLATION RULES:

1. Every conversation is completely separate.
2. You may use ONLY the document context supplied for the CURRENT conversation.
3. NEVER assume that a document from another conversation is available.
4. NEVER mix information from different conversations.
5. If the user asks about the current conversation's documents,
   answer using the supplied document context.
6. If the answer cannot be found in the supplied current-chat
   documents, clearly say:
   "I couldn't find that information in the documents uploaded
   to this chat."
7. Multiple documents inside the CURRENT conversation may be
   compared with each other.
8. Website information supplied in the current conversation
   belongs only to this conversation.
9. Images supplied in the current conversation belong only to
   this conversation.
10. If the question is general knowledge and does not require
    uploaded documents, answer normally.
11. Do not invent information from documents.
12. Be clear, direct and helpful.
13. Use simple language unless the user asks for technical detail.
"""


    # ========================================================
    # BUILD HISTORY
    # ========================================================

    history_text = ""

    for message in chat_history[-10:]:

        role = message.get(
            "role",
            ""
        )

        content = message.get(
            "content",
            ""
        )

        if role == "user":
            history_text += (
                f"User: {content}\n"
            )

        elif role == "assistant":
            history_text += (
                f"Assistant: {content}\n"
            )


    # ========================================================
    # DOCUMENT CONTEXT
    # ========================================================

    if document_context.strip():

        document_section = f"""
CURRENT CHAT DOCUMENT CONTEXT:

{document_context}

END CURRENT CHAT DOCUMENT CONTEXT.
"""

    else:

        document_section = """
CURRENT CHAT DOCUMENT CONTEXT:

No relevant document context was found.
"""


    # ========================================================
    # SOURCES
    # ========================================================

    source_text = ""

    if sources:

        source_text = (
            "\nRelevant sources in this chat:\n"
            + "\n".join(
                f"- {source}"
                for source in sources
            )
        )


    # ========================================================
    # FINAL PROMPT
    # ========================================================

    prompt = f"""
{system_instruction}

{document_section}

{source_text}

PREVIOUS CONVERSATION:
{history_text}

CURRENT USER QUESTION:
{question}

Answer the current user question.
"""


    # ========================================================
    # CONTENTS
    # ========================================================

    contents = [prompt]


    # ========================================================
    # ADD CURRENT CHAT IMAGES ONLY
    # ========================================================

    for image_path in image_paths:

        try:

            image = Image.open(
                image_path
            )

            contents.append(image)

        except Exception:
            continue


    # ========================================================
    # GEMINI REQUEST
    # ========================================================

    try:

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.2,
                max_output_tokens=MAX_OUTPUT_TOKENS
            )
        )

        answer = response.text

        if not answer:
            return "I couldn't generate a response."

        return answer.strip()

    except Exception as e:

        error_text = str(e)

        return (
            "⚠️ Gemini error:\n\n"
            f"{error_text}\n\n"
            "If this says the model is unavailable, "
            "change GEMINI_MODEL in Streamlit Secrets "
            "to a model available to your Gemini API key."
        )


# ============================================================
# SAVE CHAT MESSAGE
# ============================================================

def save_message(
    username,
    chat_id,
    role,
    content
):

    chat = get_chat(
        username,
        chat_id
    )

    if not chat:
        return

    chat["messages"].append(
        {
            "role": role,
            "content": content,
            "timestamp": now_iso()
        }
    )

    chat["updated_at"] = now_iso()

    update_chat(
        username,
        chat_id,
        chat
    )


# ============================================================
# LOGIN PAGE
# ============================================================

def login_page():

    st.markdown(
        """
        <div style="
            max-width:520px;
            margin:80px auto 0 auto;
            text-align:center;
        ">
            <div style="font-size:55px;">🤖</div>
            <div style="
                font-size:32px;
                font-weight:700;
                margin-top:10px;
            ">
                AI Document Intelligence
            </div>
            <div style="
                color:#999;
                margin-top:8px;
            ">
                Chat with your documents using AI
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    login_tab, signup_tab = st.tabs(
        ["🔐 Login", "➕ Create Account"]
    )


    # ========================================================
    # LOGIN
    # ========================================================

    with login_tab:

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

            if login_user(
                username,
                password
            ):

                st.session_state.logged_in = True
                st.session_state.username = username

                user_chats = get_user_chats(
                    username
                )

                if user_chats:

                    sorted_chats = sorted(
                        user_chats.values(),
                        key=lambda x: x.get(
                            "updated_at",
                            ""
                        ),
                        reverse=True
                    )

                    st.session_state.chat_id = (
                        sorted_chats[0]["id"]
                    )

                else:

                    st.session_state.chat_id = create_chat(
                        username
                    )

                load_current_chat_knowledge()

                st.rerun()

            else:

                st.error(
                    "Invalid username or password."
                )


    # ========================================================
    # SIGNUP
    # ========================================================

    with signup_tab:

        new_username = st.text_input(
            "Choose username",
            key="signup_username"
        )

        new_password = st.text_input(
            "Choose password",
            type="password",
            key="signup_password"
        )

        confirm_password = st.text_input(
            "Confirm password",
            type="password",
            key="signup_confirm"
        )

        if st.button(
            "Create Account",
            use_container_width=True
        ):

            if not new_username.strip():

                st.error(
                    "Please enter a username."
                )

            elif not new_password:

                st.error(
                    "Please enter a password."
                )

            elif new_password != confirm_password:

                st.error(
                    "Passwords do not match."
                )

            else:

                success, message = create_user(
                    new_username.strip(),
                    new_password
                )

                if success:

                    chat_id = create_chat(
                        new_username.strip()
                    )

                    st.session_state.logged_in = True
                    st.session_state.username = (
                        new_username.strip()
                    )
                    st.session_state.chat_id = chat_id

                    load_current_chat_knowledge()

                    st.success(message)

                    st.rerun()

                else:

                    st.error(message)


# ============================================================
# INITIALIZE SESSION STATE
# ============================================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = None

if "chat_id" not in st.session_state:
    st.session_state.chat_id = None

if "active_vectorstore" not in st.session_state:
    st.session_state.active_vectorstore = None

if "upload_version" not in st.session_state:
    st.session_state.upload_version = 0


# ============================================================
# SHOW LOGIN
# ============================================================

if not st.session_state.logged_in:

    login_page()

    st.stop()


# ============================================================
# CURRENT USER
# ============================================================

username = st.session_state.username


# ============================================================
# MAKE SURE USER HAS A CHAT
# ============================================================

user_chats = get_user_chats(username)

if not user_chats:

    new_chat_id = create_chat(
        username
    )

    st.session_state.chat_id = new_chat_id

    load_current_chat_knowledge()

    st.rerun()


# ============================================================
# MAKE SURE CURRENT CHAT EXISTS
# ============================================================

if (
    st.session_state.chat_id not in user_chats
):

    latest_chat = sorted(
        user_chats.values(),
        key=lambda x: x.get(
            "updated_at",
            ""
        ),
        reverse=True
    )[0]

    st.session_state.chat_id = latest_chat["id"]

    load_current_chat_knowledge()

    st.rerun()


# ============================================================
# CURRENT CHAT
# ============================================================

current_chat = get_chat(
    username,
    st.session_state.chat_id
)

if current_chat is None:

    st.error(
        "Current chat could not be loaded."
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="
            font-size:22px;
            font-weight:700;
            margin-bottom:12px;
        ">
            🤖 AI Document Intelligence
        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # NEW CHAT
    # ========================================================

    if st.button(
        "➕  New Chat",
        use_container_width=True
    ):

        new_chat_id = create_chat(
            username
        )

        st.session_state.chat_id = new_chat_id

        st.session_state.active_vectorstore = None

        st.session_state.upload_version += 1

        st.rerun()


    st.markdown(
        '<div class="divider"></div>',
        unsafe_allow_html=True
    )


    # ========================================================
    # RECENTS
    # ========================================================

    st.markdown(
        "### 🕘 Recents"
    )

    all_chats = list(
        get_user_chats(username).values()
    )

    grouped_chats = group_chats_by_date(
        all_chats
    )


    for group_name, chats_in_group in grouped_chats.items():

        if not chats_in_group:
            continue

        st.markdown(
            f'<div class="recent-heading">{group_name}</div>',
            unsafe_allow_html=True
        )

        for chat in chats_in_group:

            chat_id = chat["id"]

            title = chat.get(
                "title",
                "New Chat"
            )

            if len(title) > 35:
                title = title[:35] + "..."

            is_current = (
                chat_id ==
                st.session_state.chat_id
            )

            button_text = (
                "🟢  " if is_current else "💬  "
            ) + title

            if st.button(
                button_text,
                key=f"open_{chat_id}",
                use_container_width=True
            ):

                st.session_state.chat_id = chat_id

                st.session_state.active_vectorstore = None

                st.session_state.upload_version += 1

                load_current_chat_knowledge()

                st.rerun()


    st.markdown(
        '<div class="divider"></div>',
        unsafe_allow_html=True
    )


    # ========================================================
    # CHAT MANAGEMENT
    # ========================================================

    with st.expander(
        "⚙️ Chat Settings"
    ):

        new_title = st.text_input(
            "Rename current chat",
            value=current_chat.get(
                "title",
                "New Chat"
            ),
            key="rename_chat"
        )

        if st.button(
            "Save Chat Name",
            use_container_width=True
        ):

            if new_title.strip():

                rename_chat(
                    username,
                    st.session_state.chat_id,
                    new_title
                )

                st.rerun()


        if st.button(
            "🗑️ Delete Current Chat",
            use_container_width=True
        ):

            current_id = (
                st.session_state.chat_id
            )

            delete_chat(
                username,
                current_id
            )

            remaining = list(
                get_user_chats(username).values()
            )

            if remaining:

                remaining.sort(
                    key=lambda x: x.get(
                        "updated_at",
                        ""
                    ),
                    reverse=True
                )

                st.session_state.chat_id = (
                    remaining[0]["id"]
                )

            else:

                st.session_state.chat_id = create_chat(
                    username
                )

            st.session_state.active_vectorstore = None

            st.session_state.upload_version += 1

            load_current_chat_knowledge()

            st.rerun()


    # ========================================================
    # DOCUMENT UPLOAD
    # ========================================================

    st.markdown(
        "### 📄 Documents"
    )

    st.caption(
        "Documents uploaded here belong only to this chat."
    )

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=[
            "pdf",
            "docx",
            "txt",
            "md",
            "csv",
            "xlsx",
            "xls"
        ],
        accept_multiple_files=True,
        key=(
            f"documents_"
            f"{st.session_state.chat_id}_"
            f"{st.session_state.upload_version}"
        )
    )


    if uploaded_files:

        manifest = load_knowledge_manifest(
            username,
            st.session_state.chat_id
        )

        existing_hashes = {
            item.get("hash")
            for item in manifest.get(
                "documents",
                []
            )
        }

        new_document_added = False


        for uploaded_file in uploaded_files:

            data = uploaded_file.getvalue()

            current_hash = file_hash(data)

            if current_hash in existing_hashes:
                continue

            filename = safe_filename(
                uploaded_file.name
            )

            save_path = (
                get_documents_folder(
                    username,
                    st.session_state.chat_id
                )
                / filename
            )

            # Avoid same filename conflict
            if save_path.exists():

                stem = save_path.stem
                suffix = save_path.suffix

                save_path = (
                    save_path.parent
                    / f"{stem}_{current_hash[:8]}{suffix}"
                )


            with open(
                save_path,
                "wb"
            ) as f:

                f.write(data)


            manifest.setdefault(
                "documents",
                []
            ).append(
                {
                    "name": save_path.name,
                    "path": str(save_path),
                    "hash": current_hash,
                    "uploaded_at": now_iso()
                }
            )

            existing_hashes.add(
                current_hash
            )

            current_chat.setdefault(
                "documents",
                []
            ).append(
                save_path.name
            )

            new_document_added = True


        if new_document_added:

            save_knowledge_manifest(
                username,
                st.session_state.chat_id,
                manifest
            )

            current_chat["updated_at"] = now_iso()

            update_chat(
                username,
                st.session_state.chat_id,
                current_chat
            )

            load_current_chat_knowledge()

            st.success(
                "Document added to this chat."
            )

            st.rerun()


    # ========================================================
    # SHOW DOCUMENTS
    # ========================================================

    manifest = load_knowledge_manifest(
        username,
        st.session_state.chat_id
    )

    documents = manifest.get(
        "documents",
        []
    )

    if documents:

        st.caption(
            f"{len(documents)} document(s) in this chat"
        )

        for item in documents:

            st.markdown(
                f"""
                <div class="file-item">
                    📄 {item.get("name", "Document")}
                </div>
                """,
                unsafe_allow_html=True
            )


    # ========================================================
    # IMAGE UPLOAD
    # ========================================================

    st.markdown(
        "### 🖼️ Images"
    )

    uploaded_images = st.file_uploader(
        "Upload images",
        type=[
            "png",
            "jpg",
            "jpeg",
            "webp"
        ],
        accept_multiple_files=True,
        key=(
            f"images_"
            f"{st.session_state.chat_id}_"
            f"{st.session_state.upload_version}"
        )
    )


    if uploaded_images:

        current_chat = get_chat(
            username,
            st.session_state.chat_id
        )

        existing_image_hashes = set(
            current_chat.get(
                "image_hashes",
                []
            )
        )

        new_image_added = False


        for uploaded_image in uploaded_images:

            data = uploaded_image.getvalue()

            current_hash = file_hash(data)

            if current_hash in existing_image_hashes:
                continue

            filename = safe_filename(
                uploaded_image.name
            )

            save_path = (
                get_images_folder(
                    username,
                    st.session_state.chat_id
                )
                / filename
            )


            if save_path.exists():

                stem = save_path.stem
                suffix = save_path.suffix

                save_path = (
                    save_path.parent
                    / f"{stem}_{current_hash[:8]}{suffix}"
                )


            with open(
                save_path,
                "wb"
            ) as f:

                f.write(data)


            current_chat.setdefault(
                "images",
                []
            ).append(
                save_path.name
            )

            current_chat.setdefault(
                "image_hashes",
                []
            ).append(
                current_hash
            )

            existing_image_hashes.add(
                current_hash
            )

            new_image_added = True


        if new_image_added:

            current_chat["updated_at"] = now_iso()

            update_chat(
                username,
                st.session_state.chat_id,
                current_chat
            )

            st.success(
                "Image added to this chat."
            )

            st.rerun()


    # ========================================================
    # SHOW IMAGES
    # ========================================================

    image_files = load_current_chat_images()

    if image_files:

        st.caption(
            f"{len(image_files)} image(s) in this chat"
        )

        for image_file in image_files:

            st.markdown(
                f"""
                <div class="file-item">
                    🖼️ {image_file.name}
                </div>
                """,
                unsafe_allow_html=True
            )


    # ========================================================
    # WEBSITE
    # ========================================================

    st.markdown(
        "### 🌐 Website"
    )

    website_url = st.text_input(
        "Add website URL",
        placeholder="https://example.com",
        key=f"website_{st.session_state.chat_id}"
    )

    if st.button(
        "Load Website",
        use_container_width=True
    ):

        if not website_url.strip():

            st.warning(
                "Enter a website URL."
            )

        else:

            if not (
                website_url.startswith("http://")
                or website_url.startswith("https://")
            ):

                website_url = (
                    "https://" + website_url
                )

            try:

                website_text = extract_website(
                    website_url
                )

                manifest = load_knowledge_manifest(
                    username,
                    st.session_state.chat_id
                )

                existing_urls = [
                    item.get("url")
                    for item in manifest.get(
                        "websites",
                        []
                    )
                ]

                if website_url in existing_urls:

                    st.warning(
                        "This website is already added to this chat."
                    )

                else:

                    manifest.setdefault(
                        "websites",
                        []
                    ).append(
                        {
                            "url": website_url,
                            "text": website_text,
                            "added_at": now_iso()
                        }
                    )

                    save_knowledge_manifest(
                        username,
                        st.session_state.chat_id,
                        manifest
                    )

                    current_chat = get_chat(
                        username,
                        st.session_state.chat_id
                    )

                    current_chat.setdefault(
                        "website_urls",
                        []
                    ).append(
                        website_url
                    )

                    current_chat["updated_at"] = now_iso()

                    update_chat(
                        username,
                        st.session_state.chat_id,
                        current_chat
                    )

                    load_current_chat_knowledge()

                    st.success(
                        "Website added to this chat."
                    )

                    st.rerun()

            except Exception as e:

                st.error(
                    f"Website error: {e}"
                )


    # ========================================================
    # CLEAR KNOWLEDGE
    # ========================================================

    if st.button(
        "🧹 Clear Knowledge",
        use_container_width=True
    ):

        chat_folder = get_chat_folder(
            username,
            st.session_state.chat_id
        )

        documents_folder = (
            chat_folder / "documents"
        )

        images_folder = (
            chat_folder / "images"
        )

        if documents_folder.exists():
            shutil.rmtree(
                documents_folder
            )

        if images_folder.exists():
            shutil.rmtree(
                images_folder
            )

        get_documents_folder(
            username,
            st.session_state.chat_id
        )

        get_images_folder(
            username,
            st.session_state.chat_id
        )

        save_knowledge_manifest(
            username,
            st.session_state.chat_id,
            {
                "documents": [],
                "websites": []
            }
        )

        current_chat = get_chat(
            username,
            st.session_state.chat_id
        )

        current_chat["documents"] = []
        current_chat["images"] = []
        current_chat["image_hashes"] = []
        current_chat["website_urls"] = []

        update_chat(
            username,
            st.session_state.chat_id,
            current_chat
        )

        st.session_state.active_vectorstore = None

        st.success(
            "Knowledge cleared from this chat."
        )

        st.rerun()


    st.markdown(
        '<div class="divider"></div>',
        unsafe_allow_html=True
    )


    # ========================================================
    # LOGOUT
    # ========================================================

    if st.button(
        "🚪 Logout",
        use_container_width=True
    ):

        st.session_state.logged_in = False
        st.session_state.username = None
        st.session_state.chat_id = None
        st.session_state.active_vectorstore = None

        st.rerun()


# ============================================================
# MAIN CHAT AREA
# ============================================================

current_chat = get_chat(
    username,
    st.session_state.chat_id
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    f"""
    <div class="app-title">
        🤖 {current_chat.get("title", "New Chat")}
    </div>

    <div class="app-subtitle">
        Ask questions about the documents in this conversation.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CURRENT CHAT INFORMATION
# ============================================================

document_count = len(
    current_chat.get(
        "documents",
        []
    )
)

image_count = len(
    current_chat.get(
        "images",
        []
    )
)

website_count = len(
    current_chat.get(
        "website_urls",
        []
    )
)

if (
    document_count
    or image_count
    or website_count
):

    st.markdown(
        f"""
        <div class="info-card">
            📄 {document_count} document(s)
            &nbsp;&nbsp;•&nbsp;&nbsp;
            🖼️ {image_count} image(s)
            &nbsp;&nbsp;•&nbsp;&nbsp;
            🌐 {website_count} website(s)

            <div class="small-muted">
                Only knowledge from this chat is used.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# WELCOME MESSAGE
# ============================================================

messages = current_chat.get(
    "messages",
    []
)

if not messages:

    st.markdown(
        """
        <div style="
            text-align:center;
            margin-top:100px;
            color:#999;
        ">

            <div style="font-size:55px;">
                🤖
            </div>

            <h2 style="color:#f5f5f5;">
                How can I help you?
            </h2>

            <p>
                Upload a document and ask questions about it.
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in messages:

    role = message.get(
        "role",
        "assistant"
    )

    content = message.get(
        "content",
        ""
    )

    if role not in [
        "user",
        "assistant"
    ]:
        continue

    avatar = (
        "👤"
        if role == "user"
        else "🤖"
    )

    with st.chat_message(
        role,
        avatar=avatar
    ):

        st.markdown(
            content
        )


# ============================================================
# CHAT INPUT
# ============================================================

prompt = st.chat_input(
    "Message AI Document Intelligence...",
    key=f"chat_input_{st.session_state.chat_id}"
)


# ============================================================
# HANDLE USER QUESTION
# ============================================================

if prompt:

    prompt = prompt.strip()

    if not prompt:
        st.stop()


    # ========================================================
    # CURRENT CHAT
    # ========================================================

    current_chat = get_chat(
        username,
        st.session_state.chat_id
    )

    if current_chat is None:
        st.error(
            "Chat could not be found."
        )
        st.stop()


    # ========================================================
    # AUTO TITLE
    # ========================================================

    if (
        current_chat.get("title")
        in [None, "", "New Chat"]
    ):

        current_chat["title"] = (
            generate_chat_title(prompt)
        )


    # ========================================================
    # DISPLAY USER MESSAGE
    # ========================================================

    with st.chat_message(
        "user",
        avatar="👤"
    ):

        st.markdown(
            prompt
        )


    # ========================================================
    # SAVE USER MESSAGE
    # ========================================================

    save_message(
        username,
        st.session_state.chat_id,
        "user",
        prompt
    )


    # ========================================================
    # GET CURRENT CHAT HISTORY
    # ========================================================

    current_chat = get_chat(
        username,
        st.session_state.chat_id
    )

    chat_history = current_chat.get(
        "messages",
        []
    )


    # ========================================================
    # SEARCH ONLY CURRENT CHAT
    # ========================================================

    with st.spinner(
        "🔎 Searching this chat's knowledge..."
    ):

        document_context, sources = (
            get_document_context(prompt)
        )


    # ========================================================
    # CURRENT CHAT IMAGES
    # ========================================================

    image_paths = (
        load_current_chat_images()
    )


    # ========================================================
    # GENERATE AI RESPONSE
    # ========================================================

    with st.chat_message(
        "assistant",
        avatar="🤖"
    ):

        with st.spinner(
            "🤖 Thinking..."
        ):

            answer = generate_ai_response(
                question=prompt,
                chat_history=chat_history,
                document_context=document_context,
                sources=sources,
                image_paths=image_paths
            )

        st.markdown(
            answer
        )


    # ========================================================
    # SOURCE INFORMATION
    # ========================================================

    if sources:

        with st.expander(
            "📚 Sources used from this chat"
        ):

            for source in sources:

                st.write(
                    f"• {source}"
                )


    # ========================================================
    # SAVE AI RESPONSE
    # ========================================================

    save_message(
        username,
        st.session_state.chat_id,
        "assistant",
        answer
    )


    # ========================================================
    # UPDATE CHAT
    # ========================================================

    current_chat = get_chat(
        username,
        st.session_state.chat_id
    )

    current_chat["updated_at"] = now_iso()

    update_chat(
        username,
        st.session_state.chat_id,
        current_chat
    )


    # ========================================================
    # RERUN
    # ========================================================

    st.rerun()

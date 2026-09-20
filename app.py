# ============================================================
# AI DOCUMENT INTELLIGENCE
# Fast Multimodal RAG Application
# ============================================================

import os
import re
import json
import uuid
import hashlib
from datetime import datetime
from io import BytesIO

import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from PIL import Image

from pypdf import PdfReader
from docx import Document as DocxDocument

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from google import genai
from google.genai import types


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Document Intelligence",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_NAME = "AI Document Intelligence"

# Fast multimodal Gemini model
GEMINI_MODEL = "gemini-2.5-flash-lite"

# Practical response size.
# Do NOT use 50,000 here for normal chat.
MAX_OUTPUT_TOKENS = 4096

# RAG settings
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150
RETRIEVER_K = 4

# Files
USERS_FILE = "users.json"
CHATS_FILE = "chats.json"
USAGE_FILE = "usage.json"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
                 Roboto, Helvetica, Arial, sans-serif;
}

.stApp {
    background: #171717;
    color: #f5f5f5;
}

/* Main area */
.main .block-container {
    max-width: 1100px;
    padding-top: 1.5rem;
    padding-bottom: 7rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #202020;
    border-right: 1px solid #303030;
}

section[data-testid="stSidebar"] * {
    color: #eeeeee;
}

/* Buttons */
.stButton > button {
    width: 100%;
    border-radius: 9px;
    border: 1px solid #3a3a3a;
    background: #292929;
    color: #f2f2f2;
    padding: 0.55rem 0.8rem;
}

.stButton > button:hover {
    background: #343434;
    border-color: #555555;
}

/* Chat messages */
[data-testid="stChatMessage"] {
    background: transparent;
    border: none;
}

/* Chat input */
[data-testid="stChatInput"] {
    background: #252525;
}

[data-testid="stChatInput"] textarea {
    background: #252525 !important;
    color: #ffffff !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: #252525;
    border-radius: 10px;
}

/* Text */
h1, h2, h3, h4, p {
    color: #f4f4f4;
}

/* Welcome */
.welcome-title {
    font-size: 38px;
    font-weight: 600;
    text-align: center;
    margin-top: 14vh;
    margin-bottom: 8px;
}

.welcome-subtitle {
    text-align: center;
    color: #a8a8a8;
    font-size: 16px;
}

/* Sidebar title */
.sidebar-title {
    font-size: 20px;
    font-weight: 650;
    margin-bottom: 20px;
}

/* Small label */
.small-label {
    color: #999999;
    font-size: 12px;
    margin-top: 10px;
}

/* Attachment box */
.attachment-box {
    background: #242424;
    border: 1px solid #383838;
    border-radius: 10px;
    padding: 10px;
    margin-bottom: 8px;
}

/* Chat title */
.chat-item {
    font-size: 14px;
    color: #dddddd;
    padding: 5px 2px;
}

/* Source */
.source-box {
    background: #222222;
    border-left: 3px solid #666666;
    padding: 8px 12px;
    margin-top: 8px;
    border-radius: 4px;
    font-size: 12px;
    color: #aaaaaa;
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

def init_session():

    defaults = {
        "logged_in": False,
        "username": None,
        "current_chat_id": None,
        "messages": [],
        "documents": [],
        "images": [],
        "retriever": None,
        "chat_title": "New chat",
        "website_sources": [],
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session()


# ============================================================
# FILE HELPERS
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
# PASSWORD HASHING
# ============================================================

def hash_password(password):

    return hashlib.sha256(
        password.encode("utf-8")
    ).hexdigest()


# ============================================================
# USERS
# ============================================================

def register_user(username, password):

    users = load_json(USERS_FILE, {})

    username = username.strip().lower()

    if not username or not password:
        return False, "Please enter username and password."

    if username in users:
        return False, "Username already exists."

    users[username] = {
        "password": hash_password(password),
        "created_at": datetime.now().isoformat()
    }

    save_json(USERS_FILE, users)

    return True, "Account created successfully."


def authenticate_user(username, password):

    users = load_json(USERS_FILE, {})

    username = username.strip().lower()

    if username not in users:
        return False

    return users[username]["password"] == hash_password(password)


# ============================================================
# CHAT STORAGE
# ============================================================

def get_all_chats():

    chats = load_json(CHATS_FILE, {})

    return chats.get(
        st.session_state.username,
        {}
    )


def save_chat():

    if not st.session_state.logged_in:
        return

    username = st.session_state.username

    chats = load_json(CHATS_FILE, {})

    if username not in chats:
        chats[username] = {}

    chat_id = st.session_state.current_chat_id

    if not chat_id:
        return

    chats[username][chat_id] = {
        "title": st.session_state.chat_title,
        "messages": st.session_state.messages,
        "updated_at": datetime.now().isoformat(),
        "documents": [
            d["name"]
            for d in st.session_state.documents
        ],
        "websites": st.session_state.website_sources,
    }

    save_json(CHATS_FILE, chats)


def create_new_chat():

    chat_id = str(uuid.uuid4())

    st.session_state.current_chat_id = chat_id
    st.session_state.messages = []
    st.session_state.documents = []
    st.session_state.images = []
    st.session_state.retriever = None
    st.session_state.website_sources = []
    st.session_state.chat_title = "New chat"


def load_chat(chat_id):

    chats = get_all_chats()

    if chat_id not in chats:
        return

    chat = chats[chat_id]

    st.session_state.current_chat_id = chat_id
    st.session_state.chat_title = chat.get(
        "title",
        "New chat"
    )

    st.session_state.messages = chat.get(
        "messages",
        []
    )

    # We intentionally reset runtime attachments.
    # Text history is persistent, uploaded binary files are runtime data.
    st.session_state.documents = []
    st.session_state.images = []
    st.session_state.retriever = None
    st.session_state.website_sources = chat.get(
        "websites",
        []
    )


def delete_chat(chat_id):

    username = st.session_state.username

    chats = load_json(CHATS_FILE, {})

    if username in chats and chat_id in chats[username]:

        del chats[username][chat_id]

        save_json(CHATS_FILE, chats)

    if st.session_state.current_chat_id == chat_id:

        create_new_chat()


# ============================================================
# USAGE
# ============================================================

def update_usage(input_tokens=0, output_tokens=0):

    usage = load_json(USAGE_FILE, {})

    username = st.session_state.username

    if username not in usage:

        usage[username] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "requests": 0,
        }

    usage[username]["input_tokens"] += input_tokens
    usage[username]["output_tokens"] += output_tokens
    usage[username]["requests"] += 1

    save_json(USAGE_FILE, usage)


# ============================================================
# GEMINI CLIENT
# ============================================================

@st.cache_resource
def get_gemini_client():

    api_key = None

    try:
        api_key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        pass

    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:

        raise RuntimeError(
            "GEMINI_API_KEY is missing. "
            "Add it to Streamlit Secrets."
        )

    return genai.Client(
        api_key=api_key
    )


# ============================================================
# TEXT SPLITTER
# ============================================================

def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):

    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    chunks = []

    start = 0

    while start < len(text):

        end = min(
            start + chunk_size,
            len(text)
        )

        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk.strip())

        if end >= len(text):
            break

        start = end - overlap

    return chunks


# ============================================================
# DOCUMENT EXTRACTION
# ============================================================

def extract_pdf(file_bytes):

    text = []

    reader = PdfReader(
        BytesIO(file_bytes)
    )

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        try:

            page_text = page.extract_text() or ""

            if page_text.strip():

                text.append(
                    f"\n[Page {page_number}]\n"
                    f"{page_text}"
                )

        except Exception:
            continue

    return "\n".join(text)


def extract_docx(file_bytes):

    document = DocxDocument(
        BytesIO(file_bytes)
    )

    paragraphs = []

    for paragraph in document.paragraphs:

        text = paragraph.text.strip()

        if text:
            paragraphs.append(text)

    return "\n".join(paragraphs)


def extract_txt(file_bytes):

    for encoding in [
        "utf-8",
        "utf-16",
        "latin-1"
    ]:

        try:

            return file_bytes.decode(
                encoding
            )

        except Exception:
            continue

    return ""


def extract_excel(file_bytes):

    excel = pd.ExcelFile(
        BytesIO(file_bytes)
    )

    output = []

    for sheet in excel.sheet_names:

        df = pd.read_excel(
            excel,
            sheet_name=sheet
        )

        output.append(
            f"\n[Sheet: {sheet}]\n"
        )

        output.append(
            df.to_string(
                index=False
            )
        )

    return "\n".join(output)


def extract_document(file):

    file_bytes = file.getvalue()

    extension = (
        file.name
        .lower()
        .split(".")[-1]
    )

    if extension == "pdf":

        return extract_pdf(
            file_bytes
        )

    elif extension == "docx":

        return extract_docx(
            file_bytes
        )

    elif extension == "txt":

        return extract_txt(
            file_bytes
        )

    elif extension in [
        "xlsx",
        "xls"
    ]:

        return extract_excel(
            file_bytes
        )

    return ""


# ============================================================
# WEBSITE EXTRACTION
# ============================================================

def extract_website(url):

    try:

        if not url.startswith(
            ("http://", "https://")
        ):
            url = "https://" + url

        response = requests.get(
            url,
            timeout=15,
            headers={
                "User-Agent":
                "Mozilla/5.0"
            }
        )

        response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        for tag in soup([
            "script",
            "style",
            "noscript",
            "nav",
            "footer"
        ]):

            tag.decompose()

        text = soup.get_text(
            separator=" ",
            strip=True
        )

        return text

    except Exception as e:

        return f"Website extraction failed: {e}"


# ============================================================
# EMBEDDINGS
# ============================================================

@st.cache_resource
def get_embeddings():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# ============================================================
# BUILD RAG DATABASE
# ============================================================

def rebuild_retriever():

    all_documents = []

    for item in st.session_state.documents:

        filename = item["name"]
        text = item["text"]

        chunks = split_text(text)

        for index, chunk in enumerate(chunks):

            all_documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": filename,
                        "chunk": index
                    }
                )
            )

    # Website content
    for website in st.session_state.website_sources:

        chunks = split_text(
            website["text"]
        )

        for index, chunk in enumerate(chunks):

            all_documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": website["url"],
                        "chunk": index
                    }
                )
            )

    if not all_documents:

        st.session_state.retriever = None

        return

    embeddings = get_embeddings()

    vector_store = FAISS.from_documents(
        all_documents,
        embeddings
    )

    st.session_state.retriever = (
        vector_store.as_retriever(
            search_kwargs={
                "k": RETRIEVER_K
            }
        )
    )


# ============================================================
# RETRIEVE RELEVANT CONTEXT
# ============================================================

def get_rag_context(question):

    retriever = st.session_state.retriever

    if retriever is None:

        return "", []

    try:

        docs = retriever.invoke(
            question
        )

    except Exception:

        return "", []

    context_parts = []
    sources = []

    for doc in docs:

        source = doc.metadata.get(
            "source",
            "Unknown"
        )

        context_parts.append(
            f"[SOURCE: {source}]\n"
            f"{doc.page_content}"
        )

        if source not in sources:
            sources.append(source)

    context = "\n\n".join(
        context_parts
    )

    # Keep prompt reasonably small.
    context = context[:14000]

    return context, sources


# ============================================================
# TOKEN ESTIMATION
# ============================================================

def estimate_tokens(text):

    if not text:
        return 0

    # Rough estimation.
    # Gemini token counts are not the same as this estimate.
    return max(
        1,
        len(text) // 4
    )


# ============================================================
# CREATE CHAT TITLE
# ============================================================

def create_title(question):

    clean = re.sub(
        r"\s+",
        " ",
        question
    ).strip()

    if len(clean) <= 40:
        return clean

    return clean[:40].rstrip() + "..."


# ============================================================
# GEMINI STREAMING RESPONSE
# ============================================================

def generate_answer(
    question,
    context,
    image_attachments,
    previous_messages
):

    client = get_gemini_client()

    system_instruction = """
You are an AI Document Intelligence assistant.

Your job is to answer the user's question clearly and accurately.

Rules:

1. If document context is provided, use it as the main source.
2. Do not invent facts that are not supported by the document.
3. If the answer is not available in the provided documents,
   clearly say that it is not available in the provided content.
4. If an image is provided, analyze the image carefully.
5. If both documents and images are provided, combine the information.
6. Keep answers direct and useful.
7. Use headings and bullet points when useful.
8. For numerical questions, show the important calculation.
9. For tables or structured data, explain the relevant rows/columns.
10. If the user asks for a summary, summarize instead of giving
    unnecessary explanations.
"""

    prompt_parts = []

    if context:

        prompt_parts.append(
            "DOCUMENT CONTEXT:\n"
            + context
        )

    if previous_messages:

        history_text = ""

        for message in previous_messages[-6:]:

            role = message.get(
                "role",
                ""
            )

            content = message.get(
                "content",
                ""
            )

            if content:

                history_text += (
                    f"{role.upper()}: "
                    f"{content}\n"
                )

        if history_text:

            prompt_parts.append(
                "RECENT CONVERSATION:\n"
                + history_text
            )

    prompt_parts.append(
        "USER QUESTION:\n"
        + question
    )

    prompt = "\n\n".join(
        prompt_parts
    )

    contents = []

    # Add text first
    contents.append(prompt)

    # Add uploaded images
    for image_data in image_attachments:

        try:

            image_part = types.Part.from_bytes(
                data=image_data["data"],
                mime_type=image_data["mime_type"]
            )

            contents.append(
                image_part
            )

        except Exception:
            continue

    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        temperature=0.2,
    )

    response_stream = client.models.generate_content_stream(
        model=GEMINI_MODEL,
        contents=contents,
        config=config,
    )

    full_response = ""

    for chunk in response_stream:

        try:

            text = chunk.text

        except Exception:

            text = None

        if text:

            full_response += text

            yield text

    input_tokens = estimate_tokens(
        prompt
    )

    output_tokens = estimate_tokens(
        full_response
    )

    update_usage(
        input_tokens,
        output_tokens
    )


# ============================================================
# LOGIN PAGE
# ============================================================

def login_page():

    st.markdown(
        """
        <div style="
            max-width:500px;
            margin:100px auto 0 auto;
            text-align:center;
        ">
            <div style="
                font-size:42px;
                margin-bottom:10px;
            ">
                ✦
            </div>

            <h1>
                AI Document Intelligence
            </h1>

            <p style="color:#999;">
                Chat with your documents and images
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    tab1, tab2 = st.tabs(
        ["Login", "Create account"]
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
            key="login_button"
        ):

            if authenticate_user(
                username,
                password
            ):

                st.session_state.logged_in = True
                st.session_state.username = (
                    username.strip().lower()
                )

                create_new_chat()

                st.rerun()

            else:

                st.error(
                    "Invalid username or password."
                )

    with tab2:

        new_username = st.text_input(
            "Username",
            key="register_username"
        )

        new_password = st.text_input(
            "Password",
            type="password",
            key="register_password"
        )

        confirm_password = st.text_input(
            "Confirm password",
            type="password",
            key="register_confirm"
        )

        if st.button(
            "Create account",
            key="register_button"
        ):

            if new_password != confirm_password:

                st.error(
                    "Passwords do not match."
                )

            else:

                success, message = register_user(
                    new_username,
                    new_password
                )

                if success:

                    st.success(message)

                else:

                    st.error(message)


# ============================================================
# SIDEBAR
# ============================================================

def render_sidebar():

    with st.sidebar:

        st.markdown(
            '<div class="sidebar-title">✦ AI Document Intelligence</div>',
            unsafe_allow_html=True
        )

        if st.button(
            "＋  New chat",
            key="new_chat"
        ):

            create_new_chat()

            st.rerun()

        st.divider()

        # ----------------------------------------------------
        # KNOWLEDGE
        # ----------------------------------------------------

        st.markdown(
            "**Knowledge**"
        )

        documents = st.file_uploader(
            "Add documents",
            type=[
                "pdf",
                "docx",
                "txt",
                "xlsx",
                "xls"
            ],
            accept_multiple_files=True,
            key="document_uploader",
        )

        if documents:

            existing_names = {
                d["name"]
                for d in st.session_state.documents
            }

            changed = False

            for file in documents:

                if file.name in existing_names:
                    continue

                try:

                    extracted_text = extract_document(
                        file
                    )

                    if extracted_text.strip():

                        st.session_state.documents.append(
                            {
                                "name": file.name,
                                "text": extracted_text,
                            }
                        )

                        changed = True

                    else:

                        st.warning(
                            f"No text found in {file.name}"
                        )

                except Exception as e:

                    st.error(
                        f"Could not read {file.name}: {e}"
                    )

            if changed:

                rebuild_retriever()

                st.success(
                    f"{len(documents)} document(s) added."
                )

        # ----------------------------------------------------
        # IMAGES
        # ----------------------------------------------------

        images = st.file_uploader(
            "Add images",
            type=[
                "png",
                "jpg",
                "jpeg",
                "webp"
            ],
            accept_multiple_files=True,
            key="image_uploader",
        )

        if images:

            existing_images = {
                image["name"]
                for image in st.session_state.images
            }

            for image in images:

                if image.name in existing_images:
                    continue

                st.session_state.images.append(
                    {
                        "name": image.name,
                        "mime_type": image.type
                        or "image/jpeg",
                        "data": image.getvalue(),
                    }
                )

        # ----------------------------------------------------
        # SHOW ATTACHMENTS
        # ----------------------------------------------------

        if st.session_state.documents:

            st.markdown(
                '<div class="small-label">DOCUMENTS</div>',
                unsafe_allow_html=True
            )

            for document in st.session_state.documents:

                st.markdown(
                    f"""
                    <div class="attachment-box">
                        📄 {document["name"]}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        if st.session_state.images:

            st.markdown(
                '<div class="small-label">IMAGES</div>',
                unsafe_allow_html=True
            )

            for image in st.session_state.images:

                st.markdown(
                    f"""
                    <div class="attachment-box">
                        🖼️ {image["name"]}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # ----------------------------------------------------
        # WEBSITE
        # ----------------------------------------------------

        st.markdown(
            '<div class="small-label">WEBSITE</div>',
            unsafe_allow_html=True
        )

        website_url = st.text_input(
            "Add a website",
            placeholder="https://example.com",
            label_visibility="collapsed",
            key="website_url"
        )

        if st.button(
            "＋ Add website",
            key="add_website"
        ):

            if website_url.strip():

                with st.spinner(
                    "Reading website..."
                ):

                    website_text = extract_website(
                        website_url.strip()
                    )

                if website_text:

                    st.session_state.website_sources.append(
                        {
                            "url": website_url.strip(),
                            "text": website_text,
                        }
                    )

                    rebuild_retriever()

                    st.success(
                        "Website added."
                    )

        # ----------------------------------------------------
        # CHATS
        # ----------------------------------------------------

        st.divider()

        st.markdown(
            "**Chats**"
        )

        chats = get_all_chats()

        if not chats:

            st.caption(
                "No previous chats"
            )

        else:

            sorted_chats = sorted(
                chats.items(),
                key=lambda item: item[1].get(
                    "updated_at",
                    ""
                ),
                reverse=True
            )

            for chat_id, chat in sorted_chats[:30]:

                title = chat.get(
                    "title",
                    "New chat"
                )

                col1, col2 = st.columns(
                    [5, 1]
                )

                with col1:

                    if st.button(
                        title,
                        key=f"open_{chat_id}"
                    ):

                        load_chat(chat_id)

                        st.rerun()

                with col2:

                    if st.button(
                        "×",
                        key=f"delete_{chat_id}"
                    ):

                        delete_chat(
                            chat_id
                        )

                        st.rerun()

        # ----------------------------------------------------
        # ACCOUNT
        # ----------------------------------------------------

        st.divider()

        st.caption(
            f"Signed in as {st.session_state.username}"
        )

        if st.button(
            "Sign out",
            key="logout"
        ):

            st.session_state.logged_in = False
            st.session_state.username = None

            st.session_state.current_chat_id = None
            st.session_state.messages = []
            st.session_state.documents = []
            st.session_state.images = []
            st.session_state.retriever = None

            st.rerun()


# ============================================================
# MAIN CHAT
# ============================================================

def render_main():

    # --------------------------------------------------------
    # WELCOME SCREEN
    # --------------------------------------------------------

    if not st.session_state.messages:

        st.markdown(
            """
            <div class="welcome-title">
                Good Morning
            </div>

            <div class="welcome-subtitle">
                How can I help you with your documents today?
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.markdown(
                """
                **📄 Analyze documents**

                Upload PDF, Word, Excel or text files
                and ask questions about them.
                """
            )

        with col2:

            st.markdown(
                """
                **🖼️ Understand images**

                Upload an image and ask questions
                about charts, screenshots or text.
                """
            )

        with col3:

            st.markdown(
                """
                **🔎 RAG search**

                Find relevant information from
                multiple uploaded documents.
                """
            )

    # --------------------------------------------------------
    # DISPLAY MESSAGES
    # --------------------------------------------------------

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

            st.markdown(
                content
            )

            sources = message.get(
                "sources",
                []
            )

            if sources:

                st.markdown(
                    '<div class="source-box">'
                    + "Sources: "
                    + ", ".join(sources)
                    + "</div>",
                    unsafe_allow_html=True
                )

    # --------------------------------------------------------
    # CHAT INPUT
    # --------------------------------------------------------

    question = st.chat_input(
        "Ask anything about your documents..."
    )

    if not question:
        return

    question = question.strip()

    if not question:
        return

    # --------------------------------------------------------
    # USER MESSAGE
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    # Automatic title
    if (
        st.session_state.chat_title
        == "New chat"
    ):

        st.session_state.chat_title = (
            create_title(question)
        )

    # --------------------------------------------------------
    # SHOW USER MESSAGE
    # --------------------------------------------------------

    with st.chat_message("user"):

        st.markdown(
            question
        )

    # --------------------------------------------------------
    # RAG
    # --------------------------------------------------------

    context, sources = get_rag_context(
        question
    )

    # --------------------------------------------------------
    # IMAGE ATTACHMENTS
    # --------------------------------------------------------

    image_attachments = (
        st.session_state.images.copy()
    )

    # --------------------------------------------------------
    # GENERATE
    # --------------------------------------------------------

    with st.chat_message(
        "assistant"
    ):

        try:

            response_placeholder = st.empty()

            response_text = ""

            generator = generate_answer(
                question=question,
                context=context,
                image_attachments=image_attachments,
                previous_messages=st.session_state.messages[:-1],
            )

            for token in generator:

                response_text += token

                response_placeholder.markdown(
                    response_text
                )

            if not response_text.strip():

                response_text = (
                    "I couldn't generate an answer."
                )

                response_placeholder.markdown(
                    response_text
                )

            if sources:

                st.markdown(
                    '<div class="source-box">'
                    + "Sources: "
                    + ", ".join(sources)
                    + "</div>",
                    unsafe_allow_html=True
                )

        except Exception as e:

            response_text = (
                "Sorry, something went wrong.\n\n"
                f"`{str(e)}`"
            )

            st.error(
                response_text
            )

    # --------------------------------------------------------
    # SAVE ASSISTANT MESSAGE
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response_text,
            "sources": sources,
        }
    )

    save_chat()

    # Keep uploaded images available for
    # the current chat. They are intentionally
    # not placed inside chats.json.

    st.rerun()


# ============================================================
# APPLICATION
# ============================================================

def main():

    if not st.session_state.logged_in:

        login_page()

        return

    # Ensure current chat exists
    if not st.session_state.current_chat_id:

        create_new_chat()

    render_sidebar()

    render_main()


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    main()

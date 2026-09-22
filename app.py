# ============================================================
# AI DOCUMENT INTELLIGENCE
# ChatGPT-style Recent Chats
# Separate Chat History + Separate Documents per Chat
# ============================================================

import os
import re
import json
import uuid
import hashlib
import shutil
import time
import subprocess
import tempfile
from pathlib import Path

try:
    from textblob import TextBlob
except Exception:
    TextBlob = None
from datetime import datetime, timedelta

import streamlit as st
import requests
import pandas as pd

from PIL import Image, ImageDraw, ImageFont
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
# SETTINGS
# ============================================================

APP_NAME = "AI Document Intelligence"

GEMINI_MODEL = "gemini-3.5-flash"

GEMINI_FALLBACK_MODELS = [
    "gemini-3.5-flash-lite",
]

GEMINI_RETRY_COUNT = 2
GEMINI_RETRY_DELAY_SECONDS = 2

MAX_OUTPUT_TOKENS = 65536

AVAILABLE_GEMINI_MODELS = [
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
]

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150
RETRIEVER_K = 5

BASE_DIR = Path(".")

USERS_FILE = BASE_DIR / "users.json"
CHATS_FILE = BASE_DIR / "chats.json"

CHAT_STORAGE = BASE_DIR / "chat_storage"

CHAT_STORAGE.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
<style>

/* ========================================================
   ChatGPT-inspired dark workspace
   ======================================================== */
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
                 Helvetica, Arial, sans-serif;
}

.stApp {
    background: #212121;
    color: #ececec;
}

header[data-testid="stHeader"] {
    background: #212121;
}

.main .block-container {
    max-width: 768px;
    padding-top: 1.25rem;
    padding-bottom: 8rem;
}

section[data-testid="stSidebar"] {
    background: #171717;
    border-right: 1px solid #2f2f2f;
}

section[data-testid="stSidebar"] > div {
    background: #171717;
}

section[data-testid="stSidebar"] * {
    color: #ececec;
}

section[data-testid="stSidebar"] button {
    border: 0 !important;
    background: transparent !important;
    text-align: left !important;
    border-radius: 8px !important;
}

section[data-testid="stSidebar"] button:hover {
    background: #2a2a2a !important;
}

section[data-testid="stSidebar"] [data-testid="stFileUploader"] {
    background: #202020;
    border-radius: 10px;
    padding: 4px;
}

[data-testid="stChatMessage"] {
    background: transparent !important;
    border: 0 !important;
    padding: 1.1rem 0 !important;
    margin: 0 !important;
}

[data-testid="stChatMessageContent"] {
    font-size: 15.5px;
    line-height: 1.65;
    color: #ececec;
}

[data-testid="stChatInput"] {
    background: #2f2f2f !important;
    border: 1px solid #424242 !important;
    border-radius: 26px !important;
    box-shadow: 0 2px 12px rgba(0,0,0,.22);
}

[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: #f5f5f5 !important;
    border: none !important;
    font-size: 15px !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: #9b9b9b !important;
}

[data-testid="stChatInput"] button {
    border-radius: 50% !important;
}

button[kind="primary"] {
    background: #ffffff !important;
    color: #171717 !important;
    border: 0 !important;
}

.chat-title {
    font-size: 16px;
    font-weight: 600;
    color: #ececec;
}

.chat-subtitle {
    color: #8e8e8e;
    font-size: 12px;
}

.welcome-wrap {
    text-align: center;
    margin-top: 20vh;
}

.welcome-icon {
    font-size: 38px;
    margin-bottom: 14px;
}

.welcome-title {
    color: #ececec;
    font-size: 30px;
    font-weight: 600;
    margin-bottom: 8px;
}

.welcome-text {
    color: #9b9b9b;
    font-size: 14px;
}



.emotion-card {
    background: #242424;
    border: 1px solid #383838;
    border-radius: 12px;
    padding: 12px 14px;
    margin-top: 10px;
}

.divider {
    height: 1px;
    background: #303030;
    margin: 12px 0;
}

.small-muted {
    color: #8e8e8e;
    font-size: 12px;
}

/* Remove Streamlit's default decorative spacing around chat controls. */
[data-testid="stBottomBlockContainer"] {
    background: linear-gradient(transparent, #212121 25%);
}

</style>
""",
    unsafe_allow_html=True
)

# ============================================================
# GENERAL HELPERS
# ============================================================

def now_iso():
    return datetime.now().isoformat()


def safe_username(username):
    return re.sub(
        r"[^a-zA-Z0-9_.-]",
        "_",
        username
    )


def safe_filename(filename):
    filename = Path(filename).name

    return re.sub(
        r"[^a-zA-Z0-9_. -]",
        "_",
        filename
    )


def file_hash(data):
    return hashlib.sha256(data).hexdigest()


# ============================================================
# CHAT FOLDERS
# ============================================================

def get_chat_folder(
    username,
    chat_id
):

    folder = (
        CHAT_STORAGE
        / safe_username(username)
        / chat_id
    )

    folder.mkdir(
        parents=True,
        exist_ok=True
    )

    return folder


def get_documents_folder(
    username,
    chat_id
):

    folder = (
        get_chat_folder(
            username,
            chat_id
        )
        / "documents"
    )

    folder.mkdir(
        parents=True,
        exist_ok=True
    )

    return folder


def get_images_folder(
    username,
    chat_id
):

    folder = (
        get_chat_folder(
            username,
            chat_id
        )
        / "images"
    )

    folder.mkdir(
        parents=True,
        exist_ok=True
    )

    return folder


def get_knowledge_file(
    username,
    chat_id
):

    return (
        get_chat_folder(
            username,
            chat_id
        )
        / "knowledge.json"
    )


# ============================================================
# JSON FUNCTIONS
# ============================================================

def load_json(
    path,
    default
):

    try:

        if not path.exists():
            return default

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    except Exception:

        return default


def save_json(
    path,
    data
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    temp_path = path.with_suffix(
        ".tmp"
    )

    with open(
        temp_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False
        )

    temp_path.replace(path)


# ============================================================
# USERS
# ============================================================

def load_users():
    return load_json(
        USERS_FILE,
        {}
    )


def save_users(users):
    save_json(
        USERS_FILE,
        users
    )


def create_user(
    username,
    password
):

    users = load_users()

    if username in users:

        return (
            False,
            "Username already exists."
        )

    users[username] = {
        "password": password,
        "created_at": now_iso()
    }

    save_users(users)

    return (
        True,
        "Account created successfully."
    )


def login_user(
    username,
    password
):

    users = load_users()

    if username not in users:
        return False

    return (
        users[username].get("password")
        == password
    )


# ============================================================
# CHAT DATABASE
# ============================================================

def load_chats():

    return load_json(
        CHATS_FILE,
        {}
    )


def save_chats(chats):

    save_json(
        CHATS_FILE,
        chats
    )


def create_chat(
    username,
    title="New Chat"
):

    chats = load_chats()

    if username not in chats:
        chats[username] = {}

    chat_id = uuid.uuid4().hex

    chats[username][chat_id] = {
        "id": chat_id,
        "title": title,
        "name": title,
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "messages": [],
        "documents": [],
        "images": [],
        "image_hashes": [],
        "website_urls": []
    }

    save_chats(chats)

    get_documents_folder(
        username,
        chat_id
    )

    get_images_folder(
        username,
        chat_id
    )

    save_knowledge_manifest(
        username,
        chat_id,
        {
            "documents": [],
            "websites": []
        }
    )

    return chat_id


def get_user_chats(username):

    chats = load_chats()

    return chats.get(
        username,
        {}
    )


def get_chat(
    username,
    chat_id
):

    chats = load_chats()

    return (
        chats
        .get(username, {})
        .get(chat_id)
    )


def update_chat(
    username,
    chat_id,
    chat
):

    chats = load_chats()

    if username not in chats:
        chats[username] = {}

    chats[username][chat_id] = chat

    save_chats(chats)


def delete_chat(
    username,
    chat_id
):

    chats = load_chats()

    if username in chats:

        chats[username].pop(
            chat_id,
            None
        )

    save_chats(chats)

    folder = get_chat_folder(
        username,
        chat_id
    )

    if folder.exists():

        shutil.rmtree(
            folder,
            ignore_errors=True
        )


def rename_chat(
    username,
    chat_id,
    title
):

    chat = get_chat(
        username,
        chat_id
    )

    if not chat:
        return

    clean_title = title.strip()[:80]
    chat["title"] = clean_title
    chat["name"] = clean_title
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

    question = re.sub(
        r"\s+",
        " ",
        question.strip()
    )

    if not question:
        return "New Chat"

    if len(question) > 45:

        return (
            question[:45].rstrip()
            + "..."
        )

    return question


# ============================================================
# RECENT CHAT GROUPING
# ============================================================

def parse_datetime(value):

    try:
        return datetime.fromisoformat(
            value
        )

    except Exception:

        return datetime.now()


def group_chats_by_date(
    chats
):

    today = datetime.now().date()

    yesterday = (
        today
        - timedelta(days=1)
    )

    seven_days_ago = (
        today
        - timedelta(days=7)
    )

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
                chat.get(
                    "created_at",
                    now_iso()
                )
            )
        )

        chat_date = dt.date()

        if chat_date == today:

            groups["Today"].append(
                chat
            )

        elif chat_date == yesterday:

            groups["Yesterday"].append(
                chat
            )

        elif chat_date >= seven_days_ago:

            groups["Previous 7 days"].append(
                chat
            )

        else:

            groups["Older"].append(
                chat
            )

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

    while start < len(text):

        end = start + chunk_size

        chunk = text[
            start:end
        ]

        if chunk.strip():

            chunks.append(
                chunk.strip()
            )

        next_start = (
            end - overlap
        )

        if next_start <= start:

            next_start = end

        start = next_start

    return chunks


# ============================================================
# DOCUMENT EXTRACTION
# ============================================================

def extract_pdf(
    file_path
):

    result = []

    reader = PdfReader(
        str(file_path)
    )

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        try:

            text = (
                page.extract_text()
                or ""
            )

            if text.strip():

                result.append(
                    f"[Page {page_number}]\n{text}"
                )

        except Exception:
            continue

    return "\n\n".join(result)


def extract_docx(
    file_path
):

    document = Document(
        str(file_path)
    )

    result = []

    for paragraph in document.paragraphs:

        text = paragraph.text.strip()

        if text:

            result.append(text)

    return "\n".join(result)


def extract_xlsx(
    file_path
):

    result = []

    excel_file = pd.ExcelFile(
        file_path
    )

    for sheet in excel_file.sheet_names:

        df = pd.read_excel(
            file_path,
            sheet_name=sheet
        )

        result.append(
            f"Sheet: {sheet}\n"
            + df.to_string(
                index=False
            )
        )

    return "\n\n".join(result)


def extract_txt(
    file_path
):

    with open(
        file_path,
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as file:

        return file.read()


def extract_document(
    file_path
):

    extension = (
        Path(file_path)
        .suffix
        .lower()
    )

    if extension == ".pdf":
        return extract_pdf(file_path)

    if extension == ".docx":
        return extract_docx(file_path)

    if extension in [
        ".xlsx",
        ".xls"
    ]:
        return extract_xlsx(file_path)

    if extension in [
        ".txt",
        ".md",
        ".csv"
    ]:
        return extract_txt(file_path)

    return ""


# ============================================================
# WEBSITE EXTRACTION
# ============================================================

def extract_website(url):

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

    html = re.sub(
        r"<script.*?</script>",
        " ",
        html,
        flags=re.DOTALL |
        re.IGNORECASE
    )

    html = re.sub(
        r"<style.*?</style>",
        " ",
        html,
        flags=re.DOTALL |
        re.IGNORECASE
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


# ============================================================
# KNOWLEDGE MANIFEST
# ============================================================

def load_knowledge_manifest(
    username,
    chat_id
):

    return load_json(
        get_knowledge_file(
            username,
            chat_id
        ),
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

    save_json(
        get_knowledge_file(
            username,
            chat_id
        ),
        manifest
    )


# ============================================================
# EMBEDDINGS
# ============================================================

@st.cache_resource
def get_embeddings():

    return HuggingFaceEmbeddings(
        model_name=
        "sentence-transformers/all-MiniLM-L6-v2"
    )


# ============================================================
# BUILD CURRENT CHAT VECTORSTORE
# ============================================================

def build_chat_vectorstore(
    username,
    chat_id
):

    manifest = load_knowledge_manifest(
        username,
        chat_id
    )

    records = []


    # --------------------------------------------------------
    # DOCUMENTS
    # --------------------------------------------------------

    for item in manifest.get(
        "documents",
        []
    ):

        path = Path(
            item.get(
                "path",
                ""
            )
        )

        if not path.exists():
            continue

        try:

            text = extract_document(
                path
            )

            if not text.strip():
                continue

            chunks = split_text(
                text
            )

            for index, chunk in enumerate(
                chunks
            ):

                records.append(
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


    # --------------------------------------------------------
    # WEBSITES
    # --------------------------------------------------------

    for item in manifest.get(
        "websites",
        []
    ):

        text = item.get(
            "text",
            ""
        )

        if not text.strip():
            continue

        chunks = split_text(
            text
        )

        for index, chunk in enumerate(
            chunks
        ):

            records.append(
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


    if not records:
        return None


    texts = [
        item["text"]
        for item in records
    ]

    metadatas = [
        {
            "source": item["source"],
            "type": item["type"],
            "chunk": item["chunk"]
        }
        for item in records
    ]

    vectorstore = FAISS.from_texts(
        texts=texts,
        embedding=get_embeddings(),
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

        st.session_state.active_vectorstore = None

        return

    try:

        st.session_state.active_vectorstore = (
            build_chat_vectorstore(
                username,
                chat_id
            )
        )

    except Exception:

        st.session_state.active_vectorstore = None


# ============================================================
# SEARCH CURRENT CHAT ONLY
# ============================================================

def get_document_context(
    question
):

    vectorstore = (
        st.session_state.get(
            "active_vectorstore"
        )
    )

    if vectorstore is None:

        return "", []


    try:

        results = (
            vectorstore.similarity_search(
                question,
                k=RETRIEVER_K
            )
        )

    except Exception:

        return "", []


    context_parts = []
    sources = []

    for result in results:

        text = result.page_content

        metadata = (
            result.metadata
            or {}
        )

        source = metadata.get(
            "source",
            "Unknown"
        )

        context_parts.append(
            f"SOURCE: {source}\n{text}"
        )

        if source not in sources:

            sources.append(
                source
            )

    return (
        "\n\n---\n\n".join(
            context_parts
        ),
        sources
    )


# ============================================================
# CURRENT CHAT IMAGES
# ============================================================

def load_current_chat_images():

    username = (
        st.session_state.get(
            "username"
        )
    )

    chat_id = (
        st.session_state.get(
            "chat_id"
        )
    )

    if not username or not chat_id:

        return []

    folder = get_images_folder(
        username,
        chat_id
    )

    result = []

    for path in folder.iterdir():

        if path.suffix.lower() in [
            ".png",
            ".jpg",
            ".jpeg",
            ".webp"
        ]:

            result.append(path)

    return result


# ============================================================
# GEMINI CLIENT
# ============================================================

def get_gemini_client():

    api_key = st.secrets.get(
        "GEMINI_API_KEY",
        os.getenv(
            "GEMINI_API_KEY",
            ""
        )
    )

    if not api_key:

        return None

    return genai.Client(
        api_key=api_key
    )


# ============================================================
# ADVANCED AI CAPABILITIES
# ============================================================

def get_selected_model():
    """Return the model selected for the current session."""
    return st.session_state.get("selected_gemini_model", GEMINI_MODEL)


def transcribe_audio_bytes(audio_bytes, mime_type="audio/wav"):
    """Transcribe a voice recording using the selected Gemini model."""
    client = get_gemini_client()
    if client is None:
        return None, "Gemini API key is not configured."

    try:
        response = client.models.generate_content(
            model=get_selected_model(),
            contents=[
                types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
                "Transcribe this audio exactly. Return only the spoken words."
            ]
        )
        return (response.text or "").strip(), None
    except Exception as error:
        return None, str(error)


def run_python_code_safely(code):
    """Run a short Python snippet in a temporary directory with a timeout.

    This is intentionally limited and should not be treated as a secure sandbox.
    Do not use it for untrusted production workloads.
    """
    if len(code) > 12000:
        return "Code is too long for the built-in execution tool."

    blocked = [
        "os.system", "subprocess", "shutil.rmtree", "socket", "requests.",
        "urllib", "httpx", "pathlib.Path('/", "pathlib.Path(\"/",
        "__import__", "eval(", "exec(", "open("
    ]
    lowered = code.lower().replace(" ", "")
    if any(item.lower().replace(" ", "") in lowered for item in blocked):
        return "For safety, this project does not execute code that accesses the operating system, network, or arbitrary files."

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = subprocess.run(
                ["python", "-I", "-c", code],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                timeout=8,
                env={"PATH": os.environ.get("PATH", "")}
            )
            output = (result.stdout + result.stderr).strip()
            if not output:
                output = "Code executed successfully with no output."
            return output[:12000]
    except subprocess.TimeoutExpired:
        return "Execution stopped because it exceeded the 8-second limit."
    except Exception as error:
        return f"Execution error: {error}"


# ============================================================
# FLOWCHART GENERATION
# ============================================================

def is_flowchart_request(question):
    keywords = ["flowchart", "flow chart", "process diagram", "workflow diagram", "create a diagram", "draw a flowchart", "make a flowchart", "generate a flowchart"]
    return any(keyword in question.lower() for keyword in keywords)

def generate_flowchart_data(question):
    client = get_gemini_client()
    if client is None:
        return None
    prompt = f"""Create a simple and clear flowchart for this request:

{question}

Return ONLY valid JSON using exactly this format:
{{"title":"Flowchart Title","nodes":[{{"id":"1","text":"Start"}},{{"id":"2","text":"Process"}},{{"id":"3","text":"End"}}],"connections":[{{"from":"1","to":"2"}},{{"from":"2","to":"3"}}]}}

Rules: 3-10 nodes; start with Start; end with End; short clear node text; simple connections; JSON only; no Markdown."""
    try:
        response = client.models.generate_content(model=get_selected_model(), contents=prompt, config=types.GenerateContentConfig(max_output_tokens=4000))
        result = (response.text or "").strip().replace("```json", "").replace("```", "").strip()
        return json.loads(result)
    except Exception as error:
        print("Flowchart generation error:", error)
        return None

def create_flowchart_image(flowchart_data):
    nodes = flowchart_data.get("nodes", [])
    connections = flowchart_data.get("connections", [])
    if not nodes:
        return None
    width, node_width, node_height, vertical_gap = 1000, 600, 100, 80
    height = 160 + len(nodes) * (node_height + vertical_gap)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    try:
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 34)
        node_font = ImageFont.truetype("DejaVuSans.ttf", 25)
    except Exception:
        title_font = ImageFont.load_default(); node_font = ImageFont.load_default()
    title = flowchart_data.get("title", "AI Generated Flowchart")
    box = draw.textbbox((0,0), title, font=title_font)
    draw.text(((width-(box[2]-box[0]))/2,30), title, fill="black", font=title_font)
    positions = {}
    for i,node in enumerate(nodes):
        positions[str(node.get("id",i+1))] = ((width-node_width)//2, 100+i*(node_height+vertical_gap))
    for c in connections:
        source, target = positions.get(str(c.get("from",""))), positions.get(str(c.get("to","")))
        if not source or not target: continue
        x1,y1=source[0]+node_width//2,source[1]+node_height; x2,y2=target[0]+node_width//2,target[1]
        draw.line((x1,y1,x2,y2),fill="black",width=4)
        a=12; draw.polygon([(x2,y2),(x2-a,y2-a),(x2+a,y2-a)],fill="black")
    for i,node in enumerate(nodes):
        node_id=str(node.get("id",i+1)); x,y=positions[node_id]
        draw.rounded_rectangle((x,y,x+node_width,y+node_height),radius=20,outline="black",width=3)
        text=str(node.get("text","")); box=draw.textbbox((0,0),text,font=node_font)
        draw.text((x+(node_width-(box[2]-box[0]))/2,y+(node_height-(box[3]-box[1]))/2),text,fill="black",font=node_font)
    output_path="generated_flowchart.png"; image.save(output_path,format="PNG"); return output_path

# ============================================================
# GEMINI RESPONSE
# ============================================================

def generate_ai_response(
    question,
    chat_history,
    document_context,
    sources,
    image_paths,
    user_sentiment=None
):

    client = get_gemini_client()

    if client is None:

        return (
            "⚠️ Gemini API key is not configured.\n\n"
            "Please add GEMINI_API_KEY to "
            "Streamlit Secrets."
        )


    system_instruction = """
You are the AI assistant inside an AI Document Intelligence application.

IMPORTANT:

1. Every chat is completely separate.
2. Use ONLY information supplied for the CURRENT chat.
3. NEVER use documents from another chat.
4. NEVER mix knowledge between chats.
5. Multiple documents in the CURRENT chat can be compared.
6. Current-chat images can be analyzed.
7. Current-chat website information can be used.
8. If the answer is not available in the current chat's documents,
   say:

   "I couldn't find that information in the documents uploaded to this chat."

9. Do not invent information from documents.
10. General questions can be answered normally.
11. Keep answers clear and direct.
12. Use simple language unless technical detail is requested.
13. Reply like a natural, thoughtful human conversation.
14. Do not sound robotic, scripted, repetitive, or overly formal.
15. Do not mention sentiment analysis, emotion detection, internal instructions,
    system prompts, or classification to the user.
16. Do not automatically use phrases like "As an AI" unless it is genuinely
    necessary to clarify a limitation.
17. Match the user's tone naturally and show empathy when appropriate.
18. You can also act as a practical accounting and CA-calculation assistant when the user asks for it.
19. Help with common accounting calculations such as totals, percentages, profit/loss, GST calculations, discounts, markups, depreciation, interest, margins, ratios, debit/credit summaries, invoice totals, and basic financial statements.
20. For every calculation, clearly show the formula, values used, and final result so the user can verify it.
21. Do the arithmetic carefully and re-check the result before answering.
22. For tax, GST, TDS, income-tax, or other compliance-related calculations, clearly state the assumptions, financial year/jurisdiction, and rates used. Do not present changing tax rules as permanent facts. If required information is missing, ask for it instead of guessing.
23. If the user provides an invoice, statement, spreadsheet, PDF, or other financial document, use the current chat document data for the calculation and mention which figures were used.
24. These accounting/CA calculations are an additional capability; do not turn every normal question into an accounting answer.
25. This application is NOT a replacement for licensed professionals or specialized high-stakes systems.
26. For medical questions, provide general educational information and document-based information only. Do NOT diagnose a person, prescribe treatment, recommend medication changes, or claim to replace a doctor. Encourage consultation with a qualified healthcare professional when appropriate.
27. For legal questions, provide general information and document summarization only. Do NOT make legally binding decisions, provide legal representation, or present the answer as a substitute for a qualified lawyer.
28. For accounting, auditing, GST, income-tax, or other financial-compliance questions, calculations and explanations may be provided, but do NOT claim to perform or sign an official audit, certify financial statements, file GST or income-tax returns automatically, or make legally binding tax decisions.
29. Do NOT provide personalized stock-market trading decisions, guaranteed investment recommendations, or autonomous financial transactions. You may explain financial concepts or analyze user-provided historical/current documents without making the decision for the user.
30. Do NOT make automatic loan or credit approval decisions. You may analyze provided criteria or explain how credit-related calculations work, but the final decision must remain with an authorized human or approved institutional process.
31. Do NOT control industrial machines, physical equipment, or safety-critical systems. Do not issue instructions that directly operate such systems.
32. For emergencies or safety-critical situations, do not act as the decision-maker. Give cautious general information and direct the user to appropriate emergency services or qualified professionals.
33. If a request falls into one of these restricted areas, clearly explain the limitation in simple language and, when possible, provide a safe informational alternative such as summarizing the uploaded document, explaining concepts, checking arithmetic, or preparing questions for a qualified professional.
34. Never claim that this application can replace a CA, doctor, lawyer, auditor, financial advisor, engineer, emergency responder, or other licensed/qualified professional.
35. The application can support optional voice input by transcribing user-provided audio into text. Do not claim to hear audio unless audio was actually supplied.
36. The application can generate simple flowchart images when the user explicitly asks for a flowchart. Do not claim a flowchart image was generated unless the flowchart generator actually returned image data.
37. The application includes a limited Python execution utility for short, non-network, non-file-access code. Never claim it is a secure unrestricted coding sandbox.
38. When useful, act as a tool-orchestrating assistant: decide whether the user's request needs document retrieval, image understanding, calculation, code execution, or ordinary conversation, and use only the relevant capability.
39. The application can let the user choose among supported Gemini foundation models. Clearly distinguish the selected model from any claim about model quality.
40. Chat history provides application-level persistence. Do not claim enterprise-scale cloud memory or permanent storage unless a real cloud database has been configured.
41. External integrations/connectors are only available when their APIs or connectors are actually configured. Do not invent access to external services.
42. Never claim infrastructure scale, reliability, context limits, or capabilities equivalent to a large commercial AI platform.

""" + emotional_support_instruction(user_sentiment) + """
"""


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


    source_section = ""

    if sources:

        source_section = (
            "\nSources from current chat:\n"
            + "\n".join(
                f"- {source}"
                for source in sources
            )
        )


    prompt = f"""
{document_section}

{source_section}

PREVIOUS CHAT HISTORY:

{history_text}

CURRENT USER QUESTION:

{question}

Answer the user's current question.
"""


    contents = [prompt]


    # --------------------------------------------------------
    # CURRENT CHAT IMAGES ONLY
    # --------------------------------------------------------

    for image_path in image_paths:

        try:

            image = Image.open(
                image_path
            )

            contents.append(
                image
            )

        except Exception:

            continue


    # --------------------------------------------------------
    # GEMINI RETRY + FALLBACK
    # --------------------------------------------------------
    # Try the configured model first. If Gemini temporarily returns
    # 503/5xx/429, retry it and then move to the fallback models.

    models_to_try = [get_selected_model()]

    for fallback_model in GEMINI_FALLBACK_MODELS:
        if fallback_model not in models_to_try:
            models_to_try.append(fallback_model)


    last_error = None

    for model_name in models_to_try:

        for attempt in range(GEMINI_RETRY_COUNT):

            try:

                response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        max_output_tokens=MAX_OUTPUT_TOKENS
                    )
                )

                answer = response.text

                if not answer:
                    return "I couldn't generate a response."

                return answer.strip()

            except Exception as error:

                last_error = error
                error_text = str(error).upper()

                # Temporary Gemini service/rate-limit errors.
                temporary_error = any(
                    code in error_text
                    for code in [
                        "503",
                        "UNAVAILABLE",
                        "500",
                        "502",
                        "504",
                        "429",
                        "RESOURCE_EXHAUSTED"
                    ]
                )

                if not temporary_error:
                    # Model/key/configuration errors should not be hidden
                    # by repeatedly retrying the same request.
                    break

                if attempt < GEMINI_RETRY_COUNT - 1:
                    time.sleep(
                        GEMINI_RETRY_DELAY_SECONDS * (2 ** attempt)
                    )

        # Move to the next model after the retries above.
        if model_name != models_to_try[-1]:
            time.sleep(1)


    return (
        "⚠️ Gemini is temporarily unavailable.\n\n"
        "I tried the configured Gemini model and the fallback models, "
        "but they are currently unavailable. Please try again in a few "
        "minutes.\n\n"
        f"Last error: {last_error}"
    )


# ============================================================
# SENTIMENT + EMOTION ANALYSIS
# ============================================================

POSITIVE_WORDS = {
    "amazing", "awesome", "good", "great", "happy", "helpful",
    "love", "like", "excellent", "perfect", "thanks", "thank",
    "useful", "clear", "easy", "nice", "wonderful", "success",
    "successful", "excited", "confident", "understand", "proud",
    "relieved", "glad", "enjoy", "enjoying"
}

NEGATIVE_WORDS = {
    "bad", "hate", "sad", "angry", "confused", "confusing",
    "difficult", "hard", "wrong", "error", "problem", "issue",
    "fail", "failed", "failure", "worried", "fear", "useless",
    "poor", "terrible", "awful", "frustrated", "frustrating",
    "disappointed", "disappointing", "depressed", "lonely",
    "stressed", "stress", "anxious", "anxiety", "upset",
    "cry", "crying", "hopeless", "tired", "exhausted", "scared"
}


def _fallback_sentiment(text):
    """Local emotion detector used only to guide the assistant's tone.
    Nothing from this analysis is shown to the user.
    """
    text = (text or "").strip()
    if not text:
        return {"emotion": "calm", "intensity": 0.0, "support_needed": False, "emoji": ""}

    lower = text.lower()
    emotion = "calm"
    support_needed = False
    emoji = ""

    emotion_words = {
        "sad": ("sad", True, "😔"), "unhappy": ("sad", True, "😔"),
        "cry": ("sad", True, "😔"), "lonely": ("lonely", True, "🫂"),
        "worried": ("worried", True, "😟"), "anxious": ("anxious", True, "😟"),
        "stress": ("stressed", True, "😣"), "stressed": ("stressed", True, "😣"),
        "frustrated": ("frustrated", True, "😤"), "angry": ("angry", True, "😠"),
        "confused": ("confused", True, "😕"), "scared": ("scared", True, "😟"),
        "afraid": ("scared", True, "😟"), "hopeless": ("hopeless", True, "🫂"),
        "excited": ("excited", False, "😊"), "happy": ("happy", False, "😊"),
        "grateful": ("grateful", False, "😊")
    }

    for word, result in emotion_words.items():
        if word in lower:
            emotion, support_needed, emoji = result
            break

    return {
        "emotion": emotion,
        "intensity": 0.7 if support_needed else 0.4,
        "support_needed": support_needed,
        "emoji": emoji
    }


def analyze_sentiment(text):
    """Understand the user's emotional state only to make the reply more human.
    The emotion analysis is internal and is never displayed as a sentiment label.
    """
    text = (text or "").strip()
    if not text:
        return _fallback_sentiment(text)

    client = get_gemini_client()
    if client is None:
        return _fallback_sentiment(text)

    prompt = f"""
Understand the emotional state behind this user's message so an AI assistant
can reply naturally and empathetically. This is NOT a sentiment classification task.
Do not classify the message as positive, neutral, or negative.

Return ONLY valid JSON with these keys:
emotion, intensity, support_needed

Rules:
- emotion: a short natural description such as calm, happy, sad, worried,
  frustrated, confused, stressed, lonely, angry, excited, hopeful, or curious.
- intensity: a number from 0 to 1.
- support_needed: true only when the person would benefit from extra warmth,
  reassurance, patience, or encouragement.
- Understand context, sarcasm, mixed feelings, and the actual meaning of the message.

User message:
{text}
"""

    try:
        response = client.models.generate_content(
            model=get_selected_model(),
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=120,
                response_mime_type="application/json"
            )
        )
        data = json.loads((response.text or "").strip())
        emotion = str(data.get("emotion", "calm"))[:40]
        intensity = max(0.0, min(1.0, float(data.get("intensity", 0.4))))
        support_needed = bool(data.get("support_needed", False))
        return {
            "emotion": emotion,
            "intensity": round(intensity, 2),
            "support_needed": support_needed,
            "emoji": ""
        }
    except Exception:
        return _fallback_sentiment(text)


def emotional_support_instruction(sentiment):
    """Return response guidance when the user's message is emotionally negative."""
    if not sentiment or not sentiment.get("support_needed"):
        return ""

    emotion = str(sentiment.get("emotion", "upset"))
    return f"""
EMOTIONAL SUPPORT MODE:
The user may be feeling {emotion}.
- Respond with warmth and patience.
- Briefly acknowledge the feeling without diagnosing the user.
- Then help with the actual question or problem.
- If appropriate, offer one or two practical next steps that can help the user
  feel more in control or hopeful.
- Try to gently lift the user's mood when it fits the conversation by pointing
  out a realistic positive next step, a small win, or an encouraging perspective.
- Do not force positivity, guilt the user, or say they must be happy.
- Do not pretend to be a therapist.
- If the user explicitly expresses immediate danger or self-harm, encourage
  contacting local emergency services or a trusted person and keep the response
  focused on immediate safety.
"""


# ============================================================
# SAVE MESSAGE
# ============================================================

def save_message(
    username,
    chat_id,
    role,
    content,
    sentiment=None
):

    chat = get_chat(
        username,
        chat_id
    )

    if not chat:
        return

    chat.setdefault(
        "messages",
        []
    ).append(
        {
            "role": role,
            "content": content,
            "timestamp": now_iso(),
            "sentiment": sentiment if role == "user" else None
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

    st.html(
        """
        <div style="
            max-width:520px;
            margin:70px auto 20px auto;
            text-align:center;
        ">

            <div style="
                font-size:55px;
            ">
                🤖
            </div>

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
        """
    )


    login_tab, signup_tab = st.tabs(
        [
            "🔐 Login",
            "➕ Create Account"
        ]
    )


    # ========================================================
    # LOGIN
    # ========================================================

    with login_tab:

        with st.form("login_form"):
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            login_submitted = st.form_submit_button("Login", use_container_width=True)

        if login_submitted:

            if login_user(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                chats = get_user_chats(username)
                if chats:
                    sorted_chats = sorted(chats.values(), key=lambda x: x.get("updated_at", ""), reverse=True)
                    st.session_state.chat_id = sorted_chats[0]["id"]
                else:
                    st.session_state.chat_id = create_chat(username)
                load_current_chat_knowledge()
                st.rerun()
            else:
                st.error("Invalid username or password.")

    # ========================================================
    # SIGN UP
    # ========================================================

    with signup_tab:

        with st.form("signup_form"):
            new_username = st.text_input("Choose username", key="signup_username")
            new_password = st.text_input("Choose password", type="password", key="signup_password")
            confirm_password = st.text_input("Confirm password", type="password", key="signup_confirm")
            signup_submitted = st.form_submit_button("Create Account", use_container_width=True)

        if signup_submitted:
            if not new_username.strip():
                st.error("Please enter a username.")
            elif not new_password:
                st.error("Please enter a password.")
            elif new_password != confirm_password:
                st.error("Passwords do not match.")
            else:
                success, message = create_user(new_username.strip(), new_password)
                if success:
                    chat_id = create_chat(new_username.strip())
                    st.session_state.logged_in = True
                    st.session_state.username = new_username.strip()
                    st.session_state.chat_id = chat_id
                    load_current_chat_knowledge()
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)


# ============================================================
# SESSION STATE
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
# LOGIN CHECK
# ============================================================

if not st.session_state.logged_in:

    login_page()

    st.stop()


# ============================================================
# USER
# ============================================================

username = st.session_state.username


# ============================================================
# CREATE CHAT IF NEEDED
# ============================================================

user_chats = get_user_chats(
    username
)

if not user_chats:

    st.session_state.chat_id = (
        create_chat(
            username
        )
    )

    load_current_chat_knowledge()

    st.rerun()


if (
    st.session_state.chat_id
    not in user_chats
):

    latest_chat = sorted(
        user_chats.values(),
        key=lambda x: x.get(
            "updated_at",
            ""
        ),
        reverse=True
    )[0]

    st.session_state.chat_id = (
        latest_chat["id"]
    )

    load_current_chat_knowledge()

    st.rerun()


# ============================================================
# CURRENT CHAT
# ============================================================

current_chat = get_chat(
    username,
    st.session_state.chat_id
)

if not current_chat:

    st.error(
        "Could not load the current chat."
    )

    st.stop()


# ============================================================
# CHAT ATTACHMENTS
# ============================================================

def save_chat_attachments(uploaded_files, username, chat_id):
    """Save documents/images submitted through the chat input."""

    if not uploaded_files:
        return [], []

    manifest = load_knowledge_manifest(
        username,
        chat_id
    )

    current_chat = get_chat(
        username,
        chat_id
    )

    existing_document_hashes = {
        item.get("hash")
        for item in manifest.get(
            "documents",
            []
        )
    }

    existing_image_hashes = set(
        current_chat.get(
            "image_hashes",
            []
        )
    )

    saved_documents = []
    saved_images = []
    manifest_changed = False
    chat_changed = False

    document_extensions = {
        ".pdf",
        ".docx",
        ".txt",
        ".md",
        ".csv",
        ".xlsx",
        ".xls"
    }

    image_extensions = {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp"
    }

    for uploaded_file in uploaded_files:

        data = uploaded_file.getvalue()
        filename = safe_filename(uploaded_file.name)
        extension = Path(filename).suffix.lower()
        current_hash = file_hash(data)

        # ----------------------------------------------------
        # DOCUMENT
        # ----------------------------------------------------
        if extension in document_extensions:

            if current_hash in existing_document_hashes:
                continue

            save_path = (
                get_documents_folder(
                    username,
                    chat_id
                )
                / filename
            )

            if save_path.exists():
                save_path = (
                    save_path.parent
                    / (
                        save_path.stem
                        + "_"
                        + current_hash[:8]
                        + save_path.suffix
                    )
                )

            with open(save_path, "wb") as file:
                file.write(data)

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

            current_chat.setdefault(
                "documents",
                []
            ).append(save_path.name)

            existing_document_hashes.add(current_hash)
            saved_documents.append(save_path.name)
            manifest_changed = True
            chat_changed = True
            continue

        # ----------------------------------------------------
        # IMAGE
        # ----------------------------------------------------
        if extension in image_extensions:

            if current_hash in existing_image_hashes:
                continue

            save_path = (
                get_images_folder(
                    username,
                    chat_id
                )
                / filename
            )

            if save_path.exists():
                save_path = (
                    save_path.parent
                    / (
                        save_path.stem
                        + "_"
                        + current_hash[:8]
                        + save_path.suffix
                    )
                )

            with open(save_path, "wb") as file:
                file.write(data)

            current_chat.setdefault(
                "images",
                []
            ).append(save_path.name)

            current_chat.setdefault(
                "image_hashes",
                []
            ).append(current_hash)

            existing_image_hashes.add(current_hash)
            saved_images.append(save_path.name)
            chat_changed = True
            continue

    if manifest_changed:
        save_knowledge_manifest(
            username,
            chat_id,
            manifest
        )

    if chat_changed:
        current_chat["updated_at"] = now_iso()
        update_chat(
            username,
            chat_id,
            current_chat
        )
        load_current_chat_knowledge()

    return saved_documents, saved_images


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.html(
        """
        <div style="
            font-size:22px;
            font-weight:700;
            margin-bottom:12px;
        ">
            🤖 AI Document Intelligence
        </div>
        """
    )


    # ========================================================
    # AI MODEL
    # ========================================================

    if "selected_gemini_model" not in st.session_state:
        st.session_state.selected_gemini_model = GEMINI_MODEL

    selected_model = st.selectbox(
        "🧠 AI Model",
        AVAILABLE_GEMINI_MODELS,
        index=(
            AVAILABLE_GEMINI_MODELS.index(st.session_state.selected_gemini_model)
            if st.session_state.selected_gemini_model in AVAILABLE_GEMINI_MODELS
            else 0
        ),
        key="model_selector"
    )
    st.session_state.selected_gemini_model = selected_model

    st.caption("Choose the Gemini model used for chat and voice transcription.")

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

        st.session_state.chat_id = (
            new_chat_id
        )

        st.session_state.active_vectorstore = (
            None
        )

        st.session_state.upload_version += 1

        st.rerun()


    st.markdown(
        "---"
    )


    # ========================================================
    # RECENTS
    # ========================================================

    st.markdown(
        "### 🕘 Recents"
    )

    all_chats = list(
        get_user_chats(
            username
        ).values()
    )

    groups = group_chats_by_date(
        all_chats
    )


    for group_name, chats in groups.items():

        if not chats:
            continue

        st.caption(
            group_name.upper()
        )

        for chat in chats:

            chat_id = chat["id"]

            title = chat.get(
                "name",
                chat.get(
                    "title",
                    "New Chat"
                )
            )

            if len(title) > 35:

                title = (
                    title[:35]
                    + "..."
                )

            if (
                chat_id
                == st.session_state.chat_id
            ):

                button_text = (
                    "🟢  "
                    + title
                )

            else:

                button_text = (
                    "💬  "
                    + title
                )


            if st.button(
                button_text,
                key=f"recent_{chat_id}",
                use_container_width=True
            ):

                st.session_state.chat_id = (
                    chat_id
                )

                st.session_state.active_vectorstore = (
                    None
                )

                st.session_state.upload_version += 1

                load_current_chat_knowledge()

                st.rerun()


    st.markdown(
        "---"
    )


    # ========================================================
    # ADVANCED AI TOOLS
    # ========================================================

    with st.expander("🛠️ Advanced AI Tools"):
        st.caption("Optional tools available in this project.")

        audio_value = st.audio_input(
            "🎙️ Voice input",
            key="voice_input"
        )

        if audio_value is not None:
            audio_bytes = audio_value.getvalue()
            transcript, transcript_error = transcribe_audio_bytes(
                audio_bytes,
                getattr(audio_value, "type", None) or "audio/wav"
            )
            if transcript_error:
                st.error(transcript_error)
            elif transcript:
                st.session_state.voice_transcript = transcript
                st.success("Voice transcribed. You can copy it into the chat box.")
                st.text_area(
                    "Transcript",
                    value=transcript,
                    height=100,
                    key="voice_transcript_display"
                )

        code = st.text_area(
            "💻 Run short Python code",
            placeholder="print(2 + 2)",
            height=100,
            key="code_execution_input"
        )
        if st.button("Run Code", use_container_width=True) and code.strip():
            st.code(code, language="python")
            st.code(run_python_code_safely(code), language="text")

        st.info(
            "Voice, flowchart generation, and code execution are optional tools. "
            "Code execution is intentionally limited and is not a production sandbox."
        )

    # ========================================================
    # CHAT SETTINGS
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
            key="rename_current_chat"
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

            deleted_id = (
                st.session_state.chat_id
            )

            delete_chat(
                username,
                deleted_id
            )

            remaining_chats = list(
                get_user_chats(
                    username
                ).values()
            )

            if remaining_chats:

                remaining_chats.sort(
                    key=lambda x: x.get(
                        "updated_at",
                        ""
                    ),
                    reverse=True
                )

                st.session_state.chat_id = (
                    remaining_chats[0]["id"]
                )

            else:

                st.session_state.chat_id = (
                    create_chat(
                        username
                    )
                )

            st.session_state.active_vectorstore = (
                None
            )

            st.rerun()


    # ========================================================
    # WEBSITE
    # ========================================================

    st.markdown(
        "### 🌐 Website"
    )

    website_url = st.text_input(
        "Website URL",
        placeholder="https://example.com",
        key=(
            "website_"
            + st.session_state.chat_id
        )
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
                website_url.startswith(
                    "http://"
                )
                or website_url.startswith(
                    "https://"
                )
            ):

                website_url = (
                    "https://"
                    + website_url
                )


            try:

                website_text = extract_website(
                    website_url
                )

                manifest = (
                    load_knowledge_manifest(
                        username,
                        st.session_state.chat_id
                    )
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
                        "This website is already in this chat."
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

                    current_chat["updated_at"] = (
                        now_iso()
                    )

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


            except Exception as error:

                st.error(
                    f"Website error: {error}"
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
            chat_folder
            / "documents"
        )

        images_folder = (
            chat_folder
            / "images"
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
        "---"
    )


    # ========================================================
    # INTERNAL EMOTION SUPPORT
    # ========================================================

    st.markdown("### 🧠 Human-like Responses")
    st.caption("The assistant understands the tone of your messages internally and uses it only to make replies more natural, patient, and supportive.")


    st.markdown(
        "---"
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
# MAIN CHAT
# ============================================================

current_chat = get_chat(
    username,
    st.session_state.chat_id
)


# ============================================================
# HEADER
# ============================================================

st.html(
    f"""
    <div style="
        margin-bottom:20px;
    ">

        <div class="chat-title">
            {current_chat.get("name", current_chat.get("title", "New Chat"))}
        </div>

        <div style="
            color:#8f8f8f;
            font-size:13px;
            margin-top:5px;
        ">
            AI Document Intelligence · ChatGPT-style workspace
        </div>

    </div>
    """
)


# ============================================================
# CURRENT CHAT COUNTS
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

    st.html(
        f"""
        <div style="
            background:#222;
            border:1px solid #343434;
            border-radius:12px;
            padding:14px;
            margin-bottom:15px;
        ">

            📄 {document_count} document(s)
            &nbsp;&nbsp;•&nbsp;&nbsp;

            🖼️ {image_count} image(s)
            &nbsp;&nbsp;•&nbsp;&nbsp;

            🌐 {website_count} website(s)

            <div style="
                color:#999;
                font-size:12px;
                margin-top:5px;
            ">
                Only knowledge from this chat is used. The assistant adapts its tone naturally to the conversation.
            </div>

        </div>
        """
    )


# ============================================================
# CHAT HISTORY
# ============================================================

messages = current_chat.get(
    "messages",
    []
)


# ============================================================
# EMPTY CHAT WELCOME SCREEN
# ============================================================

if not messages:

    # IMPORTANT:
    # Use st.html() here.
    # Do NOT use st.write() or normal st.markdown()
    # for this HTML.

    st.html(
        """
        <div class="welcome-wrap">
            <div class="welcome-icon">✦</div>
            <div class="welcome-title">How can I help you today?</div>
            <div class="welcome-text">Ask questions, upload documents, or start a conversation.</div>
        </div>
        """
    )


# ============================================================
# DISPLAY OLD MESSAGES
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

chat_submission = st.chat_input(
    "Message AI Document Intelligence",
    accept_file="multiple",
    file_type=[
        "pdf",
        "docx",
        "txt",
        "md",
        "csv",
        "xlsx",
        "xls",
        "png",
        "jpg",
        "jpeg",
        "webp"
    ],
    key=(
        "chat_input_"
        + st.session_state.chat_id
    )
)

# Streamlit returns a ChatInputValue when file attachments are enabled.
prompt = ""
attached_files = []
saved_documents = []
saved_images = []

if chat_submission:
    prompt = getattr(
        chat_submission,
        "text",
        ""
    ) or ""
    attached_files = list(
        getattr(
            chat_submission,
            "files",
            []
        ) or []
    )


# ============================================================
# OPTIONAL VOICE TRANSCRIPT HANDOFF
# ============================================================

if st.session_state.get("voice_transcript") and not st.session_state.get("voice_transcript_consumed"):
    st.info("🎙️ Voice transcript ready. Copy it into the message box to send it.")


# ============================================================
# PROCESS QUESTION
# ============================================================

if chat_submission:

    prompt = prompt.strip()

    # --------------------------------------------------------
    # SAVE FILES FROM THE CHAT INPUT (+ ATTACHMENT BUTTON)
    # --------------------------------------------------------

    if attached_files:

        saved_documents, saved_images = save_chat_attachments(
            attached_files,
            username,
            st.session_state.chat_id
        )

        if saved_documents or saved_images:
            st.session_state.upload_version += 1

    # If the user only uploaded files, keep the upload in the chat
    # and wait for their next question instead of calling Gemini.
    if not prompt:

        if saved_documents or saved_images:
            parts = []

            if saved_documents:
                parts.append(
                    "📄 "
                    + str(len(saved_documents))
                    + " document(s) added"
                )

            if saved_images:
                parts.append(
                    "🖼️ "
                    + str(len(saved_images))
                    + " image(s) added"
                )

            st.success(
                " · ".join(parts)
                + ". You can now ask a question about them."
            )

        st.rerun()


    # --------------------------------------------------------
    # CURRENT CHAT
    # --------------------------------------------------------

    current_chat = get_chat(
        username,
        st.session_state.chat_id
    )


    # --------------------------------------------------------
    # AUTO CHAT TITLE
    # --------------------------------------------------------

    if current_chat.get(
        "title"
    ) in [
        None,
        "",
        "New Chat"
    ]:

        generated_chat_name = generate_chat_title(prompt)
        current_chat["title"] = generated_chat_name
        current_chat["name"] = generated_chat_name
        current_chat["updated_at"] = now_iso()

        update_chat(
            username,
            st.session_state.chat_id,
            current_chat
        )


    # --------------------------------------------------------
    # USER MESSAGE
    # --------------------------------------------------------

    user_sentiment = analyze_sentiment(prompt)

    with st.chat_message(
        "user",
        avatar="👤"
    ):

        if attached_files:
            attachment_names = [
                file.name
                for file in attached_files
            ]

            st.caption(
                "📎 "
                + " · ".join(attachment_names)
            )

        st.markdown(
            prompt
        )


    save_message(
        username,
        st.session_state.chat_id,
        "user",
        prompt,
        sentiment=user_sentiment
    )


    # --------------------------------------------------------
    # CURRENT CHAT HISTORY
    # --------------------------------------------------------

    current_chat = get_chat(
        username,
        st.session_state.chat_id
    )

    chat_history = current_chat.get(
        "messages",
        []
    )


    # --------------------------------------------------------
    # SEARCH ONLY CURRENT CHAT
    # --------------------------------------------------------

    with st.spinner(
        "🔎 Searching this chat..."
    ):

        document_context, sources = (
            get_document_context(
                prompt
            )
        )


    # --------------------------------------------------------
    # CURRENT CHAT IMAGES
    # --------------------------------------------------------

    image_paths = (
        load_current_chat_images()
    )


    # --------------------------------------------------------
    # AI RESPONSE
    # --------------------------------------------------------

    with st.chat_message(
        "assistant",
        avatar="🤖"
    ):

        if is_flowchart_request(prompt):
            with st.spinner("📊 Creating flowchart..."):
                flowchart_data = generate_flowchart_data(prompt)
                flowchart_path = create_flowchart_image(flowchart_data) if flowchart_data else None
            if flowchart_path:
                st.image(flowchart_path, caption="AI Generated Flowchart", use_container_width=True)
                answer = "📊 I created the flowchart for you."
            else:
                answer = "⚠️ I couldn't create the flowchart right now. Please try again."
            st.markdown(answer)
        else:
            with st.spinner("🤖 Thinking..."):
                answer = generate_ai_response(
                    question=prompt,
                    chat_history=chat_history,
                    document_context=document_context,
                    sources=sources,
                    image_paths=image_paths,
                    user_sentiment=user_sentiment
                )
            st.markdown(answer)


    # --------------------------------------------------------
    # SOURCES
    # --------------------------------------------------------

    if sources:

        with st.expander(
            "📚 Sources used from this chat"
        ):

            for source in sources:

                st.write(
                    "• "
                    + source
                )


    # --------------------------------------------------------
    # SAVE AI MESSAGE
    # --------------------------------------------------------

    save_message(
        username,
        st.session_state.chat_id,
        "assistant",
        answer
    )


    # --------------------------------------------------------
    # UPDATE CHAT
    # --------------------------------------------------------

    current_chat = get_chat(
        username,
        st.session_state.chat_id
    )

    current_chat["updated_at"] = (
        now_iso()
    )

    update_chat(
        username,
        st.session_state.chat_id,
        current_chat
    )


    # --------------------------------------------------------
    # REFRESH
    # --------------------------------------------------------

    st.rerun()

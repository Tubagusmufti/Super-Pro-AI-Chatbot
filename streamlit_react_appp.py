import os
import logging
import time
import sqlite3
import uuid
import glob
import json
import re
from datetime import datetime
from typing import TypedDict, List, Optional, Any
from dataclasses import dataclass

os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"
logging.getLogger('absl').setLevel(logging.ERROR)

import streamlit as st
import pandas as pd

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, END

from tenacity import retry, stop_after_attempt, wait_exponential

from dotenv import load_dotenv
load_dotenv()

def cleanup_temp_files():
    temp_files = glob.glob("temp_*.*")
    for file_path in temp_files:
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"⚠️ Gagal hapus {file_path}: {e}")

CFG = {
    "model": "gemini-2.0-flash",
    "temperature": 0.7,
    "retry_attempts": 3,
    "max_hops": 2,
    "deny_list": [
        "bunuh", "bom", "hack", "crack", "porno", "kontol", "memek",
        "goblok", "bangsat", "anjing", "setan", "iblis"
    ]
}

class AgentState(TypedDict):
    session_id: str
    goal: str
    context: str
    memory: List[str]
    tools_used: List[str]
    messages: List[Any]
    final_answer: str
    token_count: int
    needs_tool: bool
    needs_rag: bool
    needs_human: bool

DB_FILE = "chat_history.db"
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS conversations
                 (session_id TEXT, message_id TEXT, role TEXT, content TEXT, timestamp DATETIME)""")
    c.execute("""CREATE TABLE IF NOT EXISTS user_profiles
                 (session_id TEXT PRIMARY KEY, name TEXT, language TEXT, last_activity DATETIME)""")
    conn.commit()
    conn.close()

def save_conversation(session_id: str, messages: List[dict]):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("DELETE FROM conversations WHERE session_id = ?", (session_id,))
    for msg in messages:
        c.execute("INSERT INTO conversations VALUES (?,?,?,?,?)",
                  (session_id, str(uuid.uuid4()), msg["role"], msg["content"], datetime.now()))
    conn.commit()
    conn.close()

def load_conversation(session_id: str) -> List[dict]:
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT role, content FROM conversations WHERE session_id = ? ORDER BY timestamp", (session_id,))
    rows = c.fetchall()
    conn.close()
    return [{"role": r[0], "content": r[1]} for r in rows]

def save_user_profile(session_id: str, name: str, language: str):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""INSERT OR REPLACE INTO user_profiles
                 (session_id, name, language, last_activity) VALUES (?,?,?,?)""",
              (session_id, name, language, datetime.now()))
    conn.commit()
    conn.close()

def load_user_profile(session_id: str) -> dict:
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT name, language FROM user_profiles WHERE session_id = ?", (session_id,))
    row = c.fetchone()
    conn.close()
    if row:
        return {"name": row[0], "language": row[1]}
    return {"name": "", "language": "id"}

@dataclass
class Guard:
    @staticmethod
    def mask_pii(text: str) -> str:
        return re.sub(r'\S+@\S+', '<EMAIL>', text)

    @staticmethod
    def deny_list(text: str) -> bool:
        return any(word in text.lower() for word in CFG["deny_list"])

@tool
@retry(stop=stop_after_attempt(CFG["retry_attempts"]),
       wait=wait_exponential(multiplier=1, min=4, max=10))
def web_search(query: str) -> str:
    """Mencari informasi terkini di web menggunakan DuckDuckGo."""
    search = DuckDuckGoSearchRun()
    result = search.run(query)
    return result[:1000] if result else "Tidak ada hasil pencarian."

@tool
@retry(stop=stop_after_attempt(CFG["retry_attempts"]),
       wait=wait_exponential(multiplier=1, min=4, max=10))
def wikipedia_search(query: str) -> str:
    """Mencari informasi ensiklopedis di Wikipedia."""
    search = WikipediaQueryRun()
    result = search.run(query)
    return result[:1000] if result else "Tidak ada hasil pencarian Wikipedia."

tools = [web_search, wikipedia_search]
tool_dict = {tool.name: tool for tool in tools}


def get_llm(temperature=0):
    """
    Membuat instance ChatGoogleGenerativeAI.
    """
    try:
        model_to_use = st.session_state.get("model", CFG["model"])
        # Pastikan API key ada
        if not st.session_state.get("google_api_key"):
            st.error("❌ API Key Google tidak ditemukan. Silakan masukkan di sidebar.")
            print("💥 API Key Google tidak ditemukan.")
            return None

        llm_instance = ChatGoogleGenerativeAI(
            model=model_to_use,
            temperature=temperature,
            google_api_key=st.session_state.google_api_key
        )
        return llm_instance
    except Exception as e:
        error_msg = f"Error creating LLM: {str(e)}"
        st.error(error_msg)
        print(f"💥 {error_msg}")
        return None

@st.cache_resource(show_spinner=False)
def build_rag(file_path: str, file_type: str):
    try:
        if file_type == "pdf":
            loader = PyPDFLoader(file_path)
        elif file_type in ["txt", "md"]:
            loader = TextLoader(file_path)
        elif file_type == "docx":
            loader = Docx2txtLoader(file_path)
        else:
            print(f"❌ Tipe file tidak didukung: {file_type}")
            return None

        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = splitter.split_documents(docs)
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'device': 'cpu', 'batch_size': 128}
        )
        vectorstore = FAISS.from_documents(texts, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        return retriever
    except Exception as e:
        error_msg = f"Error building RAG: {str(e)}"
        st.error(error_msg)
        print(f"💥 {error_msg}")
        return None

def planner_node(state: AgentState) -> AgentState:
    prompt = f"""Anda Planner. Pecah goal UTAMA berikut menjadi sub-task kecil (max 3):

Goal UTAMA: {state["goal"]}

Output: JSON list string."""
    llm = get_llm()
    if llm:
        try:
            out = llm.invoke(state["messages"] + [HumanMessage(content=prompt)]).content
            state["context"] = out
            state["tools_used"].append("planner")
        except Exception as e:
            error_msg = f"Error in planner: {str(e)}"
            state["final_answer"] = error_msg
            print(f"💥 {error_msg}")
    return state

def router_node(state: AgentState) -> AgentState:
    user_name = st.session_state.user_profile.get("name", "").strip()

    tool_web_search_enabled = st.session_state.get("tool_web_search", False)
    tool_wikipedia_enabled = st.session_state.get("tool_wikipedia", False)
    tool_rag_enabled = st.session_state.get("tool_rag", False) and st.session_state.get("retriever") is not None

    document_keywords = [
        "isi dokumen", "dokumen ini", "file ini", "ringkasan dokumen",
        "rangkuman dokumen", "konten dokumen", "dokumen yang saya upload",
        "dokumen yang saya unggah", "apa isinya", "ceritakan isi", "jelaskan isi",
        "dokumen", "file", "pdf", "upload", "unggah",
        "apa isi", "isi dari", "apa yang ada di", "apa isi pdf", "isi pdf"
    ]

    goal_lower = state["goal"].strip().lower()

    code_keywords = [
        "buat", "buatkan", "code", "kode", "script", "program", "fungsi", "function",
        "python", "java", "javascript", "c++", "c#", "php", "ruby", "go", "rust",
        "tabel", "table", "perbandingan", "bandingkan",
        "list", "daftar", "bisa kamu", "bisakah kamu", "please create", "buat tabel",
        "coding", "pemrograman", "syntax", "sintaks", "algoritma", "hitung", "kalkulasi"
    ]
    is_creation_request = any(kw in goal_lower for kw in code_keywords)

    if tool_rag_enabled and any(kw in goal_lower for kw in document_keywords):
        state["needs_rag"] = True
        state["needs_tool"] = False
        state["needs_human"] = False
        state["tools_used"].append("router_rag_direct")
        return state

    casual_keywords = [
        "halo", "hai", "hello", "hi", "apa kabar", "kabar", "siapa kamu",
        "berapa", "hitung", "kalkulasi", "matematika", "soal", "pecahkan",
        "makasih", "terima kasih", "thanks", "oke", "ya", "tidak", "pagi", "siang", "sore", "malam",
        "bolehkah", "bisa saya", "kasih nama", "namai", "panggil", "sebut", "perkenalkan", "kenalan",
        "tolong", "bantu", "minta", "ajarkan", "ceritakan", "jelaskan", "tunjukkan", "mau tanya"
    ]

    is_casual = any(kw in goal_lower for kw in casual_keywords) or len(goal_lower.split()) <= 7
    math_patterns = ["+", "-", "*", "/", "=", "^", "√"]
    is_math = any(symbol in state["goal"] for symbol in math_patterns)

    if is_casual or is_math or is_creation_request:
        state["needs_rag"] = False
        state["needs_tool"] = False
        state["needs_human"] = False
        state["tools_used"].append("router_direct")

        if is_creation_request:
            system_prompt = f"""Anda adalah asisten AI yang sangat baik dalam membuat kode dan konten.
Instruksi:
1.  Jika diminta membuat kode, berikan kode tersebut sebagai teks biasa dalam blok kode Markdown (contoh: ```python ... ```).
2.  Jangan pernah menampilkan kode dalam bentuk tabel atau struktur data lainnya kecuali diminta secara eksplisit.
3.  Jika diminta membuat daftar atau tabel informasi (bukan kode), Anda boleh menggunakannya.
4.  Jawab permintaan secara langsung dan jelas.
5.  Gunakan bahasa Indonesia yang baik dan benar."""
        elif user_name and any(kw in goal_lower for kw in ["halo", "hai", "pagi", "siang", "sore", "malam"]):
            system_prompt = f"""Anda asisten AI yang ramah, personal, dan suka menyapa dengan nama.
Nama pengguna: {user_name}"""
        else:
            system_prompt = "Anda asisten AI pintar. Jawab langsung & akurat."

        messages_with_context = [HumanMessage(content=system_prompt)] + state["messages"]
        llm = get_llm(temperature=0.3)
        if llm:
            try:
                answer = llm.invoke(messages_with_context).content
                state["final_answer"] = answer
            except Exception as e:
                 error_msg = f"Error saat menghasilkan jawaban langsung: {str(e)}"
                 print(f"💥 {error_msg}")
                 state["final_answer"] = f"Maaf, terjadi kesalahan: {error_msg}"
        return state

    if tool_rag_enabled:
        prompt = f"""Anda Router. Goal: {state["goal"]}
Pilih: direct | rag | tool | human. Output satu kata.
Gunakan 'rag' HANYA jika pertanyaan spesifik terkait dokumen yang diunggah.
Gunakan 'tool' HANYA jika butuh info terkini dan tool diaktifkan."""
    else:
        prompt = f"""Anda Router. Goal: {state["goal"]}
Pilih: direct | tool | human. Output satu kata.
RAG dinonaktifkan pengguna. Gunakan 'tool' HANYA jika butuh info terkini dan tool diaktifkan."""

    llm = get_llm()
    if llm:
        try:
            choice = llm.invoke([HumanMessage(content=prompt)]).content.strip().lower()
            state["needs_rag"] = "rag" in choice and tool_rag_enabled
            state["needs_tool"] = "tool" in choice and (tool_web_search_enabled or tool_wikipedia_enabled)
            state["needs_human"] = "human" in choice
            state["tools_used"].append("router")
        except Exception as e:
            error_msg = f"Error in router: {str(e)}"
            state["final_answer"] = error_msg
            print(f"💥 {error_msg}")
    return state

def rag_node(state: AgentState) -> AgentState:
    if not st.session_state.get("tool_rag", False):
        msg = "Fitur RAG dinonaktifkan oleh pengguna. Silakan aktifkan di sidebar jika ingin menggunakan dokumen."
        state["final_answer"] = msg
        state["needs_rag"] = False
        return state

    if "retriever" not in st.session_state or st.session_state.retriever is None:
        msg = "Tidak ada dokumen yang diunggah. Silakan unggah dokumen di sidebar."
        state["final_answer"] = msg
        state["needs_rag"] = False
        return state

    try:
        retriever = st.session_state.retriever
        docs = retriever.invoke(state["goal"])
        if not docs:
            msg = "Tidak menemukan informasi relevan di dokumen."
            state["final_answer"] = msg
            state["needs_rag"] = False
            return state

        context = "\n\n".join([d.page_content for d in docs])
        prompt = f"""Jawab berdasarkan konteks berikut:

{context}

Pertanyaan: {state["goal"]}"""

        messages_with_rag = state["messages"][:-1] + [HumanMessage(content=prompt)]
        llm = get_llm(temperature=CFG["temperature"])
        if llm:
            answer = llm.invoke(messages_with_rag).content
            state["final_answer"] = answer
            state["tools_used"].append("rag")
    except Exception as e:
        error_msg = f"Error RAG: {str(e)}"
        state["final_answer"] = error_msg
        print(f"💥 Error dalam rag_node: {error_msg}")
    return state

def tool_node(state: AgentState) -> AgentState:
    messages = state["messages"]
    if not messages:
        state["final_answer"] = "No messages."
        return state

    last_message = messages[-1]

    tool_web_search_enabled = st.session_state.get("tool_web_search", False)
    tool_wikipedia_enabled = st.session_state.get("tool_wikipedia", False)

    if hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
        outputs = []
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            if tool_name == "web_search" and not tool_web_search_enabled:
                msg = "Web Search dinonaktifkan oleh pengguna."
                outputs.append(ToolMessage(content=msg, name=tool_name, tool_call_id=tool_call["id"]))
                continue
            if tool_name == "wikipedia_search" and not tool_wikipedia_enabled:
                msg = "Wikipedia Search dinonaktifkan oleh pengguna."
                outputs.append(ToolMessage(content=msg, name=tool_name, tool_call_id=tool_call["id"]))
                continue

            if tool_name in tool_dict:
                try:
                    result = tool_dict[tool_name].invoke(tool_args)
                    outputs.append(ToolMessage(content=str(result), name=tool_name, tool_call_id=tool_call["id"]))
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    outputs.append(ToolMessage(content=error_msg, name=tool_name, tool_call_id=tool_call["id"]))
                    print(f"💥 Error tool {tool_name}: {error_msg}")
            else:
                msg = f"Tool '{tool_name}' not found."
                outputs.append(ToolMessage(content=msg, name=tool_name, tool_call_id=tool_call["id"]))
        state["messages"].extend(outputs)
        state["tools_used"].append("tool")
        if outputs:
            state["final_answer"] = outputs[-1].content
    else:
        if "router_direct" not in state.get("tools_used", []):
            try:
                if tool_web_search_enabled:
                    result = web_search.invoke(state["goal"])
                    state["final_answer"] = result
                    state["tools_used"].append("tool_default_search")
                elif tool_wikipedia_enabled:
                    result = wikipedia_search.invoke(state["goal"])
                    state["final_answer"] = result
                    state["tools_used"].append("tool_default_search")
                else:
                    msg = "Maaf, semua tool pencarian dinonaktifkan. Silakan aktifkan tool di sidebar."
                    state["final_answer"] = msg
            except Exception as e:
                error_msg = f"Error pencarian: {str(e)}"
                state["final_answer"] = error_msg
                print(f"💥 {error_msg}")
        else:
            state["final_answer"] = "Maaf, saya tidak dapat memproses ini."
    return state

def reflector_node(state: AgentState) -> AgentState:
    if not state.get("final_answer"):
        state["needs_rag"] = True
        state["tools_used"].append("reflector_loop")
        return state

    if any(tag in state.get("tools_used", []) for tag in ["router_direct", "router_rag_direct"]):
        state["needs_rag"] = False
        return state

    prompt = f"""Evaluasi jawaban terakhir Anda: "{state['final_answer']}".
Apakah sudah memenuhi goal pengguna? Jawab: Ya/Tidak."""

    messages_for_reflect = state["messages"] + [HumanMessage(content=prompt)]

    llm = get_llm()
    if llm:
        try:
            eval_response = llm.invoke(messages_for_reflect).content.strip().lower()
            reflection_count = sum(1 for tool in state["tools_used"] if tool == "reflector_loop")

            if "tidak" in eval_response and reflection_count < CFG["max_hops"]:
                state["needs_rag"] = True
                state["tools_used"].append("reflector_loop")
            else:
                state["needs_rag"] = False
        except Exception as e:
            state["needs_rag"] = False
            print(f"💥 Error refleksi: {e}")
    return state

def human_node(state: AgentState) -> AgentState:
    st.info("👨‍💻 Butuh bantuan manusia.")
    human_resp = st.text_input("Ketik 'ok' untuk lanjut: ", key=f"hitl_{uuid.uuid4()}")
    if human_resp and human_resp.lower() == "ok":
        state["needs_human"] = False
        state["final_answer"] = "✅ Lanjut memproses."
        state["tools_used"].append("human_intervention")
    return state

def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("planner", planner_node)
    graph.add_node("router", router_node)
    graph.add_node("rag", rag_node)
    graph.add_node("tool", tool_node)
    graph.add_node("reflector", reflector_node)
    graph.add_node("human", human_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "router")

    def router_to_next(state: AgentState):
        if state.get("final_answer") and any(tag in state.get("tools_used", []) for tag in ["router_direct", "router_rag_direct"]):
            return END
        elif state.get("needs_rag", False):
            return "rag"
        elif state.get("needs_tool", False):
            return "tool"
        elif state.get("needs_human", False):
            return "human"
        else:
            return "reflector"

    graph.add_conditional_edges("router", router_to_next, {END: END, "rag": "rag", "tool": "tool", "human": "human", "reflector": "reflector"})
    graph.add_edge("rag", "reflector")
    graph.add_edge("tool", "reflector")
    graph.add_edge("human", "reflector")
    graph.add_conditional_edges("reflector", lambda s: END if not s.get("needs_rag", False) else "rag", {END: END, "rag": "rag"})
    return graph

def render_sidebar():
    with st.sidebar:
        st.title("⚙️ Super-Pro")
        st.caption("AI Chatbot with RAG & Tools")

        with st.expander("👤 Profil", expanded=True):
            name = st.text_input("Nama", value=st.session_state.user_profile.get("name", ""))
            language = st.selectbox("Bahasa", ["id", "en"], index=0 if st.session_state.user_profile.get("language") == "id" else 1)
            if st.button("💾 Simpan", key="save_profile", use_container_width=True):
                save_user_profile(st.session_state.session_id, name, language)
                st.session_state.user_profile = {"name": name, "language": language}
                st.success("Profil disimpan!")

        st.markdown("---")
        with st.expander("🔑 Pengaturan Inti", expanded=True):
            api_key = st.text_input("Google API Key", type="password", value=st.session_state.get("google_api_key", ""), help="Dapatkan di Google AI Studio")
            system_prompt = st.text_area("System Prompt", value=st.session_state.get("system_prompt", "Anda adalah asisten AI yang ramah."), help="Instruksi awal untuk AI")

            st.markdown("**🧠 Model AI**")
            model_options = {
                "Gemini 2.0 Flash": "gemini-2.0-flash",
                "Gemini 1.5 Flash": "gemini-1.5-flash-latest",
                "Gemini 1.5 Pro": "gemini-1.5-pro-latest",
            }
            current_model = st.session_state.get("model", CFG["model"])
            if current_model not in model_options.values():
                current_model = CFG["model"]
            selected_label = st.selectbox(
                "Pilih model:",
                options=list(model_options.keys()),
                index=list(model_options.values()).index(current_model),
                key="model_selector",
                label_visibility="collapsed"
            )
            st.session_state.model = model_options[selected_label]

            if st.button("🚀 Terapkan", key="apply_core_settings", use_container_width=True):
                st.session_state.google_api_key = api_key
                st.session_state.system_prompt = system_prompt
                st.success("Pengaturan inti diterapkan!")

        st.markdown("---")
        with st.expander("📄 Dokumen & RAG", expanded=True):
            uploaded_file = st.file_uploader("Unggah Dokumen", type=["pdf", "txt", "md", "docx"], label_visibility="collapsed")

            if st.button("📤 Proses Dokumen", key="process_doc", use_container_width=True):
                if "uploaded_file" in st.session_state and st.session_state.uploaded_file:
                    old_path = st.session_state.uploaded_file.get("path")
                    if old_path and os.path.exists(old_path):
                        try:
                            os.remove(old_path)
                        except Exception as e:
                            print(f"⚠️ Gagal hapus file lama {old_path}: {e}")

                if uploaded_file:
                    file_ext = uploaded_file.name.split(".")[-1].lower()
                    path = f"temp_{uuid.uuid4()}.{file_ext}"

                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    status_text.caption("Mengunggah...")

                    with open(path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    progress_bar.progress(30)
                    status_text.caption("Memproses...")

                    st.session_state.uploaded_file = {"path": path, "type": file_ext}
                    retriever = build_rag(path, file_ext)

                    if retriever:
                        progress_bar.progress(100)
                        status_text.caption("✅ Siap!")
                        st.session_state.retriever = retriever
                        st.session_state.tool_rag = True
                        st.success("Dokumen siap digunakan!")
                    else:
                        progress_bar.progress(0)
                        status_text.caption("❌ Gagal")
                        st.error("Gagal memproses dokumen.")

                    time.sleep(0.5)
                    progress_bar.empty()
                    status_text.empty()
                else:
                     st.info("Tidak ada file yang diunggah.")

        st.markdown("---")
        with st.expander("🛠️ Tool", expanded=False):
            st.caption("Aktifkan tool tambahan:")
            col1, col2 = st.columns(2)
            with col1:
                st.session_state.tool_web_search = st.checkbox("🌐 Web Search", value=st.session_state.get("tool_web_search", False))
            with col2:
                st.session_state.tool_wikipedia = st.checkbox("📚 Wikipedia", value=st.session_state.get("tool_wikipedia", False))

            st.markdown("---")
            st.session_state.tool_rag = st.checkbox("📄 RAG (Gunakan Dokumen)", value=st.session_state.get("tool_rag", False) and bool(st.session_state.retriever))
            if st.session_state.tool_rag and not st.session_state.retriever:
                st.info("ℹ️ Aktifkan setelah unggah dokumen.")

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Reset Chat", use_container_width=True):
                st.session_state.messages = []
                save_conversation(st.session_state.session_id, [])
                st.rerun()
        with col2:
            if st.button("🧹 Reset Dokumen", use_container_width=True):
                if "uploaded_file" in st.session_state and st.session_state.uploaded_file:
                    temp_path = st.session_state.uploaded_file.get("path")
                    if temp_path and os.path.exists(temp_path):
                        try:
                            os.remove(temp_path)
                            print(f"🗑️ File dihapus saat reset: {temp_path}")
                        except Exception as e:
                            print(f"⚠️ Gagal hapus {temp_path}: {e}")
                    st.session_state.uploaded_file = None
                st.session_state.retriever = None
                st.session_state.tool_rag = False
                st.success("Dokumen direset.")
                st.rerun()

        st.markdown("---")
        with st.expander("📊 Status", expanded=False):
            st.write(f"**API Key:** {'✅' if st.session_state.get('google_api_key') else '❌'}")
            st.write(f"**Dokumen:** {'✅' if st.session_state.get('retriever') else '❌'}")
            st.write(f"**Nama:** {st.session_state.user_profile.get('name', 'Belum diatur')}")
            st.write(f"**Model:** `{st.session_state.get('model', CFG['model'])}`")

def is_json_like(text: str) -> bool:
    text = text.strip()
    return (text.startswith('{') and text.endswith('}')) or (text.startswith('[') and text.endswith(']'))

def is_table_like(text: str) -> bool:
    if "```" in text:
        return False

    if re.search(r'^[\s*•\-]+', text, re.MULTILINE):
        return False

    lines = text.strip().split('\n')
    if len(lines) < 2:
        return False

    for line in lines[:2]:
        if ',' in line or '\t' in line:
            if len(line.split(',')) > 1 or len(line.split('\t')) > 1:
                return True
    return False

def extract_and_display_data(answer: str):
    try:
        if is_json_like(answer):
            json_str = re.search(r"```(?:json)?\s*({.*?})\s*```", answer, re.DOTALL)
            if json_str:
                json_data = json.loads(json_str.group(1))
            else:
                try:
                    json_data = json.loads(answer)
                except json.JSONDecodeError:
                     raise ValueError("Invalid JSON structure")
            st.write("### Data JSON:")
            st.json(json_data)
            return True

        elif is_table_like(answer):
            table_match = re.search(r"```(?:csv|tsv)?\s*(.*?)\s*```", answer, re.DOTALL)
            table_text = table_match.group(1) if table_match else answer

            delimiter = ','
            if '\t' in table_text.split('\n')[0]:
                delimiter = '\t'
            from io import StringIO
            df = pd.read_csv(StringIO(table_text), delimiter=delimiter)
            st.write("### Data Tabel:")
            st.dataframe(df)
            return True

    except Exception as e:
        print(f"Gagal parsing  {e}")
        pass
    return False

def main():
    init_db()
    cleanup_temp_files()
    st.set_page_config(page_title="🤖 Super-Pro AI Chatbot", layout="wide")

    session_keys = {
        "session_id": str(uuid.uuid4()),
        "messages": load_conversation(st.session_state.get("session_id", "")),
        "user_profile": load_user_profile(st.session_state.get("session_id", "")),
        "google_api_key": os.getenv("GOOGLE_API_KEY", ""),
        "system_prompt": "Anda adalah asisten AI yang ramah.",
        "retriever": None,
        "model": CFG["model"],
        "tool_web_search": False,
        "tool_wikipedia": False,
        "tool_rag": False,
        "uploaded_file": None
    }
    for key, default_value in session_keys.items():
        if key not in st.session_state:
            if key == "messages" and st.session_state.get("session_id"):
                st.session_state[key] = load_conversation(st.session_state["session_id"])
            elif key == "user_profile" and st.session_state.get("session_id"):
                st.session_state[key] = load_user_profile(st.session_state["session_id"])
            else:
                st.session_state[key] = default_value

    st.title("🤖 Super-Pro AI Chatbot")
    st.caption("Ngobrol, hitung matematika, cari info, atau tanya dokumen — semua bisa! ✨")
    render_sidebar()

    api_key = st.session_state.google_api_key
    if not api_key:
        st.warning("⚠️ Masukkan Google API Key di sidebar.")
        st.stop()

    try:
        graph = build_graph().compile()
        st.session_state.graph = graph
    except Exception as e:
        error_msg = f"❌ Error membangun graph: {str(e)}"
        st.error(error_msg)
        print(error_msg)
        st.stop()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if not extract_and_display_data(msg["content"]):
                 st.markdown(msg["content"])

    if prompt := st.chat_input("💬 Tanya apa saja..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🧠 _Thinking..._"):
                placeholder = st.empty()
                placeholder.markdown("🧠 Memproses...")

                MAX_HISTORY_MESSAGES = 8
                recent_messages = st.session_state.messages[-MAX_HISTORY_MESSAGES:] if len(st.session_state.messages) > MAX_HISTORY_MESSAGES else st.session_state.messages

                langchain_messages = []
                for msg in recent_messages:
                    if msg["role"] == "user":
                        langchain_messages.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        langchain_messages.append(AIMessage(content=msg["content"]))

                state_in = AgentState(
                    session_id=st.session_state.session_id,
                    goal=prompt,
                    context="",
                    memory=[],
                    tools_used=[],
                    messages=langchain_messages,
                    final_answer="",
                    token_count=0,
                    needs_tool=False,
                    needs_rag=False,
                    needs_human=False
                )

                try:
                    config = {"recursion_limit": 50}
                    final_state = st.session_state.graph.invoke(state_in, config=config)
                    raw_answer = final_state.get("final_answer", "Maaf, saya tidak bisa menjawab.")

                    if Guard.deny_list(raw_answer):
                        final_answer = "Maaf, saya tidak dapat menjawab pertanyaan tersebut."
                    else:
                        final_answer = Guard.mask_pii(raw_answer)

                    placeholder.empty()
                    if not extract_and_display_data(final_answer):
                         st.markdown(final_answer)

                    st.session_state.messages.append({"role": "assistant", "content": raw_answer})
                    save_conversation(st.session_state.session_id, st.session_state.messages)

                except Exception as e:
                    error_msg = f"Maaf, terjadi error: {str(e)}"
                    placeholder.markdown(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    save_conversation(st.session_state.session_id, st.session_state.messages)
                    print(f"💥 Error selama eksekusi: {error_msg}")


if __name__ == "__main__":
    main()

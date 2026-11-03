"""Modular Streamlit sidebar for the Multimodal RAG application."""

import json
import urllib.request
from datetime import datetime

import streamlit as st

BACKEND_LOCAL_LABEL = "Local GGUF (llama-cpp)"
BACKEND_EXTERNAL_LABEL = "LM Studio / External App"
DEFAULT_LLM_PATH = r"\mistral-7b-instruct-v0.2.Q4_K_M.gguf"
DEFAULT_BASE_URL = "http://localhost:1234/v1"
DEFAULT_API_KEY = "lm-studio"
GPU_MODE_CPU = "CPU only"
GPU_MODE_CUDA = "CUDA GPU offloading"


def llama_cpp_supports_cuda():
    """Check llama-cpp GPU offload support (not PyTorch CUDA support)."""
    try:
        from llama_cpp import llama_cpp
        return bool(llama_cpp.llama_supports_gpu_offload()), None
    except Exception as exc:
        return False, str(exc)


def test_server_connection(base_url, api_key):
    """Check an OpenAI-compatible server's /models endpoint."""
    try:
        request = urllib.request.Request(
            f"{base_url.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        with urllib.request.urlopen(request, timeout=8) as response:
            model_data = json.loads(response.read().decode("utf-8"))
        models = model_data.get("data", [])
        if models:
            st.success(f"Connected! Available model: {models[0].get('id', 'Unknown')}")
        else:
            st.warning("Server responded, but no model is loaded. Load a model first.")
    except Exception as exc:
        st.error(f"Could not connect: {exc}")


def render_backend_selector():
    backend_options = [
        BACKEND_LOCAL_LABEL,
        BACKEND_EXTERNAL_LABEL,
    ]

    current_backend = st.session_state.get("llm_backend", "local")

    selected = st.segmented_control(
        "Text Generation Backend",
        options=backend_options,
        default=(
            BACKEND_EXTERNAL_LABEL
            if current_backend == "external"
            else BACKEND_LOCAL_LABEL
        ),
        selection_mode="single",
        key="backend_selector",
        width="stretch",
    )

    backend = "external" if selected == BACKEND_EXTERNAL_LABEL else "local"

    st.session_state.llm_backend = backend
    return backend

def render_local_gguf_settings():
    """Only shown when Local GGUF is selected."""
    st.text_input(
        "Local LLM Path (GGUF)",
        value=st.session_state.get("llm_path", DEFAULT_LLM_PATH),
        key="llm_path_input",
        help="Path to the GGUF model used by llama-cpp.",
    )
    st.session_state.llm_path = st.session_state.llm_path_input

    with st.expander("⚡ GGUF GPU Acceleration", expanded=False):
        cuda_available, cuda_error = llama_cpp_supports_cuda()
        if cuda_available:
            st.success("CUDA support detected in llama-cpp-python.")
            options = [GPU_MODE_CPU, GPU_MODE_CUDA]
            saved_mode = st.session_state.get("gguf_gpu_mode", GPU_MODE_CUDA)
            mode = st.selectbox(
                "GGUF execution device",
                options,
                index=options.index(saved_mode) if saved_mode in options else 1,
                key="gguf_gpu_mode_dropdown",
                help="CUDA mode offloads GGUF layers to the GPU; CPU mode keeps them on the CPU.",
            )
            st.session_state.gguf_gpu_mode = mode
            st.session_state.llm_n_gpu_layers = -1 if mode == GPU_MODE_CUDA else 0
        else:
            st.session_state.gguf_gpu_mode = GPU_MODE_CPU
            st.session_state.llm_n_gpu_layers = 0
            st.selectbox("GGUF execution device", [GPU_MODE_CPU], disabled=True)
            st.warning("This llama-cpp-python build has no CUDA offload support. GGUF will use the CPU.")
            if cuda_error:
                st.caption(f"llama-cpp check: {cuda_error}")


def render_external_server_settings():
    """Only shown when LM Studio / External App is selected."""
    with st.expander("🚀 LM Studio / External App Connection", expanded=True):
        st.caption("Load a model in LM Studio, start its local API server, then enter its API base URL.")
        base_url = st.text_input(
            "Local API Base URL",
            value=st.session_state.get("llm_base_url", DEFAULT_BASE_URL),
            key="llm_base_url_input",
            placeholder=DEFAULT_BASE_URL,
        )
        st.session_state.llm_base_url = base_url.rstrip("/")
        api_key = st.text_input(
            "API Key (if required)",
            value=st.session_state.get("llm_api_key", DEFAULT_API_KEY),
            key="llm_api_key_input",
            type="password",
        )
        st.session_state.llm_api_key = api_key or DEFAULT_API_KEY
        if st.button("🔌 Test Server Connection", use_container_width=True):
            test_server_connection(st.session_state.llm_base_url, st.session_state.llm_api_key)


def render_system_status():
    if not st.session_state.get("initialized", False):
        st.warning("⚠️ System not initialized. Choose a backend and click Initialize System.")
        return
    st.markdown("### 📊 System Status")
    stats = st.session_state.rag_system.get_statistics()
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Docs", stats["total_documents"])
        st.metric("Text Chunks", stats["text_chunks"])
    with col2:
        st.metric("Images", stats["image_chunks"])
        st.metric("Audio", stats["audio_chunks"])
    st.success("🟢 System Active")


def render_supported_formats():
    st.markdown("### 📁 Supported Formats")
    st.markdown("- 📄 **Documents**: PDF, DOCX, TXT\n- 🖼️ **Images**: PNG, JPG, JPEG, GIF\n- 🎵 **Audio**: MP3, WAV, M4A")


def render_quick_actions():
    st.markdown("### ⚡ Quick Actions")
    if st.button("💣 Delete Database"):
        if st.session_state.get("initialized"):
            try:
                st.session_state.rag_system.drop_database()
                st.session_state.rag_system._init_vector_db()
                st.session_state.search_history = []
                st.session_state.uploaded_files_list = []
                st.success("✅ Database deleted and reset successfully!")
                st.rerun()
            except Exception as exc:
                st.error(f"❌ Failed to delete database: {exc}")
        else:
            st.warning("Please initialize the system first")
    if st.button("📤 Export Index"):
        if st.session_state.get("initialized"):
            try:
                export_path = f"rag_backup_{datetime.now():%Y%m%d_%H%M%S}.json"
                st.session_state.rag_system.export_index(export_path)
                st.success(f"✅ Index exported to {export_path}")
            except Exception as exc:
                st.error(f"❌ Export failed: {exc}")
        else:
            st.warning("Please initialize the system first")
    if st.button("🗑️ Clear Cache"):
        st.cache_data.clear()
        st.success("✅ Cache cleared")


def render_sidebar(initialize_callback):
    """Render the sidebar. The callback initializes the selected backend."""
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/search.png", width=80)
        st.markdown("## ⚙️ System Configuration")
        backend = render_backend_selector()
        if backend == "local":
            render_local_gguf_settings()
        else:
            render_external_server_settings()
        if st.button("🚀 Initialize System", type="primary", use_container_width=True):
            st.session_state.initialized = False
            st.session_state.rag_system = None
            initialize_callback()
        if st.button("🗑️ Unload Local LLM", use_container_width=True):
                                rag = st.session_state.get("rag_system")
                    
                                if rag and hasattr(rag, "unload_llm"):
                                    rag.unload_llm()
                                    st.success("LLM unloaded from memory.")
                                else:
                                    st.warning("No local LLM is currently loaded.")
        st.markdown("---")
        render_system_status()
        st.markdown("---")
        render_supported_formats()
        st.markdown("---")
        render_quick_actions()

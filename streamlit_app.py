"""
Multimodal RAG System - Streamlit Application
Complete offline-capable RAG system with beautiful UI
"""

import streamlit as st
import os
from pathlib import Path
import time
from datetime import datetime
import json

# Import the RAG system
from multimodal_rag_system import MultimodalRAGSystem

# Page configuration
st.set_page_config(
    page_title="Multimodal RAG System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4F46E5;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    .stButton>button:hover {
        background-color: #4338CA;
        border-color: #4338CA;
    }
    .citation-box {
        background-color: #F3F4F6;
        border-left: 4px solid #4F46E5;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 4px;
    }
    .metric-card {
        background-color: #F9FAFB;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #E5E7EB;
    }
    .success-message {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .error-message {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
    st.session_state.initialized = False
    st.session_state.search_history = []
    st.session_state.uploaded_files_list = []

def initialize_system():
    """Initialize the RAG system"""
    if not st.session_state.initialized:
        with st.spinner("🚀 Initializing RAG System... This may take a minute on first run."):
            try:
                st.session_state.rag_system = MultimodalRAGSystem(
                    storage_dir="./rag_storage",
                    use_gpu=st.session_state.get('use_gpu', False),
                    llm_path=st.session_state.get('llm_path', None),
                    oracle_dsn=st.session_state.get('oracle_dsn', None),
                    oracle_user=st.session_state.get('oracle_user', None),
                    oracle_password=st.session_state.get('oracle_password', None),
                )
                st.session_state.initialized = True
                st.success("✅ System initialized successfully!")
                return True
            except Exception as e:
                st.error(f"❌ Failed to initialize system: {str(e)}")
                return False
    return True

def format_file_size(size_bytes):
    """Format file size in human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"

def get_file_icon(file_type):
    """Get emoji icon for file type"""
    if 'pdf' in file_type.lower():
        return "📄"
    elif 'word' in file_type.lower() or 'docx' in file_type.lower():
        return "📝"
    elif 'image' in file_type.lower() or any(ext in file_type.lower() for ext in ['png', 'jpg', 'jpeg']):
        return "🖼️"
    elif 'audio' in file_type.lower() or any(ext in file_type.lower() for ext in ['mp3', 'wav']):
        return "🎵"
    else:
        return "📎"

def main():
    if 'current_results' not in st.session_state:
        st.session_state.current_results = None
    if 'current_query_display' not in st.session_state:
        st.session_state.current_query_display = None
    if 'current_llm_query' not in st.session_state:
        st.session_state.current_llm_query = None
    if 'current_answer' not in st.session_state:
        st.session_state.current_answer = None

    # Header
    st.markdown("""
        <h1 style='text-align: center; color: #4F46E5; margin-bottom: 0;'>
            🔍 Multimodal RAG System
        </h1>
        <p style='text-align: center; color: #6B7280; margin-top: 0.5rem;'>
            Semantic Search Across Documents, Images & Audio • Offline Capable
        </p>
        <hr style='margin: 1rem 0; border: none; border-top: 2px solid #E5E7EB;'>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/search.png", width=80)
        st.markdown("## ⚙️ System Configuration")
        
        # GPU toggle
        use_gpu = st.checkbox(
            "Use GPU Acceleration",
            value=st.session_state.get('use_gpu', False),
            help="Enable GPU for faster processing (requires CUDA)"
        )
        st.session_state.use_gpu = use_gpu

        # Oracle 23ai credentials
        st.markdown("#### 🗄️ Oracle 23ai Connection")
        oracle_dsn = st.text_input(
            "Oracle DSN",
            value=st.session_state.get('oracle_dsn', os.environ.get('ORACLE_DSN', 'localhost:1521/FREEPDB1')),
            help="EZConnect string, e.g. host:port/service or a TNS alias"
        )
        st.session_state.oracle_dsn = oracle_dsn

        oracle_user = st.text_input(
            "Oracle User",
            value=st.session_state.get('oracle_user', os.environ.get('ORACLE_USER', 'rag_user')),
        )
        st.session_state.oracle_user = oracle_user

        oracle_password = st.text_input(
            "Oracle Password",
            value=st.session_state.get('oracle_password', os.environ.get('ORACLE_PASSWORD', '')),
            type='password'
        )
        st.session_state.oracle_password = oracle_password

        st.markdown("---")

        # Local LLM Path
        llm_path = st.text_input(
            "Local LLM Path (GGUF)",
            value=st.session_state.get('llm_path', "put your model path here(full path is recommended)"),
            help="Path to the GGUF model file for text generation"
        )
        st.session_state.llm_path = llm_path
        if st.button("🚀 Initialize System", type="primary"):
            initialize_system()
        
        st.markdown("---")
        
        # System status
        if st.session_state.initialized:
            st.markdown("### 📊 System Status")
            
            # Get statistics
            stats = st.session_state.rag_system.get_statistics()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Docs", stats['total_documents'])
                st.metric("Text Chunks", stats['text_chunks'])
            with col2:
                st.metric("Images", stats['image_chunks'])
                st.metric("Audio", stats['audio_chunks'])
            
            # Status indicator
            st.markdown("""
                <div style='background-color: #D1FAE5; padding: 0.5rem; border-radius: 8px; text-align: center; margin-top: 1rem;'>
                    <span style='color: #065F46; font-weight: 600;'>🟢 System Active</span>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ System not initialized. Click the button above to start.")
        
        st.markdown("---")
        
        # Supported formats
        st.markdown("### 📁 Supported Formats")
        st.markdown("""
        - 📄 **Documents**: PDF, DOCX, TXT
        - 🖼️ **Images**: PNG, JPG, JPEG, GIF
        - 🎵 **Audio**: MP3, WAV, M4A
        """)
        
        st.markdown("---")
        
        # Quick actions
        st.markdown("### ⚡ Quick Actions")
        
        if st.button("💣 Delete Database"):
            if st.session_state.initialized:
                try:
                    st.session_state.rag_system.drop_database()
                    st.session_state.rag_system._init_vector_db()
                    st.session_state.search_history = []
                    st.session_state.uploaded_files_list = []
                    st.success("✅ Database deleted and reset successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Failed to delete database: {str(e)}")
            else:
                st.warning("Please initialize the system first")

        if st.button("📤 Export Index"):
            if st.session_state.initialized:
                try:
                    export_path = f"rag_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    st.session_state.rag_system.export_index(export_path)
                    st.success(f"✅ Index exported to {export_path}")
                except Exception as e:
                    st.error(f"❌ Export failed: {str(e)}")
            else:
                st.warning("Please initialize the system first")
        
        if st.button("🗑️ Clear Cache"):
            st.cache_data.clear()
            st.success("✅ Cache cleared")
    
    # Main content area
    if not st.session_state.initialized:
        # Welcome screen
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
                <div style='text-align: center; padding: 3rem 0;'>
                    <h2 style='color: #1F2937; margin-bottom: 1rem;'>Welcome to the Multimodal RAG System</h2>
                    <p style='color: #6B7280; font-size: 1.1rem; margin-bottom: 2rem;'>
                        A powerful offline-capable search system that understands documents, images, and audio files.
                    </p>
                    <div style='background-color: #EEF2FF; padding: 2rem; border-radius: 12px; margin: 2rem 0;'>
                        <h3 style='color: #4F46E5; margin-bottom: 1rem;'>Getting Started</h3>
                        <ol style='text-align: left; color: #4B5563;'>
                            <li style='margin: 0.5rem 0;'>Click "Initialize System" in the sidebar</li>
                            <li style='margin: 0.5rem 0;'>Upload your documents using the "Upload Documents" tab</li>
                            <li style='margin: 0.5rem 0;'>Start searching with natural language queries</li>
                        </ol>
                    </div>
                </div>
            """, unsafe_allow_html=True)
    else:
        # Main application tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "🔍 Search & Query",
            "📤 Upload Documents", 
            "📚 Document Library",
            "📊 Analytics"
        ])
        
        # TAB 1: Search & Query
        with tab1:
            st.markdown("### 🔍 Semantic Search")
            
            # Search interface
            st.markdown("#### Select Search Mode")
            search_mode = st.radio("Search Mode", ["Text Query", "Image Query", "Audio Query"], horizontal=True, label_visibility="collapsed")

            search_query = None
            search_file = None

            if search_mode == "Text Query":
                col1, col2 = st.columns([3, 1])
                with col1:
                    search_query = st.text_input(
                        "Enter your search query",
                        placeholder="e.g., What was discussed about international development in 2024?",
                        label_visibility="collapsed"
                    )
                with col2:
                    search_button = st.button("🔍 Search Text", type="primary", use_container_width=True)
            
            elif search_mode == "Image Query":
                col1, col2 = st.columns([3, 1])
                with col1:
                    search_file = st.file_uploader("Upload an image to search", type=['png', 'jpg', 'jpeg'], label_visibility="collapsed")
                with col2:
                    search_button = st.button("🖼️ Search Image", type="primary", use_container_width=True, disabled=not search_file)

            elif search_mode == "Audio Query":
                col1, col2 = st.columns([3, 1])
                with col1:
                    search_file = st.file_uploader("Upload an audio file to search", type=['mp3', 'wav', 'm4a'], label_visibility="collapsed")
                with col2:
                    search_button = st.button("🎵 Search Audio", type="primary", use_container_width=True, disabled=not search_file)

            # Search options
            with st.expander("⚙️ Advanced Options"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    modality_filter = st.selectbox(
                        "Filter results by type",
                        ["All Types", "Text Only", "Images Only", "Audio Only"]
                    )
                with col2:
                    top_k = st.slider("Number of results", 1, 20, 5)
                with col3:
                    cross_modal = st.checkbox("Cross-modal search", value=True, 
                                             help="Enable text-to-image and image-to-text search")
            
            # Example queries (Text mode only)
            if search_mode == "Text Query":
                with st.expander("💡 Example Text Queries"):
                    examples = [
                        "Show me the report about international development in 2024",
                        "Find the screenshot taken at 14:32",
                        "What was discussed in the meeting about budget allocation?",
                        "Show documents mentioning sustainability",
                        "Find images related to financial charts"
                    ]
                    for example in examples:
                        if st.button(f"📝 {example}", key=example):
                            search_query = example
                            # We can't auto-click the search button easily, but we can set session state to trigger
                            st.session_state.auto_search = True
                            st.rerun()

            # Handle auto-search from examples
            if st.session_state.get('auto_search') and search_query:
                search_button = True
                st.session_state.auto_search = False

            # Perform search logic
            if search_button:
                with st.spinner("🔎 Searching..."):
                    try:
                        # Convert modality filter
                        modality = None
                        if modality_filter == "Text Only":
                            modality = "text"
                        elif modality_filter == "Images Only":
                            modality = "image"
                        elif modality_filter == "Audio Only":
                            modality = "audio"
                        
                        results = []
                        query_display = ""

                        if search_mode == "Text Query" and search_query:
                            query_display = search_query
                            results = st.session_state.rag_system.search(
                                query=search_query,
                                modality=modality,
                                top_k=top_k,
                                cross_modal=cross_modal
                            )
                        
                        elif search_mode == "Image Query" and search_file:
                            query_display = f"Image: {search_file.name}"
                            # Save temp file
                            temp_dir = Path("./temp_uploads")
                            temp_dir.mkdir(exist_ok=True)
                            temp_path = temp_dir / f"query_{search_file.name}"
                            with open(temp_path, "wb") as f:
                                f.write(search_file.getbuffer())
                            
                            results = st.session_state.rag_system.search_by_image(
                                image_path=str(temp_path),
                                top_k=top_k
                            )
                            # Cleanup
                            temp_path.unlink()

                        elif search_mode == "Audio Query" and search_file:
                            # Save temp file
                            temp_dir = Path("./temp_uploads")
                            temp_dir.mkdir(exist_ok=True)
                            temp_path = temp_dir / f"query_{search_file.name}"
                            with open(temp_path, "wb") as f:
                                f.write(search_file.getbuffer())
                            
                            results, transcript = st.session_state.rag_system.search_by_audio(
                                audio_path=str(temp_path),
                                top_k=top_k
                            )
                            query_display = f"Audio: {search_file.name} (Transcript: '{transcript}')"
                            llm_query = transcript

                            # Cleanup
                            temp_path.unlink()
                        
                        # Save to history
                        if query_display:
                            st.session_state.search_history.insert(0, {
                                'query': query_display,
                                'timestamp': datetime.now().isoformat(),
                                'results_count': len(results)
                            })
                        
                        # Save results to session state
                        if results:
                            st.session_state.current_results = results
                            st.session_state.current_query_display = query_display

                            # Determine LLM query if not set (Text/Image cases)
                            if not locals().get('llm_query'):
                                if search_mode == "Text Query":
                                    llm_query = search_query
                                elif search_mode == "Image Query":
                                    llm_query = "Describe the retrieved information and how it relates to the input image."

                            st.session_state.current_llm_query = llm_query
                            st.session_state.current_answer = None

                            if llm_query:
                                with st.spinner("Generating response..."):
                                    st.session_state.current_answer = st.session_state.rag_system.generate_answer(
                                        query=llm_query,
                                        context_chunks=results
                                    )
                        else:
                            st.session_state.current_results = []
                            st.session_state.current_answer = None
                            st.warning("⚠️ No results found. Try a different query or upload more documents.")
                    
                    except Exception as e:
                        st.error(f"❌ Search failed: {str(e)}")

            # --- Render persistent results ---
            if getattr(st.session_state, 'current_results', None):
                results = st.session_state.current_results
                answer = getattr(st.session_state, 'current_answer', None)
                
                st.success(f"✅ Found {len(results)} relevant results")
                
                # AI Answer
                if answer:
                    st.markdown("### 🤖 AI Answer")
                    with st.chat_message("assistant"):
                        st.write(answer)
                
                st.markdown("---")
                st.markdown(f"### 📚 Retrieved Sources ({len(results)})")
                
                for i, result in enumerate(results, 1):
                    meta = result['metadata']
                    source_file = meta.get('source_file', 'Unknown')
                    file_type = meta.get('file_type', 'Unknown')
                    
                    with st.expander(
                        f"**[{i}] {get_file_icon(file_type)} {source_file}** "
                        f"({result['relevance']:.1%} match)",
                        expanded=(i <= 3)
                    ):
                        # Metadata Columns
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Relevance", f"{result['relevance']:.1%}")
                        with col2:
                            # Display location based on type
                            if 'page' in meta:
                                st.metric("Location", f"Page {meta['page']}")
                            elif 'start_ms' in meta:
                                start_sec = meta['start_ms'] / 1000.0
                                st.metric("Time", f"{start_sec:.1f}s")
                            elif 'image_path' in meta:
                                st.metric("Type", "Image")
                            else:
                                st.metric("Location", "N/A")
                        with col3:
                            st.metric("Format", file_type.upper())
                        
                        # Open File Button
                        file_path_meta = meta.get('file_path') or meta.get('image_path')
                        if file_path_meta:
                            if st.button(f"📂 Open File", key=f"open_{result['id']}"):
                                try:
                                    os.startfile(file_path_meta)
                                    st.toast(f"Opening {os.path.basename(file_path_meta)}...", icon="📂")
                                except Exception as e:
                                    st.error(f"Could not open file: {e}")
                        else:
                            st.caption("🚫 File path not available (re-index to fix)")

                        # Content Preview / Media
                        st.markdown("---")
                        
                        # If it's an image result (or OCR from image), try to show it
                        if 'image_path' in meta and __import__('os').path.exists(meta['image_path']):
                            st.image(meta['image_path'], caption=f"Source: {source_file}", use_column_width=True)
                        
                        st.markdown("**Content Segment:**")
                        st.info(result['content'])
                        
                        # Full metadata listing
                        with st.expander("🔍 System Metadata"):
                            st.json(meta)

            
            # Search history
            if st.session_state.search_history:
                st.markdown("---")
                st.markdown("### 📜 Recent Searches")
                for i, search in enumerate(st.session_state.search_history[:5]):
                    st.text(f"{i+1}. \"{search['query']}\" - {search['results_count']} results")
        
        # TAB 2: Upload Documents
        with tab2:
            st.markdown("### 📤 Upload Documents")
            
            # File uploader
            uploaded_files = st.file_uploader(
                "Choose files to upload",
                accept_multiple_files=True,
                type=['pdf', 'docx', 'doc', 'txt', 'png', 'jpg', 'jpeg', 'gif', 'mp3', 'wav', 'm4a'],
                help="Upload PDF, DOCX, images, or audio files"
            )
            
            if uploaded_files:
                st.markdown(f"**{len(uploaded_files)} file(s) selected**")
                
                # Initialize background task state
                if 'upload_queue' not in st.session_state:
                    st.session_state.upload_queue = []
                if 'upload_status' not in st.session_state:
                    st.session_state.upload_status = {'processing': False, 'current': '', 'progress': 0, 'total': 0, 'successful': 0, 'failed': [], 'total_chunks': 0}

                # Button to queue files
                if st.button("📥 Queue Files for Processing", type="primary"):
                    st.session_state.upload_queue.extend(uploaded_files)
                    
                    if not st.session_state.upload_status['processing']:
                        st.session_state.upload_status['processing'] = True
                        st.session_state.upload_status['total'] = len(st.session_state.upload_queue)
                        st.session_state.upload_status['progress'] = 0
                        st.session_state.upload_status['successful'] = 0
                        st.session_state.upload_status['failed'] = []
                        st.session_state.upload_status['total_chunks'] = 0
                        
                        import threading
                        
                        def process_queue():
                            try:
                                while st.session_state.upload_queue:
                                    uploaded_file = st.session_state.upload_queue.pop(0)
                                    st.session_state.upload_status['current'] = uploaded_file.name
                                    
                                    try:
                                        # Save uploaded file temporarily
                                        temp_dir = Path("./temp_uploads")
                                        temp_dir.mkdir(exist_ok=True)
                                        temp_path = temp_dir / uploaded_file.name
                                        
                                        with open(temp_path, "wb") as f:
                                            f.write(uploaded_file.getbuffer())
                                        
                                        # Ingest document
                                        chunks = st.session_state.rag_system.ingest_document(str(temp_path))
                                        st.session_state.upload_status['total_chunks'] += len(chunks)
                                        st.session_state.upload_status['successful'] += 1
                                        
                                        # Add to uploaded files list
                                        st.session_state.uploaded_files_list.append({
                                            'name': uploaded_file.name,
                                            'size': uploaded_file.size,
                                            'type': uploaded_file.type,
                                            'chunks': len(chunks),
                                            'timestamp': datetime.now().isoformat()
                                        })
                                    except Exception as e:
                                        st.session_state.upload_status['failed'].append((uploaded_file.name, str(e)))
                                    
                                    st.session_state.upload_status['progress'] += 1
                                
                                st.session_state.upload_status['processing'] = False
                            except Exception as e:
                                print(f"Background thread error: {e}")
                                st.session_state.upload_status['processing'] = False

                        # Start background thread
                        thread = threading.Thread(target=process_queue, daemon=True)
                        import streamlit.runtime.scriptrunner as scriptrunner
                        scriptrunner.add_script_run_ctx(thread)
                        thread.start()

                # --- FRAGMENT: Auto-refreshing status UI ---
                @st.fragment(run_every="2s")
                def upload_status_ui():
                    status = st.session_state.get('upload_status', {})
                    if status.get('processing'):
                        st.info(f"⏳ Processing in background... ({status['progress']}/{status['total']})")
                        st.text(f"Currently indexing: {status['current']}")
                        if status['total'] > 0:
                            st.progress(status['progress'] / status['total'])
                    elif status.get('total', 0) > 0 and not status.get('processing'):
                        st.success(f"✅ Background processing complete!")
                        st.markdown(f"""
                            <div class='success-message'>
                                <ul>
                                    <li>Successfully processed: {status['successful']} files</li>
                                    <li>Total chunks created: {status['total_chunks']}</li>
                                    <li>Failed: {len(status['failed'])} files</li>
                                </ul>
                            </div>
                        """, unsafe_allow_html=True)
                        if status.get('failed'):
                            st.error("**Failed files:**")
                            for name, error in status['failed']:
                                st.text(f"❌ {name}: {error}")
                        
                        # Reset button
                        if st.button("Clear Status"):
                            st.session_state.upload_status = {'processing': False, 'current': '', 'progress': 0, 'total': 0, 'successful': 0, 'failed': [], 'total_chunks': 0}
                            st.rerun()

                upload_status_ui()
            
            # Batch upload from directory
            st.markdown("---")
            st.markdown("### 📁 Batch Upload from Directory")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                directory_path = st.text_input(
                    "Enter directory path",
                    placeholder="e.g., ./documents",
                    help="Path to directory containing documents to index"
                )
            with col2:
                st.markdown("<br>", unsafe_allow_html=True)
                batch_upload = st.button("📂 Process Directory")
            
            if batch_upload and directory_path:
                if os.path.isdir(directory_path):
                    with st.spinner(f"Processing directory: {directory_path}"):
                        from batch_ingest import BatchIngester
                        
                        ingester = BatchIngester(st.session_state.rag_system)
                        stats = ingester.ingest_directory(directory_path, recursive=True)
                        
                        st.success(f"""
                            ✅ Batch processing complete!
                            - Files found: {stats['total_files']}
                            - Successfully indexed: {stats['successful']}
                            - Failed: {stats['failed']}
                            - Total chunks: {stats['chunks_created']}
                        """)
                else:
                    st.error("❌ Directory not found")
        
        # TAB 3: Document Library
        with tab3:
            st.markdown("### 📚 Document Library")
            
            if st.session_state.uploaded_files_list:
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                
                total_files = len(st.session_state.uploaded_files_list)
                total_size = sum(f['size'] for f in st.session_state.uploaded_files_list)
                total_chunks = sum(f['chunks'] for f in st.session_state.uploaded_files_list)
                
                with col1:
                    st.metric("Total Files", total_files)
                with col2:
                    st.metric("Total Size", format_file_size(total_size))
                with col3:
                    st.metric("Total Chunks", total_chunks)
                with col4:
                    avg_chunks = total_chunks / total_files if total_files > 0 else 0
                    st.metric("Avg Chunks/File", f"{avg_chunks:.1f}")
                
                st.markdown("---")
                
                # File list
                st.markdown("### 📄 Uploaded Files")
                
                for i, file_info in enumerate(st.session_state.uploaded_files_list):
                    with st.expander(
                        f"{get_file_icon(file_info['type'])} {file_info['name']} "
                        f"({format_file_size(file_info['size'])})"
                    ):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.text(f"Type: {file_info['type']}")
                            st.text(f"Size: {format_file_size(file_info['size'])}")
                        with col2:
                            st.text(f"Chunks: {file_info['chunks']}")
                            st.text(f"Uploaded: {file_info['timestamp'][:19]}")
            else:
                st.info("📭 No documents uploaded yet. Go to the 'Upload Documents' tab to add files.")
        
        # TAB 4: Analytics
        with tab4:
            st.markdown("### 📊 System Analytics")
            
            # Get current statistics
            stats = st.session_state.rag_system.get_statistics()
            
            # Overview metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown("""
                    <div class='metric-card'>
                        <h3 style='color: #4F46E5; margin: 0;'>{}</h3>
                        <p style='color: #6B7280; margin: 0; font-size: 0.9rem;'>Total Documents</p>
                    </div>
                """.format(stats['total_documents']), unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                    <div class='metric-card'>
                        <h3 style='color: #059669; margin: 0;'>{}</h3>
                        <p style='color: #6B7280; margin: 0; font-size: 0.9rem;'>Text Chunks</p>
                    </div>
                """.format(stats['text_chunks']), unsafe_allow_html=True)
            
            with col3:
                st.markdown("""
                    <div class='metric-card'>
                        <h3 style='color: #DC2626; margin: 0;'>{}</h3>
                        <p style='color: #6B7280; margin: 0; font-size: 0.9rem;'>Image Chunks</p>
                    </div>
                """.format(stats['image_chunks']), unsafe_allow_html=True)
            
            with col4:
                st.markdown("""
                    <div class='metric-card'>
                        <h3 style='color: #7C3AED; margin: 0;'>{}</h3>
                        <p style='color: #6B7280; margin: 0; font-size: 0.9rem;'>Audio Chunks</p>
                    </div>
                """.format(stats['audio_chunks']), unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Search history analytics
            if st.session_state.search_history:
                st.markdown("### 📈 Search Activity")
                st.markdown(f"**Total searches:** {len(st.session_state.search_history)}")
                
                # Most recent searches
                st.markdown("**Recent queries:**")
                for search in st.session_state.search_history[:10]:
                    st.text(f"• {search['query']} ({search['results_count']} results)")
            
            st.markdown("---")
            
            # System info
            st.markdown("### 🖥️ System Information")
            col1, col2 = st.columns(2)
            
            with col1:
                st.text(f"Storage Path: {stats['storage_path']}")
                st.text(f"GPU Enabled: {'Yes' if st.session_state.use_gpu else 'No'}")
            
            with col2:
                st.text(f"Status: ✅ Active")
                st.text(f"Mode: 🔒 Offline")

if __name__ == "__main__":
    main()

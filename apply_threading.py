import pathlib
import re

file_path = pathlib.Path("streamlit_app.py")
content = file_path.read_text(encoding="utf-8")

# We want to replace the synchronous for-loop with a threaded queue implementation inside an st.fragment

start_marker = "                if st.button(\"📥 Process & Index Files\", type=\"primary\"):\n"
end_marker = "                    if failed:\n                        st.error(\"**Failed files:**\")\n                        for name, error in failed:\n                            st.text(f\"❌ {name}: {error}\")\n"

start_idx = content.find(start_marker)
end_idx = content.find(end_marker, start_idx) + len(end_marker)

if start_idx == -1 or end_idx == -1:
    print("Markers not found!")
    exit(1)

new_block = """                # Initialize background task state
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
                        st.markdown(f\"\"\"
                            <div class='success-message'>
                                <ul>
                                    <li>Successfully processed: {status['successful']} files</li>
                                    <li>Total chunks created: {status['total_chunks']}</li>
                                    <li>Failed: {len(status['failed'])} files</li>
                                </ul>
                            </div>
                        \"\"\", unsafe_allow_html=True)
                        if status.get('failed'):
                            st.error("**Failed files:**")
                            for name, error in status['failed']:
                                st.text(f"❌ {name}: {error}")
                        
                        # Reset button
                        if st.button("Clear Status"):
                            st.session_state.upload_status = {'processing': False, 'current': '', 'progress': 0, 'total': 0, 'successful': 0, 'failed': [], 'total_chunks': 0}
                            st.rerun()

                upload_status_ui()
"""

content = content[:start_idx] + new_block + content[end_idx:]
file_path.write_text(content, encoding="utf-8")
print("Successfully patched streamline_app.py")

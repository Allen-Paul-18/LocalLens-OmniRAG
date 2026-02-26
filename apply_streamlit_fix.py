import pathlib

file_path = pathlib.Path("streamlit_app.py")
content = file_path.read_text(encoding="utf-8")

# We want to replace everything from `# Display results` down to the `except Exception as e:` block
start_marker = "                        # Display results\n"
end_marker = "                    except Exception as e:\n                        st.error(f\"❌ Search failed: {str(e)}\")\n"

start_idx = content.find(start_marker)
end_idx = content.find(end_marker, start_idx) + len(end_marker)

if start_idx == -1 or end_idx == -1:
    print("Markers not found!")
    exit(1)

new_block = """                        # Save results to session state
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
                                    import os
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
\n"""

content = content[:start_idx] + new_block + content[end_idx:]
file_path.write_text(content, encoding="utf-8")
print("Successfully updated streamlit_app.py")

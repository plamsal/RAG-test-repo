import streamlit as st
import numpy as np
import os
import re
import json
from snowflake.snowpark.context import get_active_session

class HybridRAGSystem:
    def __init__(self, embedding_model='all-RAG_AGENT.PUBLIC.GITHUB_ACTION-L6-v2'):
        try:
            self.session = get_active_session()
        except Exception as e:
            st.error(f"Could not get Snowflake session: {e}")
            st.stop()

        # Local embedding model setup (we'll keep this minimal for now)
        self.embedding_dimension = 384  # Default for MiniLM
        self.setup_complete = False

    def load_git_repository_content(self):
        """
        Load content from GitHub-backed Snowflake repository with robust error handling
        """
        try:
            st.info("Discovering content from Git repository...")

            # Use a more robust SQL query to list files
            files = self.session.sql("""
                SELECT METADATA$FILENAME AS file_path
                FROM @github_action/branches/main/
            """).collect()
            
            # Filter and process relevant files
            relevant_files = [
                f[0] for f in files 
                if any(f[0].endswith(ext) for ext in ['.sql', '.xml', '.md', '.txt', '.py'])
            ]

            st.info(f"Found {len(relevant_files)} potential files")
            
            loaded_content = []
            for file_path in relevant_files:
                try:
                    # Use LISTAGG to read file contents safely
                    content_query = f"""
                        SELECT LISTAGG(TO_VARCHAR($1), '\n') AS file_content
                        FROM @github_action/branches/main/{file_path}
                    """
                    
                    content_result = self.session.sql(content_query).collect()
                    
                    if content_result and content_result[0][0]:
                        loaded_content.append({
                            'filename': file_path.split('/')[-1],
                            'path': file_path,
                            'content': content_result[0][0],
                            'type': self._determine_file_type(file_path)
                        })
                    else:
                        st.warning(f"Empty content for {file_path}")
                except Exception as file_err:
                    st.warning(f"Error reading {file_path}: {file_err}")

            return loaded_content

        except Exception as e:
            st.error(f"Error loading repository: {e}")
            return []

    def _determine_file_type(self, file_path):
        """Enhanced file type determination"""
        ext_mapping = {
            '.sql': 'SQL',
            '.xml': 'XML', 
            '.md': 'Markdown', 
            '.py': 'Python',
            '.txt': 'Text'
        }
        return next((type_name for ext, type_name in ext_mapping.items() if file_path.endswith(ext)), 'Unknown')

    def setup_rag_system(self):
        """
        Comprehensive RAG system setup with enhanced error handling
        """
        st.header("🚀 Hybrid RAG System Setup")

        # Step 1: Load repository content
        loaded_content = self.load_git_repository_content()
        if not loaded_content:
            st.error("No content found in repository")
            return False

        # Display loaded content details
        st.subheader("Loaded Content Details")
        for item in loaded_content:
            st.write(f"File: {item['filename']} (Type: {item['type']})")
            st.code(item['content'][:200] + '...' if len(item['content']) > 200 else item['content'], language='text')

        self.setup_complete = True
        st.success("RAG System Setup Partially Complete!")
        return True

def main():
    st.title("🔍 Hybrid RAG Agent: GitHub + Snowflake")

    # Initialize RAG system
    rag_system = HybridRAGSystem()

    # Sidebar configuration
    with st.sidebar:
        st.header("RAG System Configuration")
        
        # Embedding model selection
        embedding_model = st.selectbox(
            "Embedding Model", 
            [
                'all-MiniLM-L6-v2',
                'all-mpnet-base-v2'
            ]
        )

        # Setup button
        if st.button("🚀 Initialize RAG System"):
            with st.spinner("Setting up RAG system..."):
                rag_system.setup_rag_system()

    # Q&A Interface
    st.header("💬 Repository Query")
    
    question = st.text_input("Enter your query:", placeholder="What changed in Sprint01?")
    
    if st.button("🔍 Search & Generate"):
        if not rag_system.setup_complete:
            st.warning("Please initialize the RAG system first!")
        else:
            st.info("Full Q&A functionality will be implemented in future updates")

if __name__ == "__main__":
    main()
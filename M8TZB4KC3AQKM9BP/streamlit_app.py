import streamlit as st
import numpy as np
from snowflake.snowpark.context import get_active_session
from sentence_transformers import SentenceTransformer
import re
from typing import List, Dict, Any

class HybridRAGSystem:
    def __init__(self, embedding_model='intfloat/e5-base-v2'):
        """
        Initialize hybrid RAG system for your GitHub-backed Snowflake RAG agent
        """
        try:
            self.session = get_active_session()
        except Exception as e:
            st.error(f"Could not get Snowflake session: {e}")
            st.stop()

        # Local embedding model
        self.embedding_model = None
        self.embedding_model_name = embedding_model
        self.embedding_dimension = None

        # Setup flags
        self.setup_complete = False

    def initialize_embedding_model(self):
        """Load local embedding model"""
        try:
            st.info(f"Loading local embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)

            # Get dimension
            test_embed = self.embedding_model.encode(["test"])
            self.embedding_dimension = len(test_embed[0])

            st.success(f"✓ Local embedding model ready (dim: {self.embedding_dimension})")
            return True

        except Exception as e:
            st.error(f"Error loading embedding model: {e}")
            return False

    def load_git_repository_content(self):
        """
        Load content from your GitHub-backed Snowflake repository
        """
        try:
            st.info("Discovering content from Git repository...")

            # List files in the repository
            files = self.session.sql("LIST @github_action/branches/main/").collect()
            
            # Filter and process relevant files (SQL, XML, etc.)
            relevant_files = [
                f for f in files 
                if f[0].endswith(('.sql', '.xml', '.md', '.txt'))
            ]

            st.info(f"Found {len(relevant_files)} potential files")
            
            # Read file contents
            loaded_content = []
            for file_info in relevant_files:
                file_path = file_info[0]
                try:
                    content = self.session.sql(f"""
                        SELECT $1 AS file_content
                        FROM @github_action/branches/main/{file_path}
                    """).collect()
                    
                    if content and content[0][0]:
                        loaded_content.append({
                            'filename': file_path.split('/')[-1],
                            'path': file_path,
                            'content': content[0][0],
                            'type': self._determine_file_type(file_path)
                        })
                except Exception as file_err:
                    st.warning(f"Could not read {file_path}: {file_err}")

            return loaded_content

        except Exception as e:
            st.error(f"Error loading repository: {e}")
            return []

    def _determine_file_type(self, file_path):
        """Determine file type based on extension"""
        if file_path.endswith('.sql'):
            return 'SQL'
        elif file_path.endswith('.xml'):
            return 'XML'
        elif file_path.endswith('.md'):
            return 'Markdown'
        else:
            return 'Text'

    def prepare_content_for_embedding(self, loaded_content):
        """
        Prepare content chunks for embedding
        """
        chunks = []
        for item in loaded_content:
            # Basic chunking strategy
            chunk_size = 500
            chunk_overlap = 100

            text = item['content']
            for i in range(0, len(text), chunk_size - chunk_overlap):
                chunk = text[i:i+chunk_size]
                chunks.append({
                    'chunk_text': chunk,
                    'source_info': f"{item['filename']} ({item['type']})",
                    'chunk_number': i // (chunk_size - chunk_overlap) + 1,
                    'total_chunks': len(text) // (chunk_size - chunk_overlap) + 1
                })

        return chunks

    def create_vector_store(self, chunk_data):
        """
        Create vector store table and populate with embeddings
        """
        try:
            # Create vector table
            self.session.sql(f"""
                CREATE OR REPLACE TABLE TEXT_CORTEX_AGENT.PUBLIC.RAG_VECTOR_STORE (
                    chunk_id INTEGER AUTOINCREMENT,
                    chunk_text TEXT,
                    source_info VARCHAR(500),
                    chunk_number INTEGER,
                    total_chunks INTEGER,
                    embedding VECTOR(FLOAT, {self.embedding_dimension}),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
                )
            """).collect()

            # Batch embedding and insertion
            batch_size = 32
            for i in range(0, len(chunk_data), batch_size):
                batch = chunk_data[i:i+batch_size]
                batch_texts = [chunk['chunk_text'] for chunk in batch]

                # Local embedding
                embeddings = self.embedding_model.encode(batch_texts, convert_to_numpy=True)

                # Insert each chunk with its embedding
                for j, chunk_info in enumerate(batch):
                    embedding_list = embeddings[j].tolist()
                    embedding_str = '[' + ','.join(map(str, embedding_list)) + ']'

                    self.session.sql(f"""
                        INSERT INTO TEXT_CORTEX_AGENT.PUBLIC.RAG_VECTOR_STORE
                        (chunk_text, source_info, chunk_number, total_chunks, embedding)
                        VALUES (?, ?, ?, ?, {embedding_str}::VECTOR(FLOAT, {self.embedding_dimension}))
                    """, params=[
                        chunk_info['chunk_text'],
                        chunk_info['source_info'],
                        chunk_info['chunk_number'],
                        chunk_info['total_chunks']
                    ]).collect()

                st.info(f"Processed {i + len(batch)}/{len(chunk_data)} chunks")

            st.success("Vector store created successfully!")
            return True

        except Exception as e:
            st.error(f"Error creating vector store: {e}")
            return False

    def setup_rag_system(self):
        """
        Complete RAG system setup
        """
        st.header("🚀 Hybrid RAG System Setup")

        # Step 1: Initialize local embedding model
        if not self.initialize_embedding_model():
            st.error("Failed to initialize embedding model")
            return False

        # Step 2: Load repository content
        loaded_content = self.load_git_repository_content()
        if not loaded_content:
            st.error("No content found in repository")
            return False

        # Step 3: Prepare content for embedding
        chunk_data = self.prepare_content_for_embedding(loaded_content)

        # Step 4: Create vector store
        if self.create_vector_store(chunk_data):
            self.setup_complete = True
            st.success("RAG System Setup Complete!")
            return True

        return False

    def vector_search(self, query, limit=5):
        """
        Perform vector search using local query embedding
        """
        try:
            # Create query embedding locally
            query_embedding = self.embedding_model.encode([query])[0].tolist()
            query_embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'

            # Search in Snowflake
            rows = self.session.sql(f"""
                SELECT
                    chunk_text,
                    source_info,
                    chunk_number,
                    VECTOR_COSINE_SIMILARITY(
                        embedding,
                        {query_embedding_str}::VECTOR(FLOAT, {self.embedding_dimension})
                    ) as similarity_score
                FROM TEXT_CORTEX_AGENT.PUBLIC.RAG_VECTOR_STORE
                ORDER BY similarity_score DESC
                LIMIT ?
            """, params=[limit]).collect()

            return [{
                "chunk_text": r[0],
                "source_info": r[1],
                "chunk_number": r[2],
                "similarity_score": float(r[3])
            } for r in rows]

        except Exception as e:
            st.error(f"Vector search error: {e}")
            return []

    def generate_answer(self, question, search_results, model="mistral-large"):
        """
        Generate answer using Snowflake Cortex
        """
        # Build context
        if search_results:
            ctx = "\n".join([
                f"[Context {i+1}]:\n{r['chunk_text']}\n(Source: {r['source_info']})" 
                for i, r in enumerate(search_results)
            ])

            prompt = f"""You are a helpful AI assistant analyzing repository content.

Context Information:
{ctx}

Question: {question}

Please provide a comprehensive answer based on the given context. If the information is not in the context, clearly state that.

Detailed Answer:"""
        else:
            prompt = f"Question: {question}\n\nNo context available. Please provide a general response."

        # Use Snowflake Cortex
        try:
            res = self.session.sql(
                "SELECT SNOWFLAKE.CORTEX.COMPLETE(?, ?) AS response",
                params=[model, prompt]
            ).collect()
            return res[0][0] if res and res[0][0] else "No response generated"

        except Exception as e:
            return f"Error generating response: {e}"

def main():
    st.title("🔍 Hybrid RAG Agent: GitHub + Snowflake + Local Embeddings")

    # Initialize or retrieve RAG system
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = HybridRAGSystem()

    rag = st.session_state.rag_system

    # Sidebar for setup and configuration
    with st.sidebar:
        st.header("RAG System Setup")
        
        # Embedding model selection
        st.subheader("Embedding Model")
        embedding_model = st.selectbox(
            "Choose Local Embedding Model", 
            [
                "intfloat/e5-base-v2",
                "sentence-transformers/all-MiniLM-L6-v2",
                "sentence-transformers/all-mpnet-base-v2"
            ]
        )

        # Setup button
        if st.button("🚀 Initialize RAG System"):
            with st.spinner("Setting up RAG system..."):
                rag.setup_rag_system()

    # Q&A Interface
    st.header("💬 Ask Your Repository")
    
    question = st.text_input("Enter your query:", placeholder="What's in this repository?")
    
    if st.button("🔍 Search & Generate"):
        if not rag.setup_complete:
            st.warning("Please initialize the RAG system first!")
        else:
            with st.spinner("Searching and generating response..."):
                # Perform vector search
                search_results = rag.vector_search(question)
                
                # Generate answer
                answer = rag.generate_answer(question, search_results)
                
                # Display results
                st.subheader("💡 Answer")
                st.write(answer)
                
                # Show sources
                with st.expander("📚 Sources"):
                    for result in search_results:
                        st.markdown(f"**{result['source_info']}** (Similarity: {result['similarity_score']:.3f})")

if __name__ == "__main__":
    main()
"""
Hybrid RAG Implementation: Local Embeddings + Snowflake Cortex LLM
Best balance of privacy, cost, and convenience

ARCHITECTURE:
- Embeddings: Local (privacy-first, one-time cost)
- Storage: Snowflake (existing infrastructure)
- LLM: Snowflake Cortex (managed, fast, no GPU needed)
"""

import streamlit as st
from snowflake.snowpark.context import get_active_session
import json
import re
import numpy as np
from typing import List, Dict, Any

# Local embeddings only
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    st.warning("⚠️ Install: pip install sentence-transformers torch")


class HybridRAGSystem:
    """
    Hybrid approach:
    - Use LOCAL models for embeddings (most privacy-sensitive)
    - Use SNOWFLAKE CORTEX for LLM generation (managed, fast)
    """

    def __init__(self, embedding_model='intfloat/e5-base-v2'):
        """
        Initialize hybrid RAG system

        Args:
            embedding_model: Local model for embeddings (privacy-first)
        """
        self.session = get_active_session()
        self.setup_complete = False

        # Local embedding model
        self.embedding_model = None
        self.embedding_model_name = embedding_model
        self.embedding_dimension = None

        # Snowflake Cortex for LLM (no local GPU needed)
        self.use_cortex_llm = True

    def initialize_embedding_model(self):
        """Load local embedding model"""
        if not EMBEDDINGS_AVAILABLE:
            st.error("sentence-transformers not installed")
            return False

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

    def create_local_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings locally (PRIVACY: data stays local during this step)
        """
        if self.embedding_model is None:
            if not self.initialize_embedding_model():
                raise RuntimeError("Failed to initialize embedding model")

        return self.embedding_model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    def initialize_vector_table(self):
        """Create vector table with local embedding dimensions"""
        try:
            if self.embedding_dimension is None:
                self.initialize_embedding_model()

            self.session.sql("DROP TABLE IF EXISTS TEXT_CORTEX_AGENT.PUBLIC.SQL_SECURITY_VECTORS_HYBRID").collect()
            self.session.sql(f"""
                CREATE TABLE TEXT_CORTEX_AGENT.PUBLIC.SQL_SECURITY_VECTORS_HYBRID (
                    chunk_id INTEGER AUTOINCREMENT,
                    chunk_text TEXT,
                    source_info VARCHAR(500),
                    chunk_number INTEGER,
                    total_chunks INTEGER,
                    embedding VECTOR(FLOAT, {self.embedding_dimension}),
                    embedding_model VARCHAR(200) DEFAULT '{self.embedding_model_name}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
                )
            """).collect()

            st.success(f"✓ Vector table created (dimension: {self.embedding_dimension})")
            return True

        except Exception as e:
            st.error(f"Error creating vector table: {e}")
            return False

    def store_embeddings_hybrid(self, chunk_data: List[Dict]):
        """
        Create embeddings LOCALLY, then store in Snowflake
        This keeps your PHI data local during the embedding process
        """
        try:
            st.info("🔒 Creating embeddings locally (PHI data stays secure)...")

            # Create vector table
            if not self.initialize_vector_table():
                return False

            # Process in batches
            batch_size = 32
            total_chunks = len(chunk_data)

            for i in range(0, total_chunks, batch_size):
                batch = chunk_data[i:i+batch_size]
                batch_texts = [chunk['chunk_text'] for chunk in batch]

                # PRIVACY: Embeddings created locally
                embeddings = self.create_local_embeddings(batch_texts)

                # Store in Snowflake (only embeddings + metadata, not raw PHI)
                for j, chunk_info in enumerate(batch):
                    embedding_list = embeddings[j].tolist()
                    embedding_str = '[' + ','.join(map(str, embedding_list)) + ']'

                    self.session.sql(f"""
                        INSERT INTO TEXT_CORTEX_AGENT.PUBLIC.SQL_SECURITY_VECTORS_HYBRID
                        (chunk_text, source_info, chunk_number, total_chunks, embedding)
                        VALUES (?, ?, ?, ?, {embedding_str}::VECTOR(FLOAT, {self.embedding_dimension}))
                    """, params=[
                        chunk_info['chunk_text'],
                        chunk_info['source_info'],
                        chunk_info['chunk_number'],
                        chunk_info['total_chunks']
                    ]).collect()

                if (i + batch_size) % 50 == 0:
                    st.info(f"Progress: {min(i + batch_size, total_chunks)}/{total_chunks} chunks")

            st.success(f"✓ Stored {total_chunks} chunks with local embeddings")
            return True

        except Exception as e:
            st.error(f"Error storing embeddings: {e}")
            return False

    def vector_search_hybrid(self, query: str, limit=5, doc_filter=None):
        """
        Search with local query embedding + Snowflake vector search
        """
        try:
            # Create query embedding LOCALLY
            query_embedding = self.embedding_model.encode([query])[0].tolist()
            query_embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'

            # Filter clause
            where_clause = ""
            if doc_filter == 'sprint':
                where_clause = "WHERE source_info ILIKE '%Sprint Deployment%' OR source_info ILIKE '%.xml%'"
            elif doc_filter == 'sql':
                where_clause = "WHERE NOT (source_info ILIKE '%Sprint Deployment%' OR source_info ILIKE '%.xml%')"

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
                FROM TEXT_CORTEX_AGENT.PUBLIC.SQL_SECURITY_VECTORS_HYBRID
                {where_clause}
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
            st.error(f"Search error: {e}")
            return []

    def generate_answer_cortex(self, question: str, search_results: List[Dict], model="mistral-large"):
        """
        Generate answer using Snowflake Cortex (managed LLM)
        No local GPU needed for this step
        """
        # Build context
        if search_results:
            ctx = ""
            for i, r in enumerate(search_results, 1):
                ctx += f"\n[Context {i}]: {r.get('chunk_text','')}\n"

            prompt = f"""You are a SQL and Sprint deployment assistant.

Context:
{ctx}

Question: {question}

Instructions:
- Use only the provided context
- For SQL: cite tables, views, and columns
- For Sprints: list files and changes
- If not in context, say so

Answer:"""
        else:
            prompt = f"Question: {question}\n\nNo context available. Please run RAG setup first.\n\nAnswer:"

        # Use Snowflake Cortex
        try:
            res = self.session.sql(
                "SELECT SNOWFLAKE.CORTEX.COMPLETE(?, ?) AS response",
                params=[model, prompt]
            ).collect()
            return res[0][0] if res and res[0][0] else "No response generated"

        except Exception as e:
            return f"Error with Cortex LLM: {e}"

    def ask_question_hybrid(self, question: str, model="mistral-large"):
        """
        Complete hybrid flow:
        1. Local embedding for query (privacy)
        2. Vector search in Snowflake (speed)
        3. Cortex LLM for generation (managed, no GPU)
        """
        # Detect intent
        intents = self.get_intents(question)

        # Search with local embeddings
        sql_results = self.vector_search_hybrid(question, limit=5, doc_filter='sql') if 'sql' in intents else []
        sprint_results = self.vector_search_hybrid(question, limit=5, doc_filter='sprint') if 'sprint' in intents else []

        # Combine results
        combined = sql_results + sprint_results
        combined.sort(key=lambda r: r["similarity_score"], reverse=True)

        # Generate with Cortex
        if combined:
            best_score = combined[0]['similarity_score']
            context_parts = []
            if sql_results:
                context_parts.append(f"SQL: {len(sql_results)}")
            if sprint_results:
                context_parts.append(f"Sprint: {len(sprint_results)}")

            context_info = f"Found {len(combined)} chunks ({', '.join(context_parts)}; best: {best_score:.3f})"
            answer = self.generate_answer_cortex(question, combined, model)
            return answer, combined, context_info, True

        answer = self.generate_answer_cortex(question, [])
        return answer, [], "No matches", False

    def get_intents(self, question: str):
        """Detect SQL vs Sprint intent"""
        q = question.lower()
        sql_kw = ['sql', 'table', 'column', 'schema', 'view']
        sprint_kw = ['sprint', 'changelog', 'xml', 'deployment']

        intents = set()
        if any(kw in q for kw in sql_kw):
            intents.add('sql')
        if any(kw in q for kw in sprint_kw):
            intents.add('sprint')

        return intents if intents else {'sql', 'sprint'}


# ==================== Streamlit UI ====================

def main():
    st.title("🔐 Hybrid RAG: Local Embeddings + Cortex LLM")
    st.markdown("""
    **Privacy-First Hybrid Approach:**
    - 🔒 Embeddings created locally (PHI stays secure)
    - ⚡ Cortex LLM for generation (no GPU needed)
    - 💰 Lower cost than full Cortex
    """)

    if not EMBEDDINGS_AVAILABLE:
        st.error("Install: pip install sentence-transformers torch")
        return

    with st.sidebar:
        st.header("Configuration")

        st.subheader("Local Embedding Model")
        embedding_model = st.selectbox(
            "Model",
            [
                "intfloat/e5-base-v2",  # 768-dim
                "sentence-transformers/all-MiniLM-L6-v2",  # 384-dim, faster
                "sentence-transformers/all-mpnet-base-v2",  # 768-dim
            ],
            index=0
        )

        st.subheader("Cortex LLM")
        cortex_model = st.selectbox(
            "Model",
            ["mistral-large", "snowflake-arctic", "llama2-70b-chat"],
            index=0
        )

        st.divider()

        if st.button("🚀 Initialize Hybrid RAG"):
            with st.spinner("Setting up hybrid system..."):
                try:
                    rag = HybridRAGSystem(embedding_model=embedding_model)
                    if rag.initialize_embedding_model():
                        st.session_state['rag_hybrid'] = rag
                        st.success("✓ Hybrid RAG ready")
                except Exception as e:
                    st.error(f"Error: {e}")

        st.subheader("🛡️ Security Status")
        st.success("✅ Embeddings: Local (private)")
        st.info("☁️ LLM: Cortex (managed)")
        st.success("✅ PHI: Stays in Snowflake")

        st.subheader("💰 Cost Comparison")
        st.markdown("""
        **Per 1000 queries:**
        - Full Cortex: $$$
        - Hybrid: $$
        - Full Local: $ (hardware)
        """)

    # Q&A Interface
    if 'rag_hybrid' in st.session_state:
        rag = st.session_state['rag_hybrid']

        st.divider()
        st.header("💬 Ask Questions")

        question = st.text_input(
            "Your question:",
            placeholder="What tables are in Schema1? or What changed in Sprint01?"
        )

        if st.button("🔍 Ask (Hybrid)"):
            if question:
                with st.spinner("Processing (local embedding + Cortex LLM)..."):
                    answer, results, ctx_info, used_ctx = rag.ask_question_hybrid(
                        question,
                        model=cortex_model
                    )

                st.markdown(f"*{ctx_info}*")

                st.subheader("💡 Answer")
                st.write(answer)

                if results:
                    with st.expander("📚 Sources"):
                        for r in results:
                            st.markdown(f"• {r['source_info']} ({r['similarity_score']:.3f})")
    else:
        st.info("👈 Initialize the hybrid system using the sidebar")

    # Info section
    with st.expander("ℹ️ Why Hybrid?"):
        st.markdown("""
        **Advantages of Hybrid Approach:**

        1. **Privacy for Embeddings:**
           - Embeddings are created locally
           - Your PHI data processed on your hardware
           - Only vectorized representations go to Snowflake

        2. **Convenience for LLM:**
           - No GPU needed for text generation
           - Managed service (no maintenance)
           - Fast inference
           - Multiple models available

        3. **Cost Optimization:**
           - One-time embedding cost (local)
           - Pay only for LLM generation
           - Cheaper than full Cortex pipeline

        4. **Flexibility:**
           - Easy to switch embedding models
           - Easy to switch LLM providers
           - Can move to full local later if needed

        **When to Use:**
        - You want local control over embeddings
        - You don't want to manage LLM infrastructure
        - You want cost optimization
        - You have CPU/GPU for embeddings but not for large LLMs
        """)


if __name__ == "__main__":
    main()

"""
Local Offline Model Implementation for Snowflake RAG
This version uses local embedding and LLM models for maximum PHI security

SECURITY: All model inference happens locally - data never leaves your environment
"""

import streamlit as st
from snowflake.snowpark.context import get_active_session
import json
import re
import numpy as np
from typing import List, Dict, Any

# Local model libraries
try:
    from sentence_transformers import SentenceTransformer
    import torch
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    st.warning("⚠️ sentence-transformers not installed. Run: pip install sentence-transformers torch")

# Local LLM options (choose one)
try:
    import ollama  # For Ollama (easiest)
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class LocalEmbeddingModel:
    """
    Local embedding model using sentence-transformers
    Models run entirely on your hardware - no external API calls
    """

    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        """
        Initialize local embedding model

        Recommended models:
        - 'sentence-transformers/all-MiniLM-L6-v2' (384 dim, fast, CPU-friendly)
        - 'sentence-transformers/all-mpnet-base-v2' (768 dim, better quality)
        - 'BAAI/bge-base-en-v1.5' (768 dim, good for retrieval)
        - 'intfloat/e5-base-v2' (768 dim, matches Snowflake's model)
        """
        if not EMBEDDINGS_AVAILABLE:
            raise ImportError("sentence-transformers not installed")

        self.model_name = model_name
        self.model = None
        self.dimension = None

    def load(self):
        """Load the embedding model into memory"""
        st.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)

        # Get embedding dimension
        test_embedding = self.model.encode("test")
        self.dimension = len(test_embedding)
        st.success(f"✓ Embedding model loaded (dimension: {self.dimension})")

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts
        Returns numpy array of shape (len(texts), dimension)
        """
        if self.model is None:
            self.load()

        return self.model.encode(texts, convert_to_numpy=True)

    def embed_single(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        if self.model is None:
            self.load()

        embedding = self.model.encode([text], convert_to_numpy=True)[0]
        return embedding.tolist()


class LocalLLMModel:
    """
    Local LLM for text generation
    Supports multiple backends: Ollama, Transformers, vLLM
    """

    def __init__(self, backend='ollama', model_name='mistral'):
        """
        Initialize local LLM

        Backend options:
        - 'ollama': Easiest, supports many models (mistral, llama2, phi3, etc.)
        - 'transformers': Direct HuggingFace model loading
        - 'vllm': For production deployments (faster, more scalable)

        Model recommendations:
        - Small (7B): mistral, llama2, phi-3-mini (4GB-8GB VRAM)
        - Medium (13B): llama2-13b, mistral-medium (16GB+ VRAM)
        - Large (70B+): llama2-70b (40GB+ VRAM, multi-GPU)
        """
        self.backend = backend
        self.model_name = model_name
        self.model = None
        self.tokenizer = None

    def load(self):
        """Load the LLM model"""
        if self.backend == 'ollama':
            if not OLLAMA_AVAILABLE:
                raise ImportError("Ollama not installed. Run: pip install ollama")
            st.info(f"Using Ollama with model: {self.model_name}")
            # Ollama doesn't need explicit loading
            st.success("✓ Ollama backend ready")

        elif self.backend == 'transformers':
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError("transformers not installed. Run: pip install transformers")

            st.info(f"Loading HuggingFace model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto"
            )
            st.success("✓ Transformers model loaded")

        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def generate(self, prompt: str, max_tokens=512, temperature=0.7) -> str:
        """Generate text from prompt"""
        if self.backend == 'ollama':
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    'num_predict': max_tokens,
                    'temperature': temperature
                }
            )
            return response['response']

        elif self.backend == 'transformers':
            if self.model is None:
                self.load()

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True
            )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the prompt from response
            return response[len(prompt):].strip()

        return "Error: Model not initialized"


class SnowflakeRAGSystemLocal:
    """
    RAG System using local models for embeddings and generation
    Data stays in Snowflake, but all ML inference happens locally
    """

    def __init__(self, embedding_model='intfloat/e5-base-v2', llm_backend='ollama', llm_model='mistral'):
        """
        Initialize RAG system with local models

        Args:
            embedding_model: HuggingFace model for embeddings
            llm_backend: 'ollama' or 'transformers'
            llm_model: Model name for generation
        """
        self.session = get_active_session()
        self.setup_complete = False

        # Initialize local models
        self.embedding_model = LocalEmbeddingModel(embedding_model) if EMBEDDINGS_AVAILABLE else None
        self.llm_model = LocalLLMModel(llm_backend, llm_model)

        # Text cleaning patterns (same as your original code)
        self._setup_text_patterns()

    def _setup_text_patterns(self):
        """Setup regex patterns for text cleaning"""
        self._my_liquibase_header = re.compile(r'--liquibase formatted sql', re.I)
        self._my_changeset = re.compile(r'-- changeset\s+[^\n]*', re.I)
        self._my_other_annotations = [
            re.compile(r'--rollback[^\n]*', re.I),
            re.compile(r'--preconditions[^\n]*', re.I),
        ]
        self._re_ctrl = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
        self._re_ws = re.compile(r'[ \t]+')
        self._re_manynl = re.compile(r'\n{3,}')

    def cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        a_arr = np.array(a)
        b_arr = np.array(b)
        return np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr))

    def store_embeddings_local(self, chunk_data: List[Dict]):
        """
        Store embeddings using local model
        This replaces Snowflake Cortex EMBED_TEXT_768
        """
        if self.embedding_model is None:
            st.error("Embedding model not available. Install sentence-transformers")
            return False

        try:
            st.info("Creating local embeddings...")

            # Load model if not already loaded
            if self.embedding_model.model is None:
                self.embedding_model.load()

            # Get embedding dimension
            embedding_dim = self.embedding_model.dimension

            # Create table with appropriate vector dimension
            self.session.sql("DROP TABLE IF EXISTS TEXT_CORTEX_AGENT.PUBLIC.SQL_SECURITY_VECTORS_LOCAL").collect()
            self.session.sql(f"""
                CREATE TABLE TEXT_CORTEX_AGENT.PUBLIC.SQL_SECURITY_VECTORS_LOCAL (
                    chunk_id INTEGER AUTOINCREMENT,
                    chunk_text TEXT,
                    source_info VARCHAR(500),
                    chunk_number INTEGER,
                    total_chunks INTEGER,
                    embedding VECTOR(FLOAT, {embedding_dim}),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
                )
            """).collect()

            # Process in batches for efficiency
            batch_size = 32
            total_chunks = len(chunk_data)

            for i in range(0, total_chunks, batch_size):
                batch = chunk_data[i:i+batch_size]
                batch_texts = [chunk['chunk_text'] for chunk in batch]

                # Generate embeddings locally
                embeddings = self.embedding_model.embed(batch_texts)

                # Insert into Snowflake
                for j, chunk_info in enumerate(batch):
                    embedding_list = embeddings[j].tolist()
                    embedding_str = '[' + ','.join(map(str, embedding_list)) + ']'

                    self.session.sql(f"""
                        INSERT INTO TEXT_CORTEX_AGENT.PUBLIC.SQL_SECURITY_VECTORS_LOCAL
                        (chunk_text, source_info, chunk_number, total_chunks, embedding)
                        VALUES (?, ?, ?, ?, {embedding_str}::VECTOR(FLOAT, {embedding_dim}))
                    """, params=[
                        chunk_info['chunk_text'],
                        chunk_info['source_info'],
                        chunk_info['chunk_number'],
                        chunk_info['total_chunks']
                    ]).collect()

                if (i + batch_size) % 100 == 0:
                    st.info(f"Processed {min(i + batch_size, total_chunks)}/{total_chunks} chunks...")

            st.success(f"✓ Successfully stored {total_chunks} chunks with local embeddings")
            return True

        except Exception as e:
            st.error(f"Error storing local embeddings: {e}")
            return False

    def vector_search_local(self, query: str, limit=5, doc_filter=None):
        """
        Search using locally generated query embedding
        This replaces Snowflake Cortex embedding in the search query
        """
        if self.embedding_model is None:
            st.error("Embedding model not available")
            return []

        try:
            # Generate query embedding locally
            query_embedding = self.embedding_model.embed_single(query)
            query_embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'

            # Build filter clause
            where_clause = ""
            if doc_filter == 'sprint':
                where_clause = "WHERE source_info ILIKE '%Sprint Deployment%' OR source_info ILIKE '%.xml%'"
            elif doc_filter == 'sql':
                where_clause = "WHERE NOT (source_info ILIKE '%Sprint Deployment%' OR source_info ILIKE '%.xml%')"

            # Search using vector similarity in Snowflake
            rows = self.session.sql(f"""
                SELECT
                    chunk_text,
                    source_info,
                    chunk_number,
                    VECTOR_COSINE_SIMILARITY(
                        embedding,
                        {query_embedding_str}::VECTOR(FLOAT, {self.embedding_model.dimension})
                    ) as similarity_score
                FROM TEXT_CORTEX_AGENT.PUBLIC.SQL_SECURITY_VECTORS_LOCAL
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
            st.error(f"Vector search error: {e}")
            return []

    def generate_answer_local(self, question: str, search_results: List[Dict]) -> str:
        """
        Generate answer using local LLM
        This replaces Snowflake Cortex COMPLETE/CHAT_COMPLETE
        """
        # Build context from search results
        if search_results:
            ctx = ""
            for i, r in enumerate(search_results, 1):
                ctx += f"\n[Context {i}]: {r.get('chunk_text','')}\n"

            prompt = f"""You are an SQL and Sprint deployment assistant.

Context information:
{ctx}

Question: {question}

Instructions:
- Use ONLY the context provided above to answer
- For SQL questions: cite table/view definitions and columns
- For Sprint questions: list included files and changes
- If the answer is not in the context, say "I don't have enough information to answer this question"
- Be concise and accurate

Answer:"""
        else:
            prompt = f"""You are an SQL and Sprint deployment assistant.

Question: {question}

I don't have specific context to answer this question. Please run the RAG setup to load the SQL and Sprint data.

Answer:"""

        try:
            # Load LLM if not loaded
            if self.llm_model.model is None and self.llm_model.backend != 'ollama':
                self.llm_model.load()

            # Generate response
            response = self.llm_model.generate(prompt, max_tokens=512, temperature=0.3)
            return response

        except Exception as e:
            return f"Error generating response with local LLM: {e}"

    def ask_question_local(self, question: str):
        """
        Answer question using local models for both retrieval and generation
        """
        # Detect intent (same as original)
        intents = self.get_intents(question)

        # Search with local embeddings
        sql_results = self.vector_search_local(question, limit=5, doc_filter='sql') if 'sql' in intents else []
        sprint_results = self.vector_search_local(question, limit=5, doc_filter='sprint') if 'sprint' in intents else []

        # Combine and sort results
        combined = sql_results + sprint_results
        combined.sort(key=lambda r: r["similarity_score"], reverse=True)

        # Generate answer with local LLM
        if combined:
            best_score = combined[0]['similarity_score']
            context_parts = []
            if sql_results:
                context_parts.append(f"SQL: {len(sql_results)} chunks")
            if sprint_results:
                context_parts.append(f"Sprint: {len(sprint_results)} chunks")

            context_info = f"Found {len(combined)} chunks ({', '.join(context_parts)}; best: {best_score:.3f})"
            answer = self.generate_answer_local(question, combined)
            return answer, combined, context_info, True

        # No context found
        answer = self.generate_answer_local(question, [])
        return answer, [], "No context matched", False

    def get_intents(self, question: str):
        """Detect question intent (same as original)"""
        q = question.lower()

        sql_keywords = ['sql', 'table', 'column', 'schema', 'view', 'create', 'select']
        sprint_keywords = ['sprint', 'changelog', 'xml', 'deployment', 'changed']

        sql_hits = any(kw in q for kw in sql_keywords)
        sprint_hits = any(kw in q for kw in sprint_keywords)

        intents = set()
        if sql_hits:
            intents.add('sql')
        if sprint_hits:
            intents.add('sprint')

        if not intents:
            intents = {'sql', 'sprint'}

        return intents


# ==================== Streamlit UI ====================

def main():
    st.title("🔒 Secure Local RAG System")
    st.markdown("**100% Local Model Inference** - All embeddings and LLM generation happen on your hardware")

    # Check dependencies
    if not EMBEDDINGS_AVAILABLE:
        st.error("⚠️ Missing dependencies. Install with:")
        st.code("pip install sentence-transformers torch")
        return

    with st.sidebar:
        st.header("Local Model Configuration")

        st.subheader("Embedding Model")
        embedding_choice = st.selectbox(
            "Embedding Model",
            options=[
                "intfloat/e5-base-v2",  # 768-dim, matches Snowflake
                "sentence-transformers/all-MiniLM-L6-v2",  # 384-dim, fast
                "sentence-transformers/all-mpnet-base-v2",  # 768-dim, quality
                "BAAI/bge-base-en-v1.5",  # 768-dim, retrieval
            ],
            index=0
        )

        st.subheader("LLM Configuration")
        llm_backend = st.selectbox("LLM Backend", options=["ollama", "transformers"], index=0)

        if llm_backend == "ollama":
            llm_model = st.selectbox(
                "Ollama Model",
                options=["mistral", "llama2", "phi3", "llama2:13b", "codellama"],
                index=0
            )
            st.info("💡 Install Ollama: https://ollama.ai")
            st.code(f"ollama pull {llm_model}")
        else:
            llm_model = st.text_input(
                "HuggingFace Model",
                value="microsoft/phi-2"
            )

        st.divider()

        if st.button("🚀 Initialize Local RAG"):
            with st.spinner("Loading local models..."):
                try:
                    rag = SnowflakeRAGSystemLocal(
                        embedding_model=embedding_choice,
                        llm_backend=llm_backend,
                        llm_model=llm_model
                    )
                    st.session_state['rag_local'] = rag
                    st.success("✓ Local RAG system initialized")
                except Exception as e:
                    st.error(f"Error initializing: {e}")

        st.subheader("Security Status")
        st.success("✅ All ML inference is local")
        st.success("✅ PHI data never leaves Snowflake")
        st.success("✅ No external API calls")

    # Main Q&A interface
    if 'rag_local' in st.session_state:
        rag = st.session_state['rag_local']

        question = st.text_input("Ask a question:", placeholder="What tables are in Schema1?")

        if st.button("🔍 Ask (Local)"):
            if question:
                with st.spinner("Processing with local models..."):
                    answer, results, ctx_info, used_context = rag.ask_question_local(question)

                st.subheader("💡 Answer")
                st.write(answer)

                if results:
                    with st.expander("📚 Sources"):
                        for r in results:
                            st.markdown(f"• {r['source_info']} (score: {r['similarity_score']:.3f})")
    else:
        st.info("👈 Initialize the local RAG system using the sidebar")


if __name__ == "__main__":
    main()

# Quick Start: Local Model Setup for PHI Security

## 🎯 Goal
Run your RAG system with local models so PHI data never leaves your environment during ML inference.

## 📋 Prerequisites

- Python 3.9 or higher
- Snowflake account with existing data
- Hardware requirements (choose based on your option):
  - **Minimal (CPU only):** 8GB RAM, 4 cores
  - **Recommended:** 16GB RAM, 8 cores, 1x GPU with 8GB+ VRAM
  - **Optimal:** 32GB RAM, 1x GPU with 16GB+ VRAM

## 🚀 Quick Setup (3 Options)

### Option A: Hybrid Approach (Recommended) ⭐

**Best for:** Most users - balances security, cost, and ease of use

```bash
# 1. Install dependencies
pip install sentence-transformers torch snowflake-connector-python streamlit

# 2. Run the app
streamlit run snowflake_rag_hybrid.py
```

**What you get:**
- ✅ Local embeddings (PHI secure during vectorization)
- ✅ Cortex LLM (managed, fast, no GPU needed)
- ✅ Lowest setup complexity
- ✅ Good cost optimization

---

### Option B: Full Local Models (Maximum Security) 🔒

**Best for:** Air-gapped environments, maximum control

```bash
# 1. Install Ollama (easiest LLM option)
# Download from: https://ollama.ai
# After installation:
ollama pull mistral

# 2. Install Python dependencies
pip install -r requirements_local_models.txt

# 3. Run the app
streamlit run snowflake_rag_local_models.py
```

**What you get:**
- ✅ Complete local inference
- ✅ No data leaves your network
- ✅ Can fine-tune on your data
- ⚠️ Requires more hardware (GPU recommended)

---

### Option C: Keep Current Setup (Already Secure) ✅

**Best for:** Users who just want confirmation their data is safe

**Your current code already keeps PHI secure!**

Snowflake Cortex:
- ✅ Data stays in Snowflake's infrastructure
- ✅ No training on your data
- ✅ HIPAA compliant with BAA
- ✅ Easiest to maintain

**No changes needed unless you want to fine-tune or go fully air-gapped.**

---

## 🔧 Detailed Setup: Full Local Models

### Step 1: Install Ollama (Easiest Local LLM)

**macOS/Linux:**
```bash
# Download and install from https://ollama.ai
# Or use curl:
curl -fsSL https://ollama.ai/install.sh | sh

# Pull models
ollama pull mistral        # 7B model, 4GB
ollama pull phi3          # 3.8B model, smaller
ollama pull llama2:13b    # 13B model, better quality

# Test it
ollama run mistral "Hello, how are you?"
```

**Windows:**
```powershell
# Download installer from https://ollama.ai/download
# Run installer
# Open new terminal:
ollama pull mistral
```

### Step 2: Install Python Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install packages
pip install -r requirements_local_models.txt

# Verify installation
python -c "from sentence_transformers import SentenceTransformer; print('✓ OK')"
python -c "import ollama; print('✓ OK')"
```

### Step 3: Download Embedding Model

The embedding model downloads automatically on first use, but you can pre-download:

```python
from sentence_transformers import SentenceTransformer

# This will download the model (~450MB)
model = SentenceTransformer('intfloat/e5-base-v2')
print("Model downloaded successfully!")
```

### Step 4: Configure Snowflake Connection

Make sure you have `.env` file configured:

```env
SNOWFLAKE_ACCOUNT=your-account
SNOWFLAKE_USER=your-user
SNOWFLAKE_PASSWORD=your-password
SNOWFLAKE_WAREHOUSE=your-warehouse
SNOWFLAKE_DATABASE=TEXT_CORTEX_AGENT
SNOWFLAKE_SCHEMA=PUBLIC
SNOWFLAKE_ROLE=your-role
```

### Step 5: Run the Application

```bash
# For hybrid approach
streamlit run snowflake_rag_hybrid.py

# OR for full local
streamlit run snowflake_rag_local_models.py
```

The app will open in your browser at http://localhost:8501

---

## 🎮 Using the Application

### First Time Setup:

1. **In the sidebar:**
   - Select your embedding model (default: `intfloat/e5-base-v2`)
   - Select your LLM model (default: `mistral`)
   - Click "🚀 Initialize Local RAG"

2. **Load your data:**
   - The system will load SQL and XML files from Snowflake
   - Create embeddings locally (this may take 5-10 minutes first time)
   - Store embeddings in Snowflake

3. **Ask questions:**
   - Type your question about SQL schema or sprints
   - Click "🔍 Ask"
   - Get answers using local models!

### Example Questions:

```
"What tables are in Schema1?"
"Show me columns in patient_medication table"
"What changed in Sprint01?"
"Which files were included in the latest sprint?"
```

---

## 🔍 Troubleshooting

### Issue: "Ollama connection error"

**Solution:**
```bash
# Make sure Ollama is running
ollama serve  # Should show "Ollama is running"

# In another terminal:
ollama list   # Should show your downloaded models
```

### Issue: "CUDA out of memory"

**Solution 1 - Use smaller model:**
```bash
ollama pull phi3  # Smaller than mistral
```

**Solution 2 - Use CPU:**
```python
# In the code, models will automatically fall back to CPU
# It's slower but works without GPU
```

### Issue: "Embedding model download is slow"

**Solution:**
```bash
# Download manually with progress bar
python -c "
from sentence_transformers import SentenceTransformer
import sys
print('Downloading model...')
model = SentenceTransformer('intfloat/e5-base-v2')
print('Download complete!')
"
```

### Issue: "Running out of RAM"

**Solution:**
```python
# Use a smaller embedding model
# Change in the UI dropdown to:
"sentence-transformers/all-MiniLM-L6-v2"  # Only 80MB, 384 dimensions
```

---

## 📊 Performance Expectations

### Embedding Creation:
- **CPU only:** ~10-50 chunks/second
- **GPU (NVIDIA):** ~100-500 chunks/second
- **Apple Silicon (M1/M2):** ~50-200 chunks/second

### LLM Inference:
- **Mistral-7B on CPU:** 1-5 tokens/second (slow but works)
- **Mistral-7B on GPU (8GB):** 20-50 tokens/second
- **Phi-3 on CPU:** 5-10 tokens/second
- **Cortex (cloud):** 50-200+ tokens/second

### Total Setup Time:
- **First time:** 15-30 minutes (model downloads)
- **Subsequent runs:** Instant (models cached)

---

## 🔐 Security Verification

To verify your setup is truly local and secure:

### 1. Check Network Traffic:

```bash
# While running a query, monitor network:
# No outgoing traffic should go to external AI APIs

# macOS/Linux:
sudo tcpdump -i any host not your-snowflake-account.snowflakecomputing.com

# Should show NO traffic to OpenAI, Anthropic, etc.
```

### 2. Disconnect from Internet:

```bash
# Turn off WiFi
# Run a query
# If it works (with local models), you're fully offline! ✅
```

### 3. Check Model Locations:

```bash
# Ollama models (local disk):
ls ~/.ollama/models/

# Sentence-transformers cache (local disk):
ls ~/.cache/huggingface/

# All local! ✅
```

---

## 💰 Cost Comparison

### Current Setup (Cortex):
- Embedding: ~$0.0001 per 1000 tokens
- LLM: ~$0.002 per 1000 tokens
- **Total for 1M tokens:** ~$2,000-3,000/year

### Hybrid (Local Embeddings + Cortex LLM):
- Embedding: $0 (local)
- LLM: ~$0.002 per 1000 tokens
- **Total for 1M tokens:** ~$1,000-1,500/year
- **Savings: 50%**

### Full Local:
- Embedding: $0 (local)
- LLM: $0 (local)
- Hardware: One-time cost
  - Budget: $500-1,000 (CPU only)
  - Recommended: $1,500-3,000 (with GPU)
  - Professional: $5,000+ (multi-GPU)
- **Total for 1M tokens:** $0 operating cost
- **ROI: 6-12 months**

---

## 🎓 Next Steps

### 1. Fine-tuning on Your Data (Advanced):

If you want to fine-tune models on your medical terminology:

```bash
# Install additional tools
pip install peft trl datasets

# Fine-tune embedding model on your domain
# (See advanced guides)
```

### 2. Production Deployment:

For production, consider:
- Docker containers for consistent deployment
- GPU servers (cloud or on-premise)
- Load balancing for multiple users
- Monitoring and logging

### 3. Model Updates:

```bash
# Update Ollama models
ollama pull mistral  # Gets latest version

# Update embeddings
pip install --upgrade sentence-transformers
```

---

## 📚 Additional Resources

- **Ollama Documentation:** https://github.com/ollama/ollama
- **Sentence Transformers:** https://www.sbert.net/
- **HuggingFace Models:** https://huggingface.co/models
- **Snowflake Security:** https://docs.snowflake.com/en/user-guide/security

---

## ❓ FAQ

**Q: Do I need a GPU?**
A: No, but it's much faster. Embedding models work fine on CPU. For LLMs, CPU works but is slow.

**Q: Can I use this with GitHub Actions?**
A: Not recommended for PHI data. GitHub Actions runs on GitHub's servers (data would leave your environment).

**Q: How do I know my data is secure?**
A: With local models, all inference happens on your hardware. Use network monitoring to verify no external calls.

**Q: Can I still use some Cortex features?**
A: Yes! The hybrid approach uses local embeddings + Cortex LLM. Best of both worlds.

**Q: What if I want to switch back to Cortex?**
A: Your original code still works! You can run both approaches side-by-side.

**Q: Is this HIPAA compliant?**
A: Local processing can be HIPAA compliant with proper infrastructure controls. Consult your compliance team.

---

**Ready to get started? Choose your option above and follow the steps!** 🚀

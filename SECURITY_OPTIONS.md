# PHI Security Options for RAG System

## Current Setup: Snowflake Cortex

Your current implementation uses Snowflake Cortex, which:
- ✅ Keeps data within Snowflake's security boundary
- ✅ Does NOT train on your data
- ✅ Uses pre-trained models that run on Snowflake infrastructure
- ✅ Data never leaves your Snowflake account
- ✅ Compliant with most PHI/HIPAA requirements when Snowflake BAA is in place

**Snowflake's Cortex LLM Functions are already secure for PHI data.**

## Why Consider Alternatives?

You might want more control if:
1. You need to **fine-tune** models on your specific medical terminology
2. You want **complete air-gapped** deployment
3. You need to **audit every inference** at the model level
4. You want to **reduce costs** for high-volume queries
5. Your compliance team requires **on-premise** model hosting

---

## Option 1: Local Offline Models (Maximum Control) ⭐

**Best for:** Complete control, air-gapped environments, fine-tuning on PHI

### Architecture:
```
User → Streamlit (Local) → Snowflake (Data Only) → Local Models (Embedding + LLM)
```

### Pros:
- ✅ Complete control over model and data flow
- ✅ Can fine-tune on your PHI data safely
- ✅ No data leaves your network
- ✅ One-time setup cost, no per-query fees
- ✅ Works offline/air-gapped

### Cons:
- ❌ Requires GPU hardware or powerful CPU
- ❌ More complex setup and maintenance
- ❌ You manage model updates
- ❌ Slower inference without proper hardware

### Implementation:
See `snowflake_rag_local_models.py`

**Estimated Hardware:**
- Embedding: CPU-only works (2-4 cores, 8GB RAM)
- Small LLM (Phi-3, Mistral-7B): 1x GPU with 16GB+ VRAM
- Medium LLM (Mistral-Large): 2-4x GPUs with 24GB+ VRAM each

---

## Option 2: Snowpark Container Services (Recommended Balance) ⭐⭐⭐

**Best for:** Staying in Snowflake ecosystem with custom models

### Architecture:
```
User → Streamlit (Snowflake) → Snowflake DB → Snowpark Containers (Your Models)
```

### Pros:
- ✅ Data never leaves Snowflake
- ✅ Use your own models (open source or fine-tuned)
- ✅ Leverage Snowflake's compute infrastructure
- ✅ Can fine-tune models safely within Snowflake
- ✅ Integrated with Snowflake security/governance
- ✅ Scales automatically

### Cons:
- ❌ Requires Snowpark Container Services setup
- ❌ More complex than Cortex (but simpler than fully local)
- ❌ Still incurs Snowflake compute costs

### Implementation:
See `snowpark_containers_setup/` directory

---

## Option 3: Hybrid Approach (Best of Both)

**Architecture:**
```
- Embeddings: Local models (fast, cheap, privacy)
- Data Storage: Snowflake (existing infrastructure)
- LLM: Snowflake Cortex OR Local (choose based on needs)
```

### Pros:
- ✅ Embeddings run locally (most privacy-sensitive operation)
- ✅ Can still use Cortex LLMs for generation (faster)
- ✅ Flexible: swap components as needs evolve
- ✅ Lower cost than full Cortex

### Cons:
- ❌ Split architecture to maintain
- ❌ Need to manage embedding model updates

### Implementation:
See `snowflake_rag_hybrid.py`

---

## GitHub Actions - NOT Recommended ❌

**Why NOT GitHub Actions:**
- GitHub Actions runs on GitHub's infrastructure (data leaves your environment)
- Designed for CI/CD, not model training/inference
- Cannot access your Snowflake data securely during inference
- Would require exporting PHI data to GitHub (compliance violation)

**What GitHub Actions CAN do:**
- Automate deployment of your model containers
- Run tests on synthetic/de-identified data
- Deploy infrastructure as code
- NOT suitable for handling PHI data directly

---

## Security Comparison Table

| Feature | Current (Cortex) | Local Models | Snowpark Containers | Hybrid |
|---------|-----------------|--------------|---------------------|---------|
| Data leaves Snowflake? | No | No* | No | No* |
| Fine-tune on PHI? | No | Yes | Yes | Yes |
| Air-gapped capable? | No | Yes | No | Partial |
| Setup complexity | ⭐ Easy | ⭐⭐⭐⭐⭐ Hard | ⭐⭐⭐ Medium | ⭐⭐⭐⭐ Hard |
| Operating cost | $$ | $ (hardware) | $$ | $ |
| Inference speed | Fast | Varies | Fast | Fast |
| Model updates | Automatic | Manual | Manual | Manual |
| HIPAA compliant** | Yes | Yes | Yes | Yes |

\* Data only leaves Snowflake to go to your secure local environment
\** With proper configuration and BAA in place

---

## Recommendations

### If you're concerned about PHI security with current setup:
**→ Your current Snowflake Cortex setup is already secure!**
- Verify you have a BAA (Business Associate Agreement) with Snowflake
- Review Snowflake's HIPAA compliance documentation
- No changes needed unless you have specific requirements

### If you want to fine-tune models on PHI data:
**→ Use Option 2: Snowpark Container Services**
- Keeps everything in Snowflake
- Allows custom/fine-tuned models
- Easier than managing local infrastructure

### If you need complete air-gapped deployment:
**→ Use Option 1: Local Models**
- Maximum control and isolation
- Requires significant infrastructure investment
- Best for highly sensitive environments

### If you want cost optimization:
**→ Use Option 3: Hybrid**
- Local embeddings (cheap, private)
- Cortex LLM only when needed (pay per use)
- Best balance of cost and convenience

---

## Next Steps

1. **Verify Current Security:** Check if you have Snowflake BAA for HIPAA compliance
2. **Assess Requirements:** Do you need fine-tuning? Air-gap? Cost reduction?
3. **Choose Option:** Based on your specific needs (see recommendations above)
4. **Implement:** Use the provided code examples
5. **Test:** Validate with de-identified data first
6. **Deploy:** Roll out to production with proper monitoring

---

## Questions to Consider

1. **Do you have a Business Associate Agreement (BAA) with Snowflake?**
   - If YES → Current Cortex setup is likely compliant
   - If NO → Contact Snowflake to establish BAA

2. **Do you need to fine-tune models on medical terminology?**
   - If YES → Consider Snowpark Containers or Local Models
   - If NO → Current Cortex is sufficient

3. **What's your budget for infrastructure?**
   - Limited → Stay with Cortex
   - Moderate → Snowpark Containers
   - Significant → Local Models

4. **Do you need air-gapped deployment?**
   - YES → Local Models only
   - NO → Cortex or Snowpark Containers work

5. **What's your technical team's capacity?**
   - Small team → Stick with Cortex (managed service)
   - DevOps/MLOps team → Any option works

---

## Additional Resources

- [Snowflake HIPAA Compliance](https://www.snowflake.com/wp-content/uploads/2020/03/Snowflake-HIPAA-Whitepaper.pdf)
- [Snowpark Container Services](https://docs.snowflake.com/en/developer-guide/snowpark-container-services/overview)
- [HuggingFace Models for Healthcare](https://huggingface.co/models?pipeline_tag=text-generation&other=medical)
- [HIPAA Security Rule](https://www.hhs.gov/hipaa/for-professionals/security/index.html)

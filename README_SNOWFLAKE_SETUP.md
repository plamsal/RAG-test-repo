# Snowflake RAG Agent Setup Guide

This guide will help you connect to Snowflake and build a RAG (Retrieval-Augmented Generation) agent.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### 2. Configure Snowflake Credentials

**Option A: Username/Password Authentication (Recommended for Development)**

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and add your Snowflake credentials:
   ```env
   SNOWFLAKE_ACCOUNT=your-account-name
   SNOWFLAKE_USER=your-username
   SNOWFLAKE_PASSWORD=your-password
   SNOWFLAKE_WAREHOUSE=your-warehouse
   SNOWFLAKE_DATABASE=your-database
   SNOWFLAKE_SCHEMA=your-schema
   SNOWFLAKE_ROLE=your-role
   ```

**Finding Your Snowflake Account Name:**
- Your Snowflake URL looks like: `https://[account].snowflakecomputing.com`
- The `[account]` part is your account name (e.g., `xy12345.us-east-1`)

### 3. Test Your Connection

```bash
python snowflake_connector.py
```

You should see:
```
Testing Snowflake Connection...
✓ Successfully connected to Snowflake account: your-account-name
```

### 4. Run the RAG Agent Demo

```bash
python rag_agent.py
```

## 🔧 No More OAuth Issues!

**Why you were having OAuth callback problems:**
- The OAuth integration you were trying to set up (`CREATE SECURITY INTEGRATION`) is for **Snowflake's web UI** or **custom OAuth applications**
- For a RAG agent or any Python application, you **don't need OAuth**
- Simple username/password or key-pair authentication works better

**What we're using instead:**
- Direct authentication via `snowflake-connector-python`
- No redirect URIs needed
- No security integrations required
- Works immediately with your existing Snowflake credentials

## 📁 Project Structure

```
.
├── snowflake_connector.py   # Snowflake connection handler
├── rag_agent.py             # RAG agent implementation
├── requirements.txt         # Python dependencies
├── .env.example            # Template for credentials
├── .env                    # Your actual credentials (gitignored)
└── README_SNOWFLAKE_SETUP.md
```

## 🔐 Security Best Practices

### For Development:
- Use `.env` file (already in `.gitignore`)
- Never commit credentials to git

### For Production:
Use key-pair authentication instead of passwords:

1. Generate a key pair:
   ```bash
   openssl genrsa 2048 | openssl pkcs8 -topk8 -inform PEM -out snowflake_key.p8 -nocrypt
   openssl rsa -in snowflake_key.p8 -pubout -out snowflake_key.pub
   ```

2. Add the public key to Snowflake:
   ```sql
   ALTER USER your_username SET RSA_PUBLIC_KEY='MIIBIjANBg...';
   ```

3. Update your code to use key-pair auth (see `snowflake_connector.py` for examples)

## 🎯 Next Steps: Building Your RAG Agent

### Current Capabilities:
- ✅ Connect to Snowflake
- ✅ Execute SQL queries
- ✅ Retrieve table schemas
- ✅ Get sample data

### To Add Full RAG Functionality:

1. **Add LLM Integration** (OpenAI, Anthropic, etc.):
   ```python
   # Example with OpenAI
   from openai import OpenAI

   client = OpenAI(api_key="your-api-key")
   # Use to convert natural language to SQL
   ```

2. **Implement Text-to-SQL**:
   - Send table schemas to LLM
   - Ask LLM to generate SQL from natural language
   - Execute generated SQL
   - Format results

3. **Add Vector Embeddings** (for semantic search):
   - Create embeddings of table/column descriptions
   - Find relevant tables for user questions
   - Improve context for LLM

4. **Create a Conversational Interface**:
   - Chat history
   - Follow-up questions
   - Error handling

## 📊 Example Usage

```python
from snowflake_connector import SnowflakeRAGConnector
from rag_agent import SnowflakeRAGAgent

# Connect to Snowflake
with SnowflakeRAGConnector() as connector:
    # Create RAG agent
    agent = SnowflakeRAGAgent(connector)
    agent.initialize()

    # Query your data
    result = agent.execute_sql_with_explanation(
        "SELECT * FROM your_table LIMIT 10"
    )

    print(result['results'])
```

## 🐛 Troubleshooting

### Connection Issues:

**Error: "Incorrect username or password"**
- Double-check credentials in `.env`
- Ensure no extra spaces in values
- Try logging into Snowflake web UI with same credentials

**Error: "Account must be specified"**
- Make sure `SNOWFLAKE_ACCOUNT` is set correctly
- Format: `account-name.region` (e.g., `xy12345.us-east-1`)

**Error: "Database/Schema does not exist"**
- Verify the database and schema names
- Check that your user has access to them
- Try connecting without specifying database/schema first

### Still Having Issues?

1. Test connection manually:
   ```python
   import snowflake.connector

   conn = snowflake.connector.connect(
       user='YOUR_USER',
       password='YOUR_PASSWORD',
       account='YOUR_ACCOUNT'
   )
   print("Connected!")
   ```

2. Check Snowflake account status at: https://status.snowflake.com/

## 📚 Additional Resources

- [Snowflake Python Connector Docs](https://docs.snowflake.com/en/user-guide/python-connector)
- [Snowflake SQL Reference](https://docs.snowflake.com/en/sql-reference)
- [Building RAG Applications](https://www.anthropic.com/research/retrieval-augmented-generation)

---

**Ready to start?** Run `python snowflake_connector.py` to test your connection!

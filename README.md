# Generative AI Engineering Handbook

> A comprehensive, hands-on master class for building production-grade Generative AI systems, RAG pipelines, autonomous agents, and multi-modal architectures.

---

## Architecture Overview


```
                             +-----------------------+
                             |   API Gateway / CLI   |
                             +-----------+-----------+
                                         |
                                         v
                             +-----------------------+
                             |   LangGraph Router    |
                             +-----------+-----------+
                                         |
                +------------------------+------------------------+
                |                                                 |
                v                                                 v
   +-------------------------+                       +-------------------------+
   |   Retrieval Pipeline    |                       |    Function Calling     |
   |  (Hybrid + Re-Ranking)  |                       |    (Structured Tools)   |
   +------------+------------+                       +------------+------------+
                |                                                 |
                v                                                 v
   +-------------------------+                       +-------------------------+
   |  Qdrant Vector Database |                       |   External Services /   |
   |  (Dense + Sparse Index) |                       |   APIs / DB Execution   |
   +------------+------------+                       +------------+------------+
                |                                                 |
                +------------------------+------------------------+
                                         |
                                         v
                             +-----------------------+
                             |   Response Generation |
                             |   & Guardrail Check   |
                             +-----------------------+
```

---

## Master Index & Curriculum

### Theoretical Foundations (`chapters/`)

| File | Topic | Key Focus Areas |
| :--- | :--- | :--- |
| [`01-generative-ai-fundamentals.md`](chapters/01-generative-ai-fundamentals.md) | **GenAI Core** | Discriminative vs. Generative models, Autoregressive decoding, Softmax, Temperature ($T$), Top-$k$, and Top-$p$. |
| [`02-llms.md`](chapters/02-llms.md) | **LLM Architectures** | Transformers, Scaled Dot-Product Attention, Quantization (GGUF, AWQ), and Context Windows. |
| [`03-prompt-engineering.md`](chapters/03-prompt-engineering.md) | **Prompt Engineering** | Zero/Few-shot, Chain-of-Thought (CoT), Prompt Injection mitigation, and Pydantic schemas. |
| [`04-embeddings.md`](chapters/04-embeddings.md) | **Vector Spaces** | Dense representations, Cosine similarity, Euclidean distance, and Loss functions (InfoNCE). |
| [`05-vector-databases.md`](chapters/05-vector-databases.md) | **Vector Indexing** | HNSW graphs, IVF-Flat indexing, Sparse-Dense Hybrid Search, and Pre/Post filtering. |
| [`06-rag.md`](chapters/06-rag.md) | **RAG Pipelines** | Chunking strategies, Re-ranking (Cohere/Cross-Encoders), Parent-Document retrieval, and HyDE. |
| [`07-function-calling.md`](chapters/07-function-calling.md) | **Tool Calling** | OpenAPI schemas, Structured outputs, JSON argument validation, and Execution loops. |
| [`08-multimodal-ai.md`](chapters/08-multimodal-ai.md) | **Multimodal Systems** | Vision Transformers (ViT), Cross-attention projection layers, Image & Audio embedding alignment. |
| [`09-ai-evaluation.md`](chapters/09-ai-evaluation.md) | **Evals & Metrics** | RAG Triad (Context Relevance, Groundedness, Answer Relevance) and LLM-as-a-Judge benchmarking. |

---

### Hands-On Code Labs (`labs/`)

| Lab Directory | Core Functionality | Quick Link |
| :--- | :--- | :--- |
| `labs/llm-chatbot/` | Stateful, token-managed streaming terminal chatbot. | [`labs/llm-chatbot/app.py`](labs/llm-chatbot/app.py) |
| `labs/embeddings/` | Local vector extraction using `SentenceTransformers`. | [`labs/embeddings/app.py`](labs/embeddings/app.py) |
| `labs/semantic-search/` | In-memory similarity ranking and distance calculations. | [`labs/semantic-search/app.py`](labs/semantic-search/app.py) |
| `labs/rag/` | End-to-end Naive & Advanced RAG implementations. | [`labs/rag/app.py`](labs/rag/app.py) |
| `labs/pdf-qa/` | Document parsing, recursive chunking, and PDF Q&A. | [`labs/pdf-qa/app.py`](labs/pdf-qa/app.py) |
| `labs/multimodal-ai/` | Vision model prompts and image context processing. | [`labs/multimodal-ai/app.py`](labs/multimodal-ai/app.py) |

---

### Production Projects (`projects/`)

| Project | Description | Entry Point |
| :--- | :--- | :--- |
| **Research Assistant** | Autonomous multi-step topic synthesis API. | [`projects/research-assistant/main.py`](projects/research-assistant/main.py) |
| **Document Assistant** | FastAPI service for extracting and querying uploaded PDFs. | [`projects/document-assistant/main.py`](projects/document-assistant/main.py) |
| **Knowledge Base AI** | Full RAG microservice with Qdrant Vector Store integration. | [`projects/knowledge-base-ai/main.py`](projects/knowledge-base-ai/main.py) |

---

## Environment & Quickstart

### Prerequisites

* **Operating System:** Linux (Ubuntu 22.04 LTS / 24.04 LTS recommended) / macOS
* **Python Runtime:** `Python >= 3.10`
* **Package Manager:** `uv` or `pip`
* **Containers:** Docker Engine & Docker Compose v2

### Setup Guide

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-org/generative-ai-engineering.git](https://github.com/your-org/generative-ai-engineering.git)
   cd generative-ai-engineering
   ```

2. **Configure virtual environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

3. **Configure Environment Variables:**
    ```bash
    cp .env.example .env
    # Add your OPENAI_API_KEY and Qdrant settings inside .env
    ```

4. **Launch Local Infrastructure:**
    ```bash
    docker compose up -d
    ```

5. **Verify Lab Execution:**
    ```bash
    python labs/semantic-search/app.py
    ```

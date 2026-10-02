# Research RAG

A modular Retrieval-Augmented Generation (RAG) system designed to search and retrieve arXiv research papers using advanced query techniques and language models.

## Overview

Research RAG is a Python-based application that leverages Large Language Models (LLMs) and vector databases to intelligently search and retrieve relevant research papers from arXiv. The system uses sophisticated query translation and construction techniques to overcome limitations of traditional similarity-based search.

## Key Features

- **Multi-Query Generation**: Generates multiple perspectives of user queries to improve retrieval accuracy
- **Self-Query Retriever**: Uses LLMs to construct structured queries based on metadata filters
- **Advanced RAG Techniques**: Implements multiple RAG strategies including: Multi-Query RAG, RAG Fusion, Query Decomposition etc
- **Vector Database Integration**: Uses Chroma for efficient document storage and retrieval
- **Metadata Filtering**: Supports filtering by title, authors, publication year, and word count
- **Document Summarization**: Generates summaries of papers for improved retrieval
- **Cached Storage**: Efficiently caches documents and summaries to avoid reprocessing

## Tech Stack

- **Language Model**: OpenAI GPT-4o-mini
- **Vector Store**: Chroma
- **Framework**: LangChain
- **Paper Source**: arXiv API
- **PDF Processing**: PyMuPDF
- **Tracing**: LangSmith
- **Web Search**: Tavily Search API
- **Workflow Engine**: Langgraph - state graph for CRAG decision loop

## Project Structure

```
research-rag/
├── main_script.py              # Main execution script demonstrating RAG pipeline
├── QueryTranslation.py         # Query translation and RAG chain implementations
├── QueryConstruction.py        # Self-query retriever construction
├── indexing.py                 # Document indexing and multi-vector retriever setup
├── docDownload.py              # arXiv paper downloader
├── arxiv_papers/               # Directory storing downloaded PDF papers
├── chroma_db/                  # Vector database storage
├── data/                       # Additional data directory
├── ingestion/                  # Document ingestion pipeline
├── retrieval/                  # Retrieval logic
├── documents_cache.pkl         # Cached document metadata
├── summaries.pkl               # Cached document summaries
└── environmentVariables.env    # Environment configuration (not in repo)
```

## Installation

### Prerequisites

- Python 3.8+
- OpenAI API Key
- LangChain API Key (optional, for tracing)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/mpeshwe/research-rag.git
cd research-rag
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Create `environmentVariables.env` file with your API keys:
```env
OPENAI_API_KEY=your_openai_key_here
LANGCHAIN_API_KEY=your_langchain_key_here  # Optional
```

## Usage

### 1. Download Papers

First, download research papers from arXiv:

```bash
python docDownload.py
```

This script downloads papers from multiple categories:
- cs.LG (Machine Learning)
- stat.ML (Machine Learning Statistics)
- cs.AI (Artificial Intelligence)
- cs.CL (Computational Linguistics)
- cs.RO (Robotics)
- cs.CR (Cryptography)

### 2. Run the Main Script

Execute the main RAG pipeline:

```bash
python main_script.py
```

**Example Query:**
```python
question = "What are the most influential papers on reinforcement learning with words less than 50000?"
```

### 3. Component Descriptions

- **main_script.py** - Orchestrates the complete RAG pipeline
- **QueryConstruction.py** - Implements self-query retrieval
- **QueryTranslation.py** - Implements multiple RAG techniques
- **indexing.py** - Handles document processing
- **docDownload.py** - Downloads papers from arXiv API by category
- **CRAG.py** - Implements two core functions: grader and query rewriter 
- **graphCrag.py** - Implements a state-based workflow:

### 4. Document Format

Documents should include metadata:

```python
{
    "title": "Paper Title",
    "authors": ["Author 1", "Author 2"],
    "published_year": 2023,
    "word_count": 5000,
    "page_content": "Full paper content..."
}
```

### 5. Basic Example

```python
import main_script as ms
import graphCrag

# Define your research question
question = "What are the most influential papers on reinforcement learning with words less than 50000?"

# 1. Initialize the LLM
llm = ms.getllm()

# 2. Build the retrieval pipeline
multi_vector_retriever = ms.IDX.indexing()
vectorstore = multi_vector_retriever.vectorstore
self_query_retriever = ms.QC.construct_self_query_retriever(vectorstore, llm)

# 3. Generate multiple query variants
multi_query_chain = ms.QT.get_multi_query_chain_only_queries(llm)
queries = multi_query_chain.invoke({"question": question})

# 4. Retrieve documents
all_docs = []
for q in queries:
    docs = self_query_retriever.invoke(q)
    all_docs.extend(docs)

# 5. Deduplicate documents
unique_docs = {doc.metadata["title"]: doc for doc in all_docs}
curr_docs = list(unique_docs.values())
curr_docs = ms.IDX.deserialize_docs(curr_docs)

# 6. Generate answer with CRAG
graphCrag.generate_graph(question, curr_docs)
```

## Pipeline Workflow
```
                                        User Question
                                             │
                                             ▼
                    ┌─────────────────────────────────────────────────────────┐
                    │  STAGE 1 — Document Ingestion                           │
                    │  docDownload.py · indexing.py                           │
                    │                                                         │
                    │  → Download PDFs from arXiv API (cs.LG, stat.ML, ...)  │
                    │  → Extract text + metadata via PyMuPDFLoader            │
                    │  → Generate LLM summaries (GPT-4o-mini)                 │
                    │  → Index summaries into ChromaDB (MultiVectorRetriever) │
                    │  → Persist full docs in InMemoryByteStore (pickled)     │
                    │  → Cache docs + summaries to disk                       │
                    └─────────────────────────────────────────────────────────┘
                                             │
                                             ▼
                    ┌─────────────────────────────────────────────────────────┐
                    │  STAGE 2 — Query Translation                            │
                    │  QueryTranslation.py                                    │
                    │                                                         │
                    │  → Generate 5 semantically diverse query variants       │
                    │  → Multi-Query prompt | LLM | StrOutputParser           │
                    │  → Filter empty strings from LLM output                 │
                    │  → Optional: RAG Fusion, HyDE, Step-Back, Decomposition │
                    └─────────────────────────────────────────────────────────┘
                                             │
                                             ▼
                    ┌─────────────────────────────────────────────────────────┐
                    │  STAGE 3 — Query Construction & Retrieval               │
                    │  QueryConstruction.py · main_script.py                  │
                    │                                                         │
                    │  → SelfQueryRetriever constructs structured metadata    │
                    │    filters from natural language (title, authors,       │
                    │    published_year, word_count)                          │
                    │  → enable_limit=True: LLM decides doc count per query   │
                    │  → fix_invalid=True: fallback on invalid filter attrs   │
                    │  → Deduplicate results by title across all 5 queries    │
                    │  → Deserialize parent docs from byte store              │
                    └─────────────────────────────────────────────────────────┘
                                             │
                                             ▼
                    ┌─────────────────────────────────────────────────────────┐
                    │  STAGE 4 — Corrective RAG (CRAG) — LangGraph            │
                    │  CRAG.py · graphCrag.py                                 │
                    │                                                         │
                    │  → RETRIEVE: inject docs into graph state               │
                    │  → GRADE DOCUMENTS: binary LLM relevance scoring        │
                    │  → DECIDE (conditional edge):                           │
                    │       all relevant   ──▶  GENERATE                      │
                    │       any irrelevant ──▶  TRANSFORM QUERY               │
                    │                               └──▶  WEB SEARCH          │
                    │                                       └──▶  GENERATE    │
                    └─────────────────────────────────────────────────────────┘
                                             │
                                             ▼
                    ┌─────────────────────────────────────────────────────────┐
                    │  STAGE 5 — Answer Generation                            │
                    │  graphCrag.py                                           │
                    │                                                         │
                    │  → rlm/rag-prompt | GPT-4o-mini | StrOutputParser       │
                    │  → Final answer streamed node-by-node via app.stream()  │
                    └─────────────────────────────────────────────────────────┘
                                             │
                                             ▼
                                          LLM Answer
```
## CRAG Graph Decision Flow
```
              START
                └─▶  retrieve              ← injects pre-retrieved docs into graph state
                         └─▶  grade_documents   ← binary LLM relevance scoring per document
                                    └─▶  decide_to_generate  (conditional edge)
                                                 ├── [all relevant]   ──▶  generate ──▶ END
                                                 └── [any irrelevant] ──▶  transform_query
                                                                              └─▶  web_search_node
                                                                                         └─▶  generate ──▶ END
```


## Caching

The system implements intelligent caching to improve performance:
- **documents_cache.pkl**: Caches document content and metadata
- **summaries.pkl**: Caches generated paper summaries

This avoids re-downloading and re-processing papers on subsequent runs.

---

## Configuration & Customization

### 1. Document Limit
The pipeline currently loads **10 PDFs** by default to keep OpenAI API token costs low during development. To increase this, edit `indexing.py`:
```python
for filename in os.listdir(directory)[:10]:  # ← change this number
```

> ⚠️ Increasing the document count significantly raises the number of OpenAI API calls for embedding and summarisation.

---

### 2. Query Construction
`SelfQueryRetriever` is currently configured with `search_kwargs={"k": 2}` since the populated vector store is small (10 docs). If you scale up the document count, tune this accordingly:
```python
retriever = SelfQueryRetriever.from_llm(
    ...
    enable_limit=True,
    fix_invalid=True,
    search_kwargs={"k": 2}   # ← increase for larger corpora
)
```

Other parameters worth experimenting with:
- `search_type` — switch between `"similarity"`, `"mmr"` (max marginal relevance), or `"similarity_score_threshold"`
- `search_kwargs={"score_threshold": 0.7}` — filter out low-confidence results when using threshold mode
- `enable_limit=False` — fix the retrieval count to always return exactly `k` documents

---

### 3. Query Translation
The pipeline uses `get_multi_query_chain_only_queries()` by default, which generates 5 query variants. However, `QueryTranslation.py` includes several other ready-to-use techniques depending on your use case:

| Function | Technique | Best For |
|---|---|---|
| `get_multi_query_chain_only_queries()` | Multi-Query | General use — default |
| `get_multi_query_rag_chain()` | Multi-Query + RAG | End-to-end chain with retrieval |
| `get_rag_fusion_rag_chain()` | RAG Fusion (RRF) | Re-ranking across multiple queries |
| `get_decomposition_rag_chain()` | Decomposition | Complex multi-part questions |
| `get_step_back_rag_chain()` | Step-Back | Abstract or conceptual questions |
| `get_hyde_rag_chain()` | HyDE | Sparse or technical corpora |

## Author

- **mpeshwe** 
- **namanmawandia**

## Acknowledgements

 [**RAG From Scratch**](https://github.com/langchain-ai/rag-from-scratch/tree/main) by LangChain — provided foundational code patterns and an excellent reference for building modular RAG pipelines. Highly recommended for anyone learning RAG from the ground up.

**Last Updated**: March 2026
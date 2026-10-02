import os
import re
import uuid
import pickle

from dotenv import load_dotenv
import arxiv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.stores import InMemoryByteStore
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

load_dotenv("environmentVariables.env")
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_PROJECT'] = 'RAG_Modular_Script'
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY') 

CHROMA_PATH = "./chroma_db"
DOCSTORE_PATH = "./parent_docs_store"
CACHE_DOCS_FILE = "documents_cache.pkl"
SUMMARY_CACHE_FILE = "summaries.pkl"


def get_metadata_from_files(directory):
    client = arxiv.Client()
    # Regex to find standard arXiv ID patterns in filenames
    id_pattern = re.compile(r'(\d{4}\.\d{4,5})') 
    # metaData = []
    documents = []
    
    for filename in os.listdir(directory)[:10]:  # Limit to first 30 files for testing
        if filename.endswith(".pdf"):
            match = id_pattern.search(filename)
            if match:
                arxiv_id = match.group(1)
                
                # Search for this specific ID
                search = arxiv.Search(id_list=[arxiv_id])
                result = next(client.results(search))
                length=0
                loader = PyMuPDFLoader(os.path.join(directory, filename))
                try:
                    pages = loader.load()
                    full_text = " ".join([page.page_content for page in pages])
                    length = len(full_text)
                    # MAX_WORDS = 3000  # Limit to first 3000 words for metadata storage
                    # words = full_text.split()
                    # truncated_text = " ".join(words[:MAX_WORDS])
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
                    full_text = ""
                    # truncated_text = ""
                    length = 0
                documents.append(Document(page_content=full_text, 
                                metadata={"title": result.title, 
                                "authors": [a.name for a in result.authors], 
                                "published_year": result.published.year if result.published else None,
                                "word_count": length}))
                # metaData.append({
                #     "title": result.title,
                #     "authors": [a.name for a in result.authors],
                #     "published_year": result.published.year if result.published else None,
                #     "word_count" : length
                # })
            else:
                documents.append(Document(page_content="",
                                         metadata={"title": "Unknown",
                                         "authors": [], "published_year": None, "word_count": 0}))
                # metaData.append({
                #     "title": "Unknown",
                #     "authors": [],
                #     "published_year": None,
                #     "word_count": 0
                # })
    
    # Save to cache
    with open(CACHE_DOCS_FILE, 'wb') as f:
        pickle.dump(documents, f)
    
    print(f"Processed {len(documents)} papers for metadata and saved to {CACHE_DOCS_FILE}")
    # return metaData, documents
    return documents  



def get_summaries(docs, llm_model="gpt-4o-mini"):
    """Generates summaries for a list of documents using an LLM."""
    llm = ChatOpenAI(model=llm_model, temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        HumanMessagePromptTemplate.from_template(
            "Summarize the following arXiv research paper titled '{title}':\n\n" \
            "Paper content: \n {content}"
        )
    ])
    summaries = []
    chain = prompt | llm
    for doc in docs:
        response = chain.invoke({"title": doc.metadata['title'], "content": doc.page_content})    
        summaries.append(response.content)
    return summaries

def create_summary_based_retriever(docs, llm_model="gpt-4o-mini",skip_indexing=False):
    """Creates a summary-based MultiVectorRetriever."""

    # The vectorstore to use to index the child chunks (summaries)
    vectorstore = Chroma(collection_name="summaries_collection", 
                        embedding_function=OpenAIEmbeddings(),
                        persist_directory=CHROMA_PATH)

    store = InMemoryByteStore()
    id_key = "doc_id"

    # The retriever
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        byte_store=store,
        id_key=id_key,
    )

    if not skip_indexing:
        if os.path.exists(SUMMARY_CACHE_FILE):
            print("Loading cached summaries...")
            with open(SUMMARY_CACHE_FILE, "rb") as f:
                data = pickle.load(f)

            summaries = data["summaries"]
            doc_ids = data["doc_ids"]
            docs = data["docs"]
        else:
            print("Generating summaries...")
            summaries = get_summaries(docs, llm_model)
            doc_ids = [str(uuid.uuid4()) for _ in docs]

            with open(SUMMARY_CACHE_FILE, "wb") as f:
                pickle.dump({
                    "summaries": summaries,
                    "doc_ids": doc_ids,
                    "docs": docs
                }, f)

        print("Indexing into vectorstore...")

        summary_docs = [
            Document(page_content=s, metadata={id_key: doc_ids[i], **docs[i].metadata})
            for i, s in enumerate(summaries)
        ]

        retriever.vectorstore.add_documents(summary_docs)

        # IMPORTANT: serialize docs as required by langchain
        # MultiVectorRetriever no longer works with Document objects directly in the store.
        retriever.byte_store.mset([
            (doc_id, pickle.dumps(doc))
            for doc_id, doc in zip(doc_ids, docs)
        ])

        print("Indexing complete.")

    else:
        print("Retriever re-connected to persistent storage.")
        
    return retriever


def indexing():
    # 1. Check if we have already indexed everything
    if (os.path.exists(CHROMA_PATH) and os.path.exists(CACHE_DOCS_FILE) 
            and os.path.exists(SUMMARY_CACHE_FILE)):
        print("Loading existing index and documents...")
        with open(CACHE_DOCS_FILE, 'rb') as f:
            docs = pickle.load(f)
        
        # Re-initialize the retriever from disk
        return create_summary_based_retriever(docs, skip_indexing=True)

    # 2. If not, run the full pipeline
    print("No cache found. Starting full indexing pipeline...")
    docs = get_metadata_from_files("./arxiv_papers")
    return create_summary_based_retriever(docs, skip_indexing=False)

#you have to deserialize the retrieved docs when you are using them
def deserialize_docs(docs):
    final_docs = [
        pickle.loads(doc) if isinstance(doc, bytes) else doc
        for doc in docs
    ]
    return final_docs


indexing()



# def create_colbert_retriever(collection_data, index_name, model_name="colbert-ir/colbertv2.0"):
#     """Creates a ColBERT-based RAGatouille retriever."""
#     RAG = RAGPretrainedModel.from_pretrained(model_name)
#     RAG.index(
#         collection=collection_data,
#         index_name=index_name,
#         max_document_length=180,
#         split_documents=True,
#     )
#     print(f"ColBERT index '{index_name}' created successfully.")
#     retriever = RAG.as_langchain_retriever(k=3)
#     return retriever

# --- Example Usage --- 

# 1. Summary-Based Indexing
# web_urls = [
#     "https://lilianweng.github.io/posts/2023-06-23-agent/",
#     "https://lilianweng.github.io/posts/2024-02-05-human-data-quality/"
# ]
# web_docs = load_documents(web_urls)
# summary_retriever = create_summary_based_retriever(web_docs)
# query = "What are the components of an LLM-powered agent?"
# retrieved_summary_docs = summary_retriever.get_relevant_documents(query,n_results=1)
# print("\n--- Summary-Based Retrieval Results ---")
# print(retrieved_summary_docs[0].page_content[0:500])

# 2. COLBERT Indexing
# wikipedia_doc = get_wikipedia_page("Hayao_Miyazaki")
# colbert_retriever = create_colbert_retriever([wikipedia_doc], "Miyazaki-Colbert-Index")
# query = "What animation studio did Miyazaki found?"
# retrieved_colbert_docs = colbert_retriever.invoke(query)
# print("\n--- ColBERT Retrieval Results ---")
# print(retrieved_colbert_docs)
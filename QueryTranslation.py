import os
import bs4
from langsmith import Client
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.load import dumps, loads
from dotenv import load_dotenv
from operator import itemgetter
import numpy as np

# Set up environment variables (replace with your actual keys or use environment variables)
# For LangChain tracing
load_dotenv("environmentVariables.env")
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_PROJECT'] = 'RAG_Modular_Script'
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')  

def setup_rag_components(web_path: str, chunk_size: int = 300, chunk_overlap: int = 50):
    """
    Loads documents from a web path, splits them, creates embeddings, and initializes a vector store and retriever.

    Args:
        web_path: The URL of the document to load.
        chunk_size: The size of text chunks for splitting.
        chunk_overlap: The overlap between text chunks.

    Returns:
        A tuple containing the initialized retriever and ChatOpenAI LLM.
    """
    print(f"Loading documents from {web_path}...")
    loader = WebBaseLoader(
        web_paths=(web_path,),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    blog_docs = loader.load()

    print("Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    splits = text_splitter.split_documents(blog_docs)

    print("Creating vector store and retriever...")
    embedding_model = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_model)
    retriever = vectorstore.as_retriever()

    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    print("Setup complete.")
    return retriever, llm


def get_rag_prompt():
    """
    Pulls the default RAG prompt from LangChainHub.
    """
    client = Client()
    return client.pull_prompt("rlm/rag-prompt")

def get_multi_query_chain_only_queries(llm):
    """ Return the 5 different versions of the given user question to 
    retrieve relevant documents from a vector database. By generating multiple 
    perspectives on the user question, your goal is to help the user overcome some 
    of the limitations of the distance-based similarity search."""


    template = """You are an AI language model assistant. Your task is to generate five
    different versions of the given user question to retrieve relevant documents from a vector
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search.
    Provide these alternative questions separated by newlines. Original question: {question}"""
    prompt_perspectives = ChatPromptTemplate.from_template(template)

    generate_queries = (
        prompt_perspectives
        | llm
        | StrOutputParser()
        | (lambda x: [q.strip() for q in x.split("\n") if q.strip()])
    )
    return generate_queries


def get_multi_query_rag_chain(retriever, llm):
    """
    Returns a RAG chain implementing the Multi-Query technique.
    """
    template = """You are an AI language model assistant. Your task is to generate five
    different versions of the given user question to retrieve relevant documents from a vector
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search.
    Provide these alternative questions separated by newlines. Original question: {question}"""
    prompt_perspectives = ChatPromptTemplate.from_template(template)

    generate_queries = (
        prompt_perspectives
        | llm
        | StrOutputParser()
        | (lambda x: x.split("\n"))
    )

    def get_unique_union(documents: list[list]):
        """ Unique union of retrieved docs """
        flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
        unique_docs = list(set(flattened_docs))
        return [loads(doc) for doc in unique_docs]

    multi_query_retrieval_chain = generate_queries | retriever.map() | get_unique_union

    rag_prompt = get_rag_prompt()
    final_rag_chain = (
        {"context": multi_query_retrieval_chain, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    return final_rag_chain


def get_rag_fusion_rag_chain(retriever, llm):
    """
    Returns a RAG chain implementing the RAG Fusion technique with Reciprocal Rank Fusion.
    """
    template = """You are a helpful assistant that generates multiple search queries based on a single input query. \n\nGenerate multiple search queries related to: {question} \n\nOutput (4 queries):"""
    prompt_rag_fusion = ChatPromptTemplate.from_template(template)

    generate_queries = (
        prompt_rag_fusion
        | llm
        | StrOutputParser()
        | (lambda x: x.split("\n"))
    )

    def reciprocal_rank_fusion(results: list[list], k=60):
        """ Reciprocal_rank_fusion that takes multiple lists of ranked documents
            and an optional parameter k used in the RRF formula """
        fused_scores = {}
        for docs in results:
            for rank, doc in enumerate(docs):
                doc_str = dumps(doc)
                if doc_str not in fused_scores:
                    fused_scores[doc_str] = 0
                fused_scores[doc_str] += 1 / (rank + k)

        reranked_results = [
            (loads(doc), score)
            for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        ]
        return [doc for doc, _ in reranked_results] # Return only the documents

    retrieval_chain_rag_fusion = generate_queries | retriever.map() | reciprocal_rank_fusion

    rag_prompt = get_rag_prompt()
    final_rag_chain = (
        {"context": retrieval_chain_rag_fusion,
         "question": itemgetter("question")}
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    return final_rag_chain


def get_decomposition_rag_chain(retriever, llm):
    """
    Returns a RAG chain implementing the Decomposition technique.
    This chain first generates sub-questions and then iteratively answers them.
    """
    # Decomposition prompt for generating sub-questions
    template_decomposition_gen = """You are a helpful assistant that generates multiple sub-questions related to an input question. \n\nThe goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \n\nGenerate multiple search queries related to: {question} \n\nOutput (3 queries):"""
    prompt_decomposition_gen = ChatPromptTemplate.from_template(template_decomposition_gen)

    generate_queries_decomposition = (
        prompt_decomposition_gen
        | llm
        | StrOutputParser()
        | (lambda x: x.split("\n"))
    )

    # Prompt for answering decomposed questions using retrieved context and previous Q&A
    template_decomposition_answer = """Here is the question you need to answer:

    \n --- \n {question} \n --- \n
    Here is any available background question + answer pairs:

    \n --- \n {q_a_pairs} \n --- \n
    Here is additional context relevant to the question:

    \n --- \n {context} \n --- \n
    Use the above context and any background question + 
    answer pairs to answer the question: \n {question}"""
    decomposition_prompt_answer = ChatPromptTemplate.from_template(template_decomposition_answer)

    def format_qa_pair(question_str, answer_str):
        """Format Q and A pair"""
        return f"Question: {question_str}\nAnswer: {answer_str}\n\n".strip()

    def iterative_decomposition_qa(original_question: str, retriever_func, llm_model):
        sub_questions = generate_queries_decomposition.invoke({"question": original_question})
        q_a_pairs = ""
        final_answer = ""

        for q in sub_questions:
            # Retrieve context for the current sub-question
            context_docs = retriever_func.invoke(q)
            context_str = "\n\n".join(doc.page_content for doc in context_docs)

            # Construct the input for the decomposition_prompt_answer
            input_for_answer = {
                "question": q,
                "q_a_pairs": q_a_pairs,
                "context": context_str
            }
            
            # Invoke the LLM to get the answer for the sub-question
            answer = (decomposition_prompt_answer | llm_model | StrOutputParser()).invoke(input_for_answer)
            
            q_a_pairs += "\n---\n" + format_qa_pair(q, answer)
            final_answer = answer # Keep the last answer as the final output for simplicity, or combine them
            
        return final_answer # This can be refined to combine all answers meaningfully

    # This chain will take the original question and orchestrate the decomposition process
    return RunnableLambda(lambda x: iterative_decomposition_qa(x["question"], retriever, llm))


def get_step_back_rag_chain(retriever, llm):
    """
    Returns a RAG chain implementing the Step-Back technique.
    """
    examples = [
        {
            "input": "Could the members of The Police perform lawful arrests?",
            "output": "what can the members of The Police do?",
        },
        {
            "input": "Jan Sindel’s was born in what country?",
            "output": "what is Jan Sindel’s personal history?",
        },
    ]
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{output}"),
        ]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )
    step_back_query_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert at world knowledge. Your task is to step back and paraphrase a question to a more generic step-back question, which is easier to answer. Here are a few examples:""",
            ),
            few_shot_prompt,
            ("user", "{question}"),
        ]
    )
    generate_queries_step_back = step_back_query_prompt | llm | StrOutputParser()

    response_prompt_template = """You are an expert of world knowledge. 
    I am going to ask you a question. Your response should be comprehensive and 
    not contradicted with the following context if they are relevant.
    Otherwise, ignore them if they are not relevant.

    # {normal_context}
    # {step_back_context}

    # Original Question: {question}
    # Answer:"""
    response_prompt = ChatPromptTemplate.from_template(response_prompt_template)

    chain = (
        {
            "normal_context": RunnableLambda(lambda x: x["question"]) | retriever,
            "step_back_context": generate_queries_step_back | retriever,
            "question": lambda x: x["question"],
        }
        | response_prompt
        | llm
        | StrOutputParser()
    )
    return chain


def get_hyde_rag_chain(retriever, llm):
    """
    Returns a RAG chain implementing the HyDE technique.
    """
    template_hyde_gen = """Please write a scientific paper passage to answer the question
    Question: {question}
    Passage:"""
    prompt_hyde_gen = ChatPromptTemplate.from_template(template_hyde_gen)

    generate_docs_for_retrieval = (
        prompt_hyde_gen | llm | StrOutputParser()
    )

    # This chain first generates a hypothetical document, then retrieves based on it
    hyde_retrieval_chain = generate_docs_for_retrieval | retriever

    rag_prompt = get_rag_prompt()
    final_rag_chain = (
        {"context": hyde_retrieval_chain,
         "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    return final_rag_chain


# # --- Example Usage ---
# if __name__ == "__main__":
#     # 1. Setup RAG components
#     blog_url = "https://lilianweng.github.io/posts/2023-06-23-agent/"
#     base_retriever, base_llm = setup_rag_components(blog_url)

#     question = "What is Task Decomposition for LLM agents?"

#     print("\n--- Running Multi-Query RAG ---")
#     multi_query_chain = get_multi_query_rag_chain(base_retriever, base_llm)
#     response_multi_query = multi_query_chain.invoke(question)
#     print(f"Multi-Query Response: {response_multi_query}")

#     print("\n--- Running RAG Fusion ---")
#     rag_fusion_chain = get_rag_fusion_rag_chain(base_retriever, base_llm)
#     response_rag_fusion = rag_fusion_chain.invoke({"question": question})
#     print(f"RAG Fusion Response: {response_rag_fusion}")

#     print("\n--- Running Decomposition RAG ---")
#     # The decomposition chain is designed to be invoked with the original question.
#     # Its internal logic handles sub-question generation and iterative answering.
#     decomposition_chain = get_decomposition_rag_chain(base_retriever, base_llm)
#     response_decomposition = decomposition_chain.invoke({"question": question})
#     print(f"Decomposition Response: {response_decomposition}")

#     print("\n--- Running Step-Back RAG ---")
#     step_back_chain = get_step_back_rag_chain(base_retriever, base_llm)
#     response_step_back = step_back_chain.invoke({"question": question})
#     print(f"Step-Back Response: {response_step_back}")

#     print("\n--- Running HyDE RAG ---")
#     hyde_chain = get_hyde_rag_chain(base_retriever, base_llm)
#     response_hyde = hyde_chain.invoke({"question": question})
#     print(f"HyDE Response: {response_hyde}")

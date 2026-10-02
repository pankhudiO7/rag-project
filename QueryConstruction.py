from langchain_core.language_models import BaseLanguageModel
from langchain_core.vectorstores import VectorStore
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

# class TutorialSearch(BaseModel):
#     """Search over a database of tutorial videos about a software library."""

#     content_search: str = Field(
#         ...,
#         description="Similarity search query applied to video transcripts.",
#     )
#     title_search: str = Field(
#         ...,
#         description=(
#             "Alternate version of the content search query to apply to video titles. "
#             "Should be succinct and only include key words that could be in a video "
#             "title."
#         ),
#     )
#     min_view_count: Optional[int] = Field(
#         None,
#         description="Minimum view count filter, inclusive. Only use if explicitly specified.",
#     )
#     max_view_count: Optional[int] = Field(
#         None,
#         description="Maximum view count filter, exclusive. Only use if explicitly specified.",
#     )
#     earliest_publish_date: Optional[datetime.date] = Field(
#         None,
#         description="Earliest publish date filter, inclusive. Only use if explicitly specified.",
#     )
#     latest_publish_date: Optional[datetime.date] = Field(
#         None,
#         description="Latest publish date filter, exclusive. Only use if explicitly specified.",
#     )
#     min_length_sec: Optional[int] = Field(
#         None,
#         description="Minimum video length in seconds, inclusive. Only use if explicitly specified.",
#     )
#     max_length_sec: Optional[int] = Field(
#         None,
#         description="Maximum video length in seconds, exclusive. Only use if explicitly specified.",
#     )

#     #printing only the fields that are not None and not equal to their default value (if they have one)
#     def pretty_print(self) -> None:
#         for field in type(self).model_fields:
#             if getattr(self, field) is not None and getattr(self, field) != getattr(
#                 type(self).model_fields[field], "default", None
#             ):
#                 print(f"{field}: {getattr(self, field)}")

def construct_self_query_retriever(vectorstore: VectorStore, llm: BaseLanguageModel) -> SelfQueryRetriever:
    """
    Constructs a SelfQueryRetriever for arXiv papers.
    
    Args:
        vectorstore: The vector store containing the documents with metadata.
        llm: The language model to use for query construction.
    
    Returns:
        A SelfQueryRetriever instance.
    """
    # Define metadata field info
    metadata_field_info = [
        AttributeInfo(
            name="title",
            description="The title of the paper",
            type="string",
        ),
        AttributeInfo(
            name="authors",
            description="The authors of the paper",
            type="string",
        ),
        AttributeInfo(
            name="published_year",
            description="The year the paper was published",
            type="integer",
        ),
        AttributeInfo(
            name="word_count",
            description="The number of words in the paper content",
            type="integer",
        ),
    ]
    
    # Create the document content description
    document_content_description = "Content of arXiv papers including sections like abstract, introduction, methods, results, etc."
    
    # Initialize SelfQueryRetriever
    retriever = SelfQueryRetriever.from_llm(
        llm=llm,
        vectorstore=vectorstore,
        document_contents=document_content_description,
        metadata_field_info=metadata_field_info,
        verbose=True,
        enable_limit=True,        # ← LLM decides how many docs to fetch
        fix_invalid=True,        # ← if LLM generates an invalid query, it will try to fix it and re-run
        search_kwargs={"k": 2}    # ← acts as the max/default cap 
                                # kept small as less number of docs in sample space
    )
    
    return retriever

# retriever.invoke("papers about reinforcement learning after 2021")
# gives you the list of docs

# No need of this when using SelfQueryRetriever, since it constructs the query internally 
# based on the metadata fields and document content description

# def construct_query(llm):
#     system = """You are an expert at converting user questions into database queries. \
#     You have access to a database of arXiv papers. \
#     Given a question, return a database query optimized to retrieve the most relevant results.

#     If there are acronyms or words you are not familiar with, do not try to rephrase them."""
#     prompt = ChatPromptTemplate.from_messages(
#         [
#             ("system", system),
#             ("human", "{question}"),
#         ]
#     )
#     structured_llm = llm.with_structured_output(PaperSearch)
#     query_analyzer = prompt | structured_llm
#     return query_analyzer

# if __name__ == "__main__":
#     llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
#     query_analyzer = construct_query(llm)

#     question = "What are some good papers about LLM agents published after 2022 with more than 5000 words?"
#     query = query_analyzer(question=question)
#     query.pretty_print()



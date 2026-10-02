def retrieve_candidates(
    query_embedding,
    vector_store,
    k=30
):
    """
    Purposefully retrieve MORE than needed.
    Precision comes later.
    """
    return vector_store.similarity_search(
        query_embedding,
        top_k=k
    )

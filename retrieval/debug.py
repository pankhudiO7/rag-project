def log_retrieval(chunks):
    for i, c in enumerate(chunks):
        print(
            f"[{i}] paper={c.paper_id} "
            f"section={c.section} "
            f"chars={len(c.text)}"
        )
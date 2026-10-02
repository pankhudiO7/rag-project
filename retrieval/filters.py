def apply_filters(chunks, rq):
    filtered = []

    for c in chunks:
        if rq.paper_id and c.paper_id != rq.paper_id:
            continue
        if rq.section and c.section != rq.section:
            continue
        if rq.year_min and c.year < rq.year_min:
            continue
        if rq.year_max and c.year > rq.year_max:
            continue

        filtered.append(c)

    return filtered

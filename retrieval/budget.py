def budget_context(chunks, max_chars = 3500) :
    selected = []
    total = 0 

    for c in chunks :
        if total + len(c.text) > max_chars :
            break
        selected.append(c)
        total += len(c.text)
    return selected
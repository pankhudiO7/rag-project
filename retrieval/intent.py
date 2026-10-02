def infer_section_intent(query: str) -> str | None:
    q = query.lower()

    if any(w in q for w in ["how", "architecture", "model", "approach"]):
        return "method"
    if any(w in q for w in ["evaluate", "experiment", "benchmark", "metric"]):
        return "experiments"
    if any(w in q for w in ["result", "performance", "improve"]):
        return "results"
    if any(w in q for w in ["limitation", "future work", "drawback"]):
        return "conclusion"

    return None

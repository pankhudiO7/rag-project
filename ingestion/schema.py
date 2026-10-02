from dataclasses import dataclass
# This is a schema. 
#   - Enables filtering 
#   - Enables Evaluation
#   - Enables GraphRAG (later)
@dataclass
class PaperSection:
    paper_id: str 
    title: str 
    year: int
    authors: list[str]
    section: str
    text: str

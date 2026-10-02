from dataclasses import dataclass
from typing import Optional

@dataclass
class RetrievalQuery:
    text: str 
    paper_id: Optional[str] = None
    section: Optional[str] = None
    year_min: Optional[int] = None
    year_max: Optional[int] = None
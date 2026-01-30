from dataclasses import dataclass
from typing import List, Optional


@dataclass
class RagPath:

    query: str
    past_subqueries: Optional[List[str]]
    past_subanswers: Optional[List[str]]
    past_doc_ids: Optional[List[List[str]]]
    past_documents: Optional[List[List[str]]] = None # 실제 사용된 텍스트 저장용
    past_poisoned_flags: Optional[List[List[bool]]] = None # 오염된 문서 여부 저장용
    past_retriever_results: Optional[List[List]] = None # retriever_results 저장용
    past_adv_corpuses: Optional[List[List[str]]] = None # 생성된 독성 문서 저장용 (Dynamic/Oracle)
    past_adv_subanswers: Optional[List[str]] = None # 생성된 독성 답변 저장용 (Dynamic/Oracle)

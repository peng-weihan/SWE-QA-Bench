"""Problem statement processing, based on SWE-agent's problem_statement"""
import hashlib
import logging
from pathlib import Path
from typing import Protocol
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ProblemStatement(Protocol):
    """Problem statement protocol"""
    id: str
    
    def get_problem_statement(self) -> str: ...


class TextProblemStatement(BaseModel):
    """Text problem statement"""
    text: str
    id: str = None  # type: ignore
    
    def model_post_init(self, __context) -> None:
        if self.id is None:
            self.id = hashlib.sha256(self.text.encode()).hexdigest()[:8]
    
    def get_problem_statement(self) -> str:
        return self.text


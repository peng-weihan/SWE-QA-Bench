from pydantic import BaseModel, Field
from typing import Dict, List, Optional

class FileNode(BaseModel):
    """Data model for file node"""
    file_name: str
    upper_path: str
    module: str
    define_class: List[str]
    imports: List[str]

class CodeNode(BaseModel):
    """Data model for code node"""
    start_line: int
    end_line: int
    belongs_to: FileNode
    relative_function: List[str]
    code: str

class QAPair(BaseModel):
    """Data model for question-answer pair"""
    question: str
    answer: Optional[str] = Field(default=None,description="answer of the question")
    relative_code_list: Optional[List[Dict]] = Field(default=None,description="code list of the question")
    ground_truth: Optional[str] = Field(default=None,description="ground truth of the question")
    score: Optional[float] = None

class ResultPair(BaseModel):
    """Data model for result pair"""
    answer: str
    ground_truth: Optional[str] = Field(default=None,description="ground truth of the question")
    thought: str

from typing import List, Dict, Optional, Tuple
from pydantic import BaseModel, Field, validator
from enum import IntEnum
import json

class EvaluationScore(IntEnum):
    """Evaluation score enum"""
    INCOMPLETE = 1  # Answer is incomplete, vague, or off-topic
    BASIC = 2       # Answer addresses the question but lacks accuracy or detail
    GOOD = 3        # Answer is complete and helpful but could be improved
    VERY_GOOD = 4   # Answer is very good, accurate, and comprehensive
    PERFECT = 5     # Answer is perfect, accurate, comprehensive, and easy to understand

class GPTEvaluationResponse(BaseModel):
    """Model for parsing GPT's evaluation response"""
    score: EvaluationScore
    reasoning: str

    @validator('score')
    def validate_score(cls, v):
        if not 1 <= v <= 5:
            raise ValueError('Score must be between 1 and 5')
        return v

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

class ResultPair(BaseModel):
    """Data model for result pair"""
    answer: str
    ground_truth: Optional[str] = Field(default=None,description="ground truth of the question")
    thought: str

class QAPair(BaseModel):
    """Data model for question-answer pair"""
    question: str
    answer: Optional[str] = Field(default=None,description="answer of the question")
    relative_code_list: Optional[List[CodeNode]] = Field(default=None,description="code list of the question")
    ground_truth: Optional[str] = Field(default=None,description="ground truth of the question")
    score: Optional[float] = None

class QAGeneratorResponse(BaseModel):
    """Data model for question-answer pair generator response"""
    question: str
    ground_truth: str

class QAGeneratorResponseList(BaseModel):
    """Data model for question-answer pair generator response list"""
    qa_pairs: List[QAGeneratorResponse]

class QAPairListResponse(BaseModel):
    """Data model for question-answer pair list response"""
    qa_pairs: List[QAPair]
    
class EvaluationResult(BaseModel):
    """Data model for evaluation result"""
    qa_pair: QAPair
    score: float
    reasoning: str
    suggestions: Optional[List[str]] = Field(default_factory=list)

    @validator('score')
    def validate_score(cls, v):
        if not 1 <= v <= 5:
            raise ValueError('Score must be between 1 and 5')
        return v

class ClassAttribute(BaseModel):
    """Data model for class attribute/field information"""
    name: str
    class_name: str = Field(...,description="Name of the class it belongs to")
    related_functions: List[str] = Field(default_factory=list,description="Method names involved in modifying the attribute value")
    type_hint: Optional[str] = None

class VariableDefinition(BaseModel):
    """Data model for variable definition information"""
    name: str
    docstring: Optional[str] = None
    relative_code: Optional[CodeNode] = Field(default=None, description="Code segment related to variable definition")
    scope: str = Field("global", description="Variable scope: global/local/class")
    function_name: Optional[str] = Field(default=None, description="Function name if it's a local variable")
    class_name: Optional[str] = Field(default=None, description="Class name if it's a class variable")
    type_hint: Optional[str] = Field(default=None, description="Variable type hint")
    value: Optional[str] = Field(default=None, description="String representation of variable value")
    is_constant: bool = Field(default=False, description="Whether it's a constant (all uppercase naming)")
    references: List[str] = Field(default_factory=list, description="Functions/methods that reference this variable")

class FunctionDefinition(BaseModel):
    """Data model for function/method definition information"""
    name: str
    docstring: Optional[str] = None
    relative_code: Optional[CodeNode] = Field(..., description="Code segment related to function definition")
    is_method: bool = False
    class_name: Optional[str] = None
    parameters: List[str] = Field(default_factory=list)
    calls: List[str] = Field(default_factory=list)

class ClassDefinition(BaseModel):
    """Data model for class definition information"""
    name: str
    docstring: Optional[str] = None
    relative_code: Optional[CodeNode] = Field(..., description="Code segment related to class definition")
    methods: List[FunctionDefinition] = Field(default_factory=list,description="Methods contained in the class")
    attributes: List[ClassAttribute] = Field(default_factory=list)

class ModuleNode(BaseModel):
    """Module node model"""
    name: str
    path: str
    files: List[FileNode] = Field(default_factory=list)
    sub_modules: List["ModuleNode"] = Field(default_factory=list)
    is_package: bool = False
    
    class Config:
        arbitrary_types_allowed = True

class CodeRelationship(BaseModel):
    """Code relationship model"""
    source_type: str = Field(..., description="Source type: class/function/attribute")
    source_id: str = Field(..., description="Source identifier")
    target_type: str = Field(..., description="Target type: class/function/attribute")
    target_id: str = Field(..., description="Target identifier")
    relationship_type: str = Field(..., description="Relationship type: inherits/calls/uses/implements")

class RepositoryStructure(BaseModel):
    """Repository structure model"""
    root_modules: List[ModuleNode] = Field(default_factory=list)
    classes: List[ClassDefinition] = Field(default_factory=list)
    functions: List[FunctionDefinition] = Field(default_factory=list)
    attributes: List[ClassAttribute] = Field(default_factory=list)
    core_functionality: Optional[str] = None
    variables: List[VariableDefinition] = Field(default_factory=list)
    
    # Add dependency graph
    dependency_graph: Dict[str, List[str]] = Field(
        default_factory=dict, 
        description="Module dependency graph: {module_path: [dependent_module_paths...]}"
    )
    relationships: List[CodeRelationship] = Field(default_factory=list, description="Relationships between code elements")

class Repository(BaseModel):
    """Main model for code repository"""
    id: str = Field(..., description="Unique identifier for the repository")
    name: str = Field(..., description="Repository name")
    url: Optional[str] = Field(None, description="Repository URL")
    description: Optional[str] = Field(None, description="Repository description")
    structure: RepositoryStructure = Field(default_factory=RepositoryStructure, description="Repository structure")
    qa_pairs: List[QAPair] = Field(default_factory=list, description="Question-answer pairs associated with the repository")
    
    class Config:
        schema_extra = {
            "example": {
                "id": "repo-123",
                "name": "example-repo",
                "url": "https://github.com/user/example-repo"
            }
        } 

def load_repository_from_json(file_path: str) -> Repository:
    """Load and reconstruct Repository instance from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Use Pydantic's model_validate method (v2) or parse_obj method (v1)
    try:
        # Pydantic v2 approach (recommended)
        return Repository.model_validate(data)
    except AttributeError:
        print("Unable to use model_validate, falling back to parse_obj")
        return Repository.parse_obj(data)

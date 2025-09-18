from typing import Dict
from .data_models import CodeNode, QAPair

CODE_FORMATITING_PROMPT = """
## Code Information
- File Name: {file_name}
- Belongs to the moddule: {module}
- The code is part of the definition of the class: {define_class}
- Imports: {imports}
- Code Snippet: 
```
{code}
```
"""

CODE_INSTRUCTION = """
These are the code snippets that are potentially related to the question, please answer your question based on the following retrieved code information.
"""



ground_truth_field_prompt = (
    "\\nInclude supporting evidence only if strictly necessary (e.g., short relevant code snippets)."
    "\\nPlace all evidence, including code snippets or repository-specific knowledge, into this field."
    "\\nKeep this field factual and complete."
    "\\nDO NOT introduce any external information, guesses, or assumptions."
    "\\nIf no supporting information can be found within the repository for the query, just state in this field that not enough information from the code snippets."
)

answer_field_prompt = (
    "\\nProvide the final answer concisely and directly, without code snippets, extra explanations or commentary."
    "\\nIf no supporting code snippets or repository-specific knowledge can be found within the repository for the question, directly answer with your own knowledge."
)

ANSWER_FORMAT_INSTRUCTION = f"""

Return your answer strictly in valid JSON format. The JSON must use double quotes (\") for keys and string values. "
Escape all newlines inside strings as \\n. Do NOT include Python-style single quotes or unescaped line breaks.

The JSON format should be exactly:

{{
    "thought": "The thought process of the answer.",
    "ground_truth": "{ground_truth_field_prompt}",
    "answer": "{answer_field_prompt}"
}}
"""

def format_code_from_list(relative_code_list: list[CodeNode]):
    code_prompt = CODE_INSTRUCTION
    for code in relative_code_list:
        code_prompt += format_code_from_code_node(code)
    
    code_prompt += ANSWER_FORMAT_INSTRUCTION
    return code_prompt

def format_code_from_list(relative_code_list: list[Dict]):
    code_prompt = CODE_INSTRUCTION
    for num, code in enumerate(relative_code_list):
        code_prompt += f"\nCode {num+1}:\n"
        code_prompt += str(code)
    code_prompt += ANSWER_FORMAT_INSTRUCTION
    return code_prompt

def format_code_from_code_node(code_node: CodeNode):
    code_prompt = CODE_FORMATITING_PROMPT.format(
        file_name=code_node.belongs_to.file_name,
        module=code_node.belongs_to.module,
        define_class=code_node.belongs_to.define_class,
        imports=code_node.belongs_to.imports,
        code=code_node.code
     )
    return code_prompt

def format_context(qa_pair: QAPair):
    context_prompt = """
    ## Related Context
    Supporting Ground Truth:
    {ground_truth}
    {code_snippets}
    """

    prompt = context_prompt.format(
        ground_truth=qa_pair.ground_truth,
        code_snippets=("\n".join([format_code_from_code_node(code) for code in qa_pair.relative_code_list]))
    )
    return prompt


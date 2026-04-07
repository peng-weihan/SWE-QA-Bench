import os
import random
from typing import List
import concurrent
import openai
from dotenv import load_dotenv
from typing import List
from models.data_models import QAGeneratorResponseList, QAPair, RepositoryStructure
from qa_generator.core.generator import BaseGenerator
import json
from tqdm import tqdm
load_dotenv()
SYSTEM_PROMPT = """You are a professional code analysis assistant, you are good at generating high quality questions about code repository.
Generate as many questions as possible."""

PARALLEL_WORKERS = 1

class AgentQAGeneratorV2(BaseGenerator):
    def __init__(self, questions_dir: str = None, max_workers: int = PARALLEL_WORKERS):
        super().__init__()
        if questions_dir is None:
            # Use default template questions directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.questions_dir = os.path.join(os.path.dirname(os.path.dirname(current_dir)), 'dataset', 'seed_v3')
        else:
            self.questions_dir = questions_dir

        # Parallel processing configuration
        self.max_workers = max_workers
        self.max_workers_class = max_workers   # Class generation uses 2x parallelism
        self.max_workers_function = max_workers # Function generation uses half parallelism

        # Define 12 specific categories based on actual filenames in seed_v3
        self.categories = {
            "what": ["architecture", "concept-defi", "rela-depend"],
            "how": ["algo-impl", "api-framework", "system-design"],
            "why": ["design-rationale", "performance", "purpose"],
            "where": ["data-control-flow", "funct-loca", "iden-loca"]
        }
        
        # Load template questions for all categories
        self.template_questions = self.load_all_category_questions()
        for category_type, category_list in self.categories.items():
            for category in category_list:
                if category in self.template_questions:
                    print(f"Loaded {len(self.template_questions[category])} questions for category '{category}'")

    def load_all_category_questions(self) -> dict:
        """Load template questions for all 12 categories"""
        all_questions = {}
        
        for category_type, category_list in self.categories.items():
            category_path = os.path.join(self.questions_dir, category_type)
            if not os.path.isdir(category_path):
                continue
                
            for category in category_list:
                file_path = os.path.join(category_path, f"{category}.txt")
                if os.path.exists(file_path):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        questions = [line.strip() for line in f if line.strip() and not line.strip().startswith('//')]
                    all_questions[category] = questions
                    print(f"Loaded {len(questions)} questions from {file_path}")
                else:
                    print(f"Warning: File {file_path} not found")
                    all_questions[category] = []
        
        return all_questions

    def generate_summary_of_repo(self, repository_structure: RepositoryStructure) -> str:
         # Extract simplified representation of dependency graph
        dep_graph_summary = {}
        for module, dependencies in repository_structure.dependency_graph.items():
            module_name = os.path.basename(module)
            dep_graph_summary[module_name] = [os.path.basename(dep) for dep in dependencies]
        
        
        # Create simplified representation of classes and functions
        classes_summary = []
        for cls in repository_structure.classes:
            classes_summary.append({
                "name": cls.name,
                "methods": [method.name for method in cls.methods],
                "attributes": [attr.name for attr in cls.attributes]
            })
        
        functions_summary = []
        for func in repository_structure.functions:
            if not func.is_method:  # Only include top-level functions
                functions_summary.append({
                    "name": func.name,
                    "parameters": func.parameters,
                    "calls": func.calls[:5] if len(func.calls) > 5 else func.calls  # Limit call list size
                })
        
        # Build simplified representation of module tree
        def module_to_dict(module):
            return {
                "name": module.name,
                "is_package": module.is_package,
                "files": [file.file_name for file in module.files],
                "sub_modules": [module_to_dict(submodule) for submodule in module.sub_modules]
            }
        
        module_tree_summary = [module_to_dict(module) for module in repository_structure.root_modules]
        
        # Build input prompt
        summary = {
            "dependency_graph": dep_graph_summary,
            "classes": classes_summary,  # Limit number of classes
            "functions": functions_summary,  # Limit number of functions
            "module_tree": module_tree_summary
        }
        
        summary_str = json.dumps(summary, ensure_ascii=False, indent=2)
        return summary_str
    
    def _generate_qa_pairs_with_llm(self, prompt_content: str) -> List[QAPair]:
        """
        Helper method to generate QA pairs using LLM
        
        Args:
            prompt_content: The prompt content to send to the LLM
            
        Returns:
            List of QAPair objects
        """
        schema = QAGeneratorResponseList.model_json_schema()
        schema_str = json.dumps(schema, ensure_ascii=False, indent=2)
        
        prompt = f"{prompt_content}\n\nPlease return the response in the following JSON format:\n{schema_str}"
        
        print(f"Prompt for LLM:\n{prompt}\n")
        result_text = self._call_llm(system_prompt=SYSTEM_PROMPT, user_prompt=prompt)
        try:
            result_json = json.loads(result_text)
            result_model = QAGeneratorResponseList.model_validate(result_json)
            return [
                QAPair(
                    question=item.question,
                    ground_truth=item.ground_truth,
                ) for item in result_model.qa_pairs
            ]
        except Exception as e:
            try:
                result_json = json.loads(result_text)
                if "qa_pairs" in result_json:
                    return [
                        QAPair(
                            question=item.get("question", ""),
                            answer=item.get("answer", ""),
                            related_code=item.get("related_code")
                        ) for item in result_json["qa_pairs"]
                    ]
            except:
                pass
            
            return []

    def generate_questions(self, repo_structure: RepositoryStructure, file: str) -> List[QAPair]:
        """Generate questions based on 12 categories"""
        print("Starting question generation by categories...")
        all_questions = []
        
        # Organize questions by category hierarchy
        categorized_questions = {}
        
        # Globally track used classes and functions to ensure no duplication
        used_classes = set()
        used_functions = set()
        
        # Generate questions for each of the 12 categories
        for category_type, category_list in self.categories.items():
            categorized_questions[category_type] = {}
            
            for category in category_list:
                print(f"Generating questions for category: {category}")
                category_questions, used_classes, used_functions = self.generate_questions_by_category(
                    repo_structure, file, category, used_classes, used_functions
                )
                categorized_questions[category_type][category] = category_questions
                all_questions.extend(category_questions)
                print(f"Generated {len(category_questions)} questions for category {category}")
                print(f"Used classes so far: {len(used_classes)}, Used functions so far: {len(used_functions)}")
        
        # Write questions to file by category hierarchy
        self.write_categorized_questions_to_file(categorized_questions, file)
        
        return all_questions

    def generate_questions_by_category(self, repo_structure: RepositoryStructure, file: str, category: str, used_classes: set, used_functions: set) -> tuple:
        """Generate questions for a specific category"""
        questions = []
        
        # Get seed questions for this category
        seed_questions = self.template_questions.get(category, [])
        if not seed_questions:
            print(f"No seed questions found for category: {category}")
            return questions, used_classes, used_functions
        
        # Randomly sample some seed questions
        num_seed_questions = min(5, len(seed_questions))
        selected_seed_questions = random.sample(seed_questions, num_seed_questions)
        
        # Get unused classes and functions
        available_classes = [cls for cls in repo_structure.classes if cls.name not in used_classes]
        available_functions = [func for func in repo_structure.functions if func.name not in used_functions]
        
        # Randomly sample some unused classes and functions
        num_classes = min(10, len(available_classes))
        num_functions = min(10, len(available_functions))
        
        if num_classes > 0:
            sampled_classes = random.sample(available_classes, num_classes)
            # Mark these classes as used
            for cls in sampled_classes:
                used_classes.add(cls.name)
        else:
            sampled_classes = []
            print(f"No available classes for category {category}")
        
        if num_functions > 0:
            sampled_functions = random.sample(available_functions, num_functions)
            # Mark these functions as used
            for func in sampled_functions:
                used_functions.add(func.name)
        else:
            sampled_functions = []
            print(f"No available functions for category {category}")
        
        print(f"Category {category}: Using {len(sampled_classes)} classes and {len(sampled_functions)} functions")
        
        # Generate questions for classes
        if sampled_classes:
            class_questions = self.generate_questions_for_classes_by_category(
                sampled_classes, selected_seed_questions, category, file
            )
            questions.extend(class_questions)
        
        # Generate questions for functions
        if sampled_functions:
            function_questions = self.generate_questions_for_functions_by_category(
                sampled_functions, selected_seed_questions, category, file, repo_structure
            )
            questions.extend(function_questions)
        
        return questions, used_classes, used_functions

    def generate_questions_for_classes_by_category(self, classes: List, seed_questions: List[str], category: str, file: str) -> List[QAPair]:
        """Generate questions for classes in a specific category"""
        prompt_template = f"""
You are an expert software research assistant.

Given:
1. A class description extracted from a software repository.
2. A list of seed questions from the "{category}" category.

Task:
1. Based on the seed questions and the class description, generate **one single question** that is:
   - As difficult and complex as possible,
   - Requires multi-hop reasoning or deep technical understanding,
   - Not answerable by simple retrieval or direct lookup (i.e., not solvable by basic RAG methods),
   - Clearly related to the class/module description,
   - Technically precise and detailed,
   - Reflects the style and intent of the original seed questions but goes significantly deeper.
   - **Must not be a compound question** (e.g., no use of "and", "or", or comma-based subquestions),
   - **Must be not too long and syntactically simple**
   - **Must be specific to the "{category}" category**

2. The question should encourage advanced analysis, integration of multiple concepts, or insight beyond surface-level information.

3. Output only the single refined question without additional explanation or commentary.

Input:
Class Description:
{{class_description}}

Seed Questions from {category}:
{{seed_questions}}
"""
        
        questions = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self.generate_for_class_by_category, cls, prompt_template, file, category)
                for cls in classes
            ]
            for future in concurrent.futures.as_completed(futures):
                questions.extend(future.result())
        
        return questions

    def generate_questions_for_functions_by_category(self, functions: List, seed_questions: List[str], category: str, file: str, repo_structure: RepositoryStructure) -> List[QAPair]:
        """Generate questions for functions in a specific category"""
        prompt_template = f"""
You are an expert software research assistant.

Given:
1. A function description extracted from a software repository.
2. A list of seed questions from the "{category}" category.

Task:
1. Based on the seed questions and the function description, generate **one single question** that is:
   - As difficult and complex as possible,
   - Requires multi-hop reasoning or deep technical understanding,
   - Not answerable by simple retrieval or direct lookup (i.e., not solvable by basic RAG methods),
   - Clearly related to the function description,
   - Technically precise and detailed,
   - Reflects the style and intent of the original seed questions but goes significantly deeper.
   - **Must not be a compound question** (e.g., no use of "and", "or", or comma-based subquestions),
   - **Must be not too long and syntactically simple**
   - **Must be specific to the "{category}" category**

2. The question should encourage advanced analysis, integration of multiple concepts, or insight beyond surface-level information.

3. Output only the single refined question without additional explanation or commentary.

Input:
Function Description:
{{function_description}}

Seed Questions from {category}:
{{seed_questions}}
"""
        
        questions = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self.generate_for_function_by_category, func, prompt_template, file, category, repo_structure)
                for func in functions
            ]
            for future in concurrent.futures.as_completed(futures):
                questions.extend(future.result())
        
        return questions

    def generate_for_class_by_category(self, cls, prompt_template, file, category) -> List[QAPair]:
        """Generate questions for a single class in a specific category"""
        class_description = f"Class: {cls}\n"
        seed_questions = self.template_questions.get(category, [])
        if not seed_questions:
            return []
        
        # Randomly sample some seed questions
        num_seed_questions = min(5, len(seed_questions))
        selected_seed_questions = random.sample(seed_questions, num_seed_questions)
        
        prompt = prompt_template.format(
            class_description=class_description, 
            seed_questions="\n".join(selected_seed_questions)
        )
        result = self._generate_qa_pairs_with_llm(prompt)
        print(f"Generated {len(result)} questions for class {cls.name} in category {category}")
        for q in result:
            print(f"Question: {q.question}")
        # Remove file writing here, handle uniformly in main method
        return result

    def generate_for_function_by_category(self, func, prompt_template, file, category, repo_structure) -> List[QAPair]:
        """Generate questions for a single function in a specific category"""
        function_description = f"Function: {func}\n"
        seed_questions = self.template_questions.get(category, [])
        if not seed_questions:
            return []
        
        # Randomly sample some seed questions
        num_seed_questions = min(5, len(seed_questions))
        selected_seed_questions = random.sample(seed_questions, num_seed_questions)
        
        prompt = prompt_template.format(
            function_description=function_description,
            seed_questions="\n".join(selected_seed_questions)
        )
        result = self._generate_qa_pairs_with_llm(prompt)
        print(f"Generated {len(result)} questions for function {func.name} in category {category}")
        for q in result:
            print(f"Question: {q.question}")
        # Remove file writing here, handle uniformly in main method
        return result

    # Keep original methods for compatibility
    def generate_for_class(self, cls, prompt_template, file) -> List[QAPair]:
        
        class_description = f"Class: {cls}\n"
        question_starts = ["how", "why", "what", "when"]
        start = random.choice(question_starts)
        seed_questions = self.random_select_seed_questions_by_category(start, num_questions=10)
        prompt = prompt_template.format(class_description=class_description, seed_questions="\n".join(seed_questions))
        result = self._generate_qa_pairs_with_llm(prompt)
        print(f"Generated {len(result)} questions for class {cls.name}")
        for q in result:
            print(f"Question: {q.question}")
            # File writing operation, if writing to same file, consider adding lock; simplified direct write here
        self.write_questions_to_file(result, file)
        return result
    
    def generate_questions_by_class_parallel(self, repo_structure: RepositoryStructure, file: str) -> List[QAPair]:
        prompt_template = """
You are an expert software research assistant.

Given:
1. A class description extracted from a software repository.
2. A list of seed questions that are general or vague.

Task:
1. Based on the seed questions and the class description, generate **one single question** that is:
   - As difficult and complex as possible,
   - Requires multi-hop reasoning or deep technical understanding,
   - Not answerable by simple retrieval or direct lookup (i.e., not solvable by basic RAG methods),
   - Clearly related to the class/module description,
   - Technically precise and detailed,
   - Reflects the style and intent of the original seed questions but goes significantly deeper.
   - **Must not be a compound question** (e.g., no use of "and", "or", or comma-based subquestions),
   - **Must be not too long and syntactically simple**


2. The question should encourage advanced analysis, integration of multiple concepts, or insight beyond surface-level information.

3. Output only the single refined question without additional explanation or commentary.

Input:
Class Description:
{class_description}

Seed Questions:
{seed_questions}


"""
        questions = []
        sampled_classes = random.sample(repo_structure.classes, k=min(50, len(repo_structure.classes)))

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers_class) as executor:
            futures = [
                executor.submit(self.generate_for_class, cls, prompt_template, file)
                for cls in sampled_classes
            ]
            for future in concurrent.futures.as_completed(futures):
                questions.extend(future.result())

        return questions

    def generate_questions_by_class(self, repo_structure: RepositoryStructure, file: str) -> List[QAPair]:
        prompt_template = """
You are an expert software research assistant.

Given:
1. A class description extracted from a software repository.
2. A list of seed questions that are general or vague.

Task:
1. Based on the seed questions and the class description, generate **one single question** that is:
   - As difficult and complex as possible,
   - Requires multi-hop reasoning or deep technical understanding,
   - Not answerable by simple retrieval or direct lookup (i.e., not solvable by basic RAG methods),
   - Clearly related to the class/module description,
   - Technically precise and detailed,
   - Reflects the style and intent of the original seed questions but goes significantly deeper.
   - **Must not be a compound question** (e.g., no use of "and", "or", or comma-based subquestions),
   - **Must be not too long and syntactically simple**

2. The question should encourage advanced analysis, integration of multiple concepts, or insight beyond surface-level information.

3. Output only the single refined question without additional explanation or commentary.

Input:
Class Description:
{class_description}

Seed Questions:
{seed_questions}


"""
        questions = []
        for cls in repo_structure.classes:
            class_description = f"Class: {cls}\n"
            seed_questions = self.random_select_seed_questions()
            prompt = prompt_template.format(class_description=class_description, seed_questions="\n".join(seed_questions))
            result =self._generate_qa_pairs_with_llm(prompt)
            print(f"Generated {len(result)} questions for class {cls.name}")
            for q in result:
                print(f"Question: {q.question}")
            questions.extend(result)
            self.write_questions_to_file(result, file)
        return questions
    
    def generate_for_function(self, func, prompt_template, file) -> List[QAPair]:
        function_description = f"Function: {func}\n"
        question_starts = ["how", "why", "what", "when"]
    
        start = random.choice(question_starts)
        seed_questions = self.random_select_seed_questions_by_category(start, num_questions=10)
        prompt = prompt_template.format(
            function_description=function_description,
            seed_questions="\n".join(seed_questions),
            summary=self.generate_summary_of_repo(self.repo_structure)  # 注意这里self.repo_structure需可用，或者传入
        )
        result = self._generate_qa_pairs_with_llm(prompt)
        # Output logs
        print(f"Generated {len(result)} questions for function {func.name} with start word '{start}'")
        for q in result:
            print(f"Question: {q.question}")
        # File writing operation, note synchronization when threads write to same file or use different files
        self.write_questions_to_file(result, file)
        return result

    def generate_questions_by_function_parallel(self, repo_structure: RepositoryStructure, file: str) -> List[QAPair]:
        prompt_template = """
You are an expert software research assistant.

Given:
1. A function description extracted from a software repository.
2. A list of seed questions that are general or vague.

Task:
1. Based on the seed questions and the function description, generate **one single question** that is:
   - As difficult and complex as possible,
   - Requires multi-hop reasoning or deep technical understanding,
   - Not answerable by simple retrieval or direct lookup (i.e., not solvable by basic RAG methods),
   - Clearly related to the function description,
   - Technically precise and detailed,
   - Reflects the style and intent of the original seed questions but goes significantly deeper.

2. The question should encourage advanced analysis, integration of multiple concepts, or insight beyond surface-level information.

3. Output only the single refined question without additional explanation or commentary.

Input:
Function Description:
{function_description}

Seed Questions:
{seed_questions}

"""
        questions = []
        self.repo_structure = repo_structure  # For easy access by generate_for_function, can also be changed to parameter passing if needed
        sampled_functions = random.sample(repo_structure.functions, k=min(50, len(repo_structure.functions)))
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers_function) as executor:
            futures = [
                executor.submit(self.generate_for_function, func, prompt_template, file)
                # for func in repo_structure.functions
                for func in sampled_functions
            ]
            for future in concurrent.futures.as_completed(futures):
                questions.extend(future.result())
        return questions

    def generate_questions_by_function(self, repo_structure: RepositoryStructure, file: str) -> List[QAPair]:
        prompt = """
You are an expert software research assistant.

Given:
1. A function description extracted from a software repository.
2. A list of seed questions that are general or vague.

Task:
1. Based on the seed questions and the function description, generate **one single question** that is:
   - As difficult and complex as possible,
   - Requires multi-hop reasoning or deep technical understanding,
   - Not answerable by simple retrieval or direct lookup (i.e., not solvable by basic RAG methods),
   - Clearly related to the function description,
   - Technically precise and detailed,
   - Reflects the style and intent of the original seed questions but goes significantly deeper.

2. The question should encourage advanced analysis, integration of multiple concepts, or insight beyond surface-level information.

3. Output only the single refined question without additional explanation or commentary.

Input:
Function Description:
{function_description}

Seed Questions:
{seed_questions}

"""
        questions = []
        for func in repo_structure.functions:
            function_description = f"Function: {func}\n"
            seed_questions = self.random_select_seed_questions()
            prompt.format(function_description=function_description, seed_questions="\n".join(seed_questions), summary=self.generate_summary_of_repo(repo_structure))
            result =self._generate_qa_pairs_with_llm(prompt)
            questions.extend(result)
            self.write_questions_to_file(result, file)
        return questions

    def random_select_seed_questions(self, num_questions: int = 10) -> List[str]:
        """Randomly select a subset of seed questions for a specific class"""
        what_questions = self.template_questions.get("what", [])
        how_questions = self.template_questions.get("how", [])
        why_questions = self.template_questions.get("why", [])
        where_questions = self.template_questions.get("where", [])
        seed_questions = what_questions + how_questions + why_questions + where_questions
        return random.sample(seed_questions, min(num_questions, len(seed_questions)))

    def random_select_seed_questions_by_category(self, category: str, num_questions: int = 10) -> List[str]:
        """Randomly select a subset of seed questions for a specific category"""
        if category not in self.template_questions:
            raise ValueError(f"Category '{category}' not found in template questions.")
        questions = self.template_questions.get(category, [])
        return random.sample(questions, min(num_questions, len(questions)))

    def write_questions_to_file(self, questions: List[QAPair], output_file: str):
        """Append QAPair list to a .jsonl file, one JSON per line."""
        with open(output_file, 'a', encoding='utf-8') as f:
            for qa in questions:
                json_line = json.dumps(qa.model_dump(), ensure_ascii=False)
                f.write(json_line + '\n')

    def write_categorized_questions_to_file(self, categorized_questions: dict, output_file: str):
        """Write questions to file organized by category hierarchy"""
        # Create output file organized by category
        base_name = output_file.rsplit('.', 1)[0] if '.' in output_file else output_file
        categorized_file = f"{base_name}_categorized.jsonl"
        
        # Generate statistics
        total_questions = 0
        category_stats = {}
        
        with open(categorized_file, 'w', encoding='utf-8') as f:
            # Write category hierarchy structure
            for category_type, categories in categorized_questions.items():
                f.write(f"# {category_type.upper()} CATEGORIES\n")
                category_stats[category_type] = {}
                
                for category, questions in categories.items():
                    f.write(f"## {category}\n")
                    f.write(f"# Generated {len(questions)} questions\n")
                    category_stats[category_type][category] = len(questions)
                    total_questions += len(questions)
                    
                    for qa in questions:
                        # Add category information to question data
                        qa_data = qa.model_dump()
                        qa_data['category_type'] = category_type
                        qa_data['category'] = category
                        json_line = json.dumps(qa_data, ensure_ascii=False)
                        f.write(json_line + '\n')
                    
                    f.write(f"# End of {category}\n\n")
                
                f.write(f"# End of {category_type.upper()}\n\n")
            
            # Write statistics
            f.write(f"# STATISTICS\n")
            f.write(f"# Total questions generated: {total_questions}\n")
            for category_type, categories in category_stats.items():
                type_total = sum(categories.values())
                f.write(f"# {category_type.upper()}: {type_total} questions\n")
                for category, count in categories.items():
                    f.write(f"#   - {category}: {count} questions\n")
        
        print(f"Categorized questions written to: {categorized_file}")
        print(f"Total questions generated: {total_questions}")
        
        # Also write file in original format (maintain backward compatibility)
        with open(output_file, 'w', encoding='utf-8') as f:
            for category_type, categories in categorized_questions.items():
                for category, questions in categories.items():
                    for qa in questions:
                        qa_data = qa.model_dump()
                        qa_data['category_type'] = category_type
                        qa_data['category'] = category
                        json_line = json.dumps(qa_data, ensure_ascii=False)
                        f.write(json_line + '\n')
        
        print(f"All questions written to: {output_file}")
        
        
        
    
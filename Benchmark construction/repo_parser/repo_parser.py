import ast
import networkx as nx
import uuid
from typing import List, Dict, Optional, Set, Tuple, Any
import re
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm
from models.data_models import (
    FileNode, CodeNode, ClassDefinition, FunctionDefinition, 
    ClassAttribute, RepositoryStructure, Repository, 
    ModuleNode, CodeRelationship, VariableDefinition
)

class CodeAnalyzerSimple:
    def __init__(self):
        self.dependency_graph = nx.DiGraph()
        self.repository_structure = RepositoryStructure()
        self.current_content = ""  # Store current file content being analyzed
        self.relationships = []  # Store relationships between code elements
        self._file_cache: dict[str, FileNode | None] = {}  # Cache file analysis results to avoid repeated analysis
        
    def analyze_file(self, file_path: str, repo_root: str) -> Optional[FileNode]:
        """Analyze a single file to extract import relationships and class definitions"""
        if file_path in self._file_cache:
            return self._file_cache[file_path]
        
        print(f"Starting file analysis: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            tree = ast.parse(content)
            imports = []
            classes = []
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    imports.extend(self._extract_imports(node))
                elif isinstance(node, ast.ClassDef):
                    classes.append(node.name)
            if repo_root:
                upper_path = os.path.relpath(os.path.dirname(file_path), repo_root)
            else:
                upper_path = os.path.dirname(file_path)
                
            node = FileNode(
                file_name=os.path.basename(file_path),
                upper_path=os.path.dirname(file_path),
                module=os.path.basename(os.path.dirname(file_path)),
                define_class=classes,
                imports=imports
            )
            self._file_cache[file_path] = node
            return node
        except SyntaxError:
            print(f"Warning: File {file_path} has syntax errors, skipped")
        except UnicodeDecodeError:
            print(f"Warning: File {file_path} has encoding errors, skipped")
        except Exception as e:
            print(f"Warning: Error occurred while analyzing file {file_path}: {str(e)}")
        return None
    
    def build_dependency_graph(self, files: List[str],repo_root: str):
        """Build dependency graph from files"""
        dependency_graph = {}
        file_nodes = {}
        
        # First pass: Load all files and create file nodes
        for file_path in files:
            file_node = self.analyze_file(file_path,repo_root)
            if file_node:
                module_path = os.path.dirname(file_path)
                file_nodes[file_path] = file_node
                if module_path not in dependency_graph:
                    dependency_graph[module_path] = []
        
        # Second pass: Build dependency relationships
        for file_path, file_node in file_nodes.items():
            module_path = os.path.dirname(file_path)
            for imported in file_node.imports:
                # Look for possible import targets
                for potential_source in files:
                    if os.path.basename(potential_source).replace('.py', '') == imported.split('.')[-1]:
                        target_module = os.path.dirname(potential_source)
                        if target_module not in dependency_graph[module_path]:
                            dependency_graph[module_path].append(target_module)
        
        self.repository_structure.dependency_graph = dependency_graph
        return dependency_graph
    
    def _extract_imports(self, node: ast.AST) -> List[str]:
        """Extract import statements"""
        imports = []
        if isinstance(node, ast.Import):
            for name in node.names:
                if hasattr(name, 'name'):
                    imports.append(name.name)
                elif isinstance(name, str):
                    imports.append(name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ''
            for name in node.names:
                if hasattr(name, 'name'):
                    imports.append(f"{module}.{name.name}")
                elif isinstance(name, str):
                    imports.append(f"{module}.{name}")
        return imports
    
    def _get_related_functions(self, node: ast.AST) -> List[str]:
        """Get related function calls"""
        # Get source code of function body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            body = ast.get_source_segment(self.current_content, node)
            if body:
                # Use enhanced call analysis functionality
                return self.extract_calls_in_order(body)
        return []

    def simple_extract_calls_in_order(self, body: str) -> List[str]:
        """Simple function call extraction (fallback for syntax errors)"""
        # Extract function signature
        func_signature = re.findall(
            r"(([^\s\r\n]+.*?\s*)?\s*def\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\(.*?\)\s*(?:->\s*['\"a-zA-Z0-9\[\]_.,\s\|]*['\"]*)?:)",
            body, re.DOTALL) 
        if func_signature:
            body = body.removeprefix(func_signature[0][0])
        body_lines = [line for line in body.split("\n") if line.strip() != ""]
        calls = []

        def extract_parts(code_line):
            result = []
            # Match part before () (function calls, including a(), a.b())
            pattern = r'([a-zA-Z_][\w]*(\.[a-zA-Z_][\w]*)*)(?=\()'
            call_result = re.findall(pattern, code_line)
            if call_result:
                call_result = [item[0] for item in call_result]

            # Match object references without parentheses (like a.b)
            dot_result = re.findall(r'([a-zA-Z_][\w]*(\.[a-zA-Z_][\w]*)*)', code_line)

            # Filter out unnecessary parts (like keywords, single letters, etc.)
            dot_result = [item[0] for item in dot_result if len(item[0].split('.')) > 1 and len(item[0]) > 1]
            result.extend(call_result)
            result.extend(dot_result)
            return result

        for line in body_lines:
            calls.extend(extract_parts(line))

        return calls

    def extract_calls_in_order(self, body: str) -> List[str]:
        """Extract all calls from function body in order"""
        calls = []
        visited = set()  # Store already processed call paths to avoid duplicate extraction

        # Parse code using AST
        try:
            tree = ast.parse(body)
        except SyntaxError:
            return self.simple_extract_calls_in_order(body)

        def get_node(node):
            # Handle function calls
            if isinstance(node, ast.Call):
                # Handle a(params).b(params) and a.b.c(params)
                if isinstance(node.func, ast.Attribute):
                    # Case of a.b.c(params), extract a.b.c
                    call = self._get_attribute_call(node.func)
                    if call not in visited:
                        calls.append(call)
                        visited.add(call)
                        # Also mark its module path (like jax.random)
                        module_path = '.'.join(call.split('.')[:-1])
                        visited.add(module_path)
                    get_node(node.func.value)

                elif isinstance(node.func, ast.Name):
                    # Case of a(params)
                    call = node.func.id
                    if call not in visited:
                        calls.append(call)
                        visited.add(call)

                else:
                    get_node(node.func)

                # Recursively check calls in parameters
                for arg in node.args:
                    get_node(arg)

                for keyword in node.keywords:
                    get_node(keyword.value)

            # Handle attribute calls
            elif isinstance(node, ast.Attribute):
                # Case of a.b, extract as field access
                field_access = self._get_attribute_call(node)
                if field_access not in visited:
                    calls.append(field_access)
                    visited.add(field_access)
                    module_path = '.'.join(field_access.split('.')[:-1])
                    visited.add(module_path)
                get_node(node.value)

            # Handle other types of nodes
            elif isinstance(node, ast.Expr):
                get_node(node.value)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    get_node(target)
                get_node(node.value)
            elif isinstance(node, ast.Subscript):
                get_node(node.value)
            elif isinstance(node, ast.Name):
                pass  # Simplified handling, not considering global variables and imported variables
            elif isinstance(node, ast.Starred):
                get_node(node.value)
            elif isinstance(node, (ast.BinOp, ast.UnaryOp)):
                if isinstance(node, ast.BinOp):
                    get_node(node.left)
                    get_node(node.right)
                else:
                    get_node(node.operand)
            elif isinstance(node, ast.BoolOp):
                for value in node.values:
                    get_node(value)
            elif isinstance(node, ast.Compare):
                get_node(node.left)
                for comparator in node.comparators:
                    get_node(comparator)
            elif isinstance(node, (ast.List, ast.Tuple, ast.Set)):
                for element in node.elts:
                    get_node(element)
            elif isinstance(node, ast.Dict):
                for key, value in zip(node.keys, node.values):
                    if key is not None:  # Handle dict unpacking case
                        get_node(key)
                    get_node(value)
            elif isinstance(node, ast.Lambda):
                get_node(node.body)
            elif isinstance(node, (ast.If, ast.While)):
                get_node(node.test)
                for stmt in node.body:
                    get_node(stmt)
                if node.orelse:
                    for stmt in node.orelse:
                        get_node(stmt)
            elif isinstance(node, ast.For):
                get_node(node.iter)
                get_node(node.target)
                for stmt in node.body:
                    get_node(stmt)
                if node.orelse:
                    for stmt in node.orelse:
                        get_node(stmt)
            elif isinstance(node, ast.Assert):
                get_node(node.test)
            elif isinstance(node, (ast.Try, ast.ExceptHandler, ast.With)):
                if isinstance(node, ast.Try):
                    for stmt in node.body:
                        get_node(stmt)
                    for handler in node.handlers:
                        get_node(handler)
                    if node.finalbody:
                        for stmt in node.finalbody:
                            get_node(stmt)
                elif isinstance(node, ast.With):
                    for item in node.items:
                        get_node(item.context_expr)
                        if item.optional_vars:
                            get_node(item.optional_vars)
                    for stmt in node.body:
                        get_node(stmt)
                elif isinstance(node, ast.ExceptHandler):
                    for stmt in node.body:
                        get_node(stmt)
            elif isinstance(node, ast.Raise):
                if node.exc:
                    get_node(node.exc)
                if node.cause:
                    get_node(node.cause)
            elif isinstance(node, ast.Await):
                get_node(node.value)
            elif isinstance(node, ast.Return):
                if node.value:
                    get_node(node.value)

        # Traverse AST and find function calls and method calls
        for node in ast.walk(tree):
            get_node(node)
            if isinstance(node, ast.Return):
                break

        return calls

    def _get_attribute_call(self, node: ast.Attribute) -> str:
        """Extract complete path of attribute calls"""
        parts = []
        current = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        return '.'.join(reversed(parts))
    
    def extract_class_docstring_by_pattern(self, code:str, class_name:str) -> Optional[str]:
        match = re.search(rf'class {class_name}\s*(\([^\)]*\))?\s*:\s*"""([\s\S]*?)"""', code, re.DOTALL)

        if match:
            return match.group(2)
        return None
    
    def extract_function_docstring_by_pattern(self, code:str, function_name:str) -> Optional[str]:
        match = re.search(rf'def {function_name}\s*(\([\s\S]*?\))?\s*:\s*"""([\s\S]*?)"""', code, re.DOTALL)

        if match:
            return match.group(2)
        return None
        
    def analyze_repository(self, root_path: str,repo_root: str) -> Repository:
        
        def analyze_wrapper(file_path):
            return self._analyze_file_for_structure(file_path, repo_root)
        
        """Analyze entire code repository and extract key structural information"""
        # Create repository object
        repo_id = f"repo-{uuid.uuid4().hex[:8]}"
        repo_name = os.path.basename(os.path.abspath(root_path))
        repository = Repository(
            id=repo_id,
            name=repo_name,
            url=None,  # Can be obtained from git configuration
            description=None  # Can be obtained from README or setup.py
        )
        
        # Initialize repository structure and code node lists
        self.repository_structure = RepositoryStructure()
        
        # Get all Python files in the repository
        python_files = self._get_python_files(root_path)
        print(f"Found {len(python_files)} Python files\n")

        # Analyze file structure and build module tree
        root_modules = self._build_module_tree(root_path, python_files,repo_root)
        self.repository_structure.root_modules = root_modules
        print(f"Module tree structure has been built\n")

        max_workers = min(32, len(python_files))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = {executor.submit(analyze_wrapper, file_path): file_path for file_path in python_files}

            # Use tqdm to track progress
            for future in tqdm(as_completed(futures), total=len(futures), desc="analyze_repository: Analyzing files (parallel)"):
                try:
                    future.result()  # If you want to catch exceptions, handle them here
                except Exception as e:
                    print(f"Error analyzing {futures[future]}: {e}")

        # # # Analyze code structure of each file
        # for file_path in tqdm(python_files, desc="Analyzing files"):
        #     self._analyze_file_for_structure(file_path, repo_root)
        
        # # Build code dependency graph
        # self.build_dependency_graph(python_files,repo_root)
        # print(f"Code dependency graph has been built, total {len(self.repository_structure.dependency_graph)} modules\n")

        # # Analyze relationships between code elements
        # self._extract_code_relationships()      
        # print(f"Relationships between code elements have been extracted, total {len(self.repository_structure.relationships)} relationships\n")  
        
        # # Link class attributes with functions
        # self._link_attributes_to_functions()
        # print(f"Relationships between class attributes and functions have been linked, total {len(self.repository_structure.attributes)} attributes\n")

        # # Link variables with functions that reference them
        # self._link_variables_to_references()
        # print(f"Relationships between variables and functions that reference them have been linked, total {len(self.repository_structure.variables)} variables\n")
        
        # # Generate repository core functionality overview
        # self._summarize_core_functionality()
        # print(f"Repository core functionality overview has been generated\n")
        
        # Add repository structure to repository object
        repository.structure = self.repository_structure
        
        
        return repository
    
    def _get_python_files(self, root_path: str) -> List[str]:
        """Get all Python files under given path"""
        python_files = []
        for root, _, files in os.walk(root_path):
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
        return python_files
    
    def _analyze_file_for_structure(self, file_path: str,repo_root: str):
        """Analyze single file, extract class definitions, function definitions and class attributes"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # self.current_content = f.read()
                current_content = f.read()
            
            tree = ast.parse(current_content)
            
            # Extract top-level definitions
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    # Handle class definition
                    self._extract_class_definition(node, file_path, current_content,repo_root)                    # Handle methods and attributes in class
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            self._extract_function_definition(item, file_path, current_content, node,repo_root)
                        elif isinstance(item, ast.Assign):
                            self._extract_class_attributes(item, file_path, node)
                            # Extract class variables
                            self._extract_variables(item, file_path, scope="class", class_name=node.name, function_name=None, repo_root=repo_root, content=current_content)
                
                elif isinstance(node, ast.FunctionDef):
                    # Handle functions
                    self._extract_function_definition(node, file_path, current_content, None, repo_root)
                    # Extract variables in function
                    self._extract_function_variables(node, file_path, repo_root, current_content)
                
                elif isinstance(node, ast.Assign):
                    # Extract global variables
                    self._extract_variables(node, file_path, scope="global", class_name=None, function_name=None, repo_root=repo_root, content=current_content)
        except SyntaxError:
            print(f"Warning: File {file_path} has syntax errors, skipped")
        except UnicodeDecodeError:
            print(f"Warning: File {file_path} has encoding errors, skipped")
        except Exception as e:
            print(f"Warning: Error occurred while analyzing file {file_path}: {str(e)}")
        finally:
            self.current_content = ""  # Clear current content

    def _extract_class_definition(self, node: ast.ClassDef, file_path: str, content: str,repo_root: str):
        """Extract class definition information"""
        docstring = ast.get_docstring(node) or ''
        
        # Create CodeNode
        file_node = self.analyze_file(file_path,repo_root)
        code_node = None
        if file_node:
            code_node = CodeNode(
                start_line=node.lineno,
                end_line=node.end_lineno,
                belongs_to=file_node,
                relative_function=[],
                code=ast.get_source_segment(content, node)
            )
        
        class_def = ClassDefinition(
            name=node.name,
            docstring=docstring,
            relative_code=code_node,
            methods=[],  # Will be filled in subsequent processing
            attributes=[]  # Will be filled in subsequent processing
        )
        
        self.repository_structure.classes.append(class_def)
    
    def _extract_function_definition(self, node: ast.FunctionDef, file_path: str, content: str, current_class: Optional[ast.ClassDef] = None,repo_root: str=None):
        """Extract function/method definition information"""
        docstring = ast.get_docstring(node) or ''
        
        # Create CodeNode
        file_node = self.analyze_file(file_path,repo_root)
        code_node = None
        if file_node:
            code_node = CodeNode(
                start_line=node.lineno,
                end_line=node.end_lineno,
                belongs_to=file_node,
                relative_function=[],
                code=ast.get_source_segment(content, node)
            )
        
        # Extract parameter list
        parameters = []
        for arg in node.args.args:
            parameters.append(arg.arg)
        
        # Extract function calls
        calls = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    calls.append(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    calls.append(self._get_attribute_call(child.func))
        is_method = current_class is not None
        class_name = current_class.name if current_class else None
        func_def = FunctionDefinition(
            name=node.name,
            docstring=docstring,
            relative_code=code_node,
            is_method=is_method,
            class_name=class_name,
            parameters=parameters,
            calls=calls
        )
        
        self.repository_structure.functions.append(func_def)
        
        # If it's a method, add it to the corresponding class's method list
        if is_method:
            for cls in self.repository_structure.classes:
                if cls.name == class_name:
                    cls.methods.append(func_def)
                    break

    def _extract_class_attributes(self, node: ast.Assign, file_path: str, current_class: ast.ClassDef):
        """Extract class attribute information"""
        for target in node.targets:
            if isinstance(target, ast.Name):
                attr_name = target.id
                
                # Try to extract type hints (if any)
                type_hint = None
                if hasattr(target, 'annotation') and target.annotation:
                    type_hint = ast.unparse(target.annotation)
                
                attr = ClassAttribute(
                    name=attr_name,
                    class_name=current_class.name,
                    file_path=file_path,
                    type_hint=type_hint
                )
                
                self.repository_structure.attributes.append(attr)
                
                # Add attribute to corresponding class's attribute list
                for cls in self.repository_structure.classes:
                    if cls.name == current_class.name:
                        cls.attributes.append(attr)
                        break
    
    def _link_attributes_to_functions(self):
        class_func_map = {}
        for func in self.repository_structure.functions:
            if func.class_name not in class_func_map:
                class_func_map[func.class_name] = []
            class_func_map[func.class_name].append(func)
    
        def process_attribute(attr):
            attr_related = []
            if attr.class_name in class_func_map:
                for func in class_func_map[attr.class_name]:
                    if func.relative_code:
                        func_code = func.relative_code.code
                        if f"self.{attr.name}" in func_code:
                            attr_related.append(func.name)
            return (attr, attr_related)

        # Use thread pool for parallel processing
        max_workers = min(32, len(self.repository_structure.attributes))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all attribute processing tasks
            futures = [executor.submit(process_attribute, attr) 
                for attr in self.repository_structure.attributes]
        
            # Use tqdm to show progress and collect results
            for future in tqdm(as_completed(futures),total=len(futures),desc="_link_attributes_to_functions: Linking attributes to functions"):
                attr, related_funcs = future.result()
                attr.related_functions.extend(related_funcs)

    def _summarize_core_functionality(self):
        """Summarize repository core functionality based on function comments and code structure"""
        # This is a simplified implementation, may need LLM summarization later
        
        class_summaries = []
        for cls in self.repository_structure.classes:
            if cls.docstring:
                class_summaries.append(f"{cls.name}: {cls.docstring.strip()}")
        
        function_summaries = []
        for func in self.repository_structure.functions:
            if func.docstring and not func.is_method:  # Only consider top-level functions
                function_summaries.append(f"{func.name}: {func.docstring.strip()}")
        
        # Combine summary
        summary = "Repository core functionality:\n"
        
        if class_summaries:
            summary += "\nMain classes:\n" + "\n".join(class_summaries) + "\n"
            
        if function_summaries:
            summary += "\nMain functions:\n" + "\n".join(function_summaries)
            
        self.repository_structure.core_functionality = summary

    def _build_module_tree(self, root_path: str, python_files: List[str],repo_root: str) -> List[ModuleNode]:
        """Build module tree structure"""
        # Create root node of module tree
        root_modules = []
        
        # Create cache to avoid repeated creation of module nodes
        module_cache = {}
        
        # Normalize root path
        root_path = os.path.abspath(root_path)
        
        for file_path in python_files:
            # Get file path relative to root directory
            rel_path = os.path.relpath(file_path, root_path)
            dir_path = os.path.dirname(rel_path)
            
            # Skip hidden folders
            if any(part.startswith('.') for part in dir_path.split(os.sep)):
                continue
                
            # Split path into module hierarchy
            if dir_path:
                parts = dir_path.split(os.sep)
            else:
                parts = []
                
            # Create or update module tree
            current_modules = root_modules
            current_path = ""
            
            for i, part in enumerate(parts):
                # Build current module path
                if current_path:
                    current_path = os.path.join(current_path, part)
                else:
                    current_path = part
                    
                # Check if module already exists
                module_node = None
                for mod in current_modules:
                    if mod.name == part:
                        module_node = mod
                        break
                        
                # If module doesn't exist, create new module node
                if not module_node:
                    is_package = os.path.exists(os.path.join(root_path, current_path, '__init__.py'))
                    module_node = ModuleNode(
                        name=part,
                        path=os.path.join(root_path, current_path),
                        is_package=is_package
                    )
                    current_modules.append(module_node)
                    module_cache[current_path] = module_node
                    
                # Update current module list
                current_modules = module_node.sub_modules
                
            # Add file to the last level module
            file_node = self.analyze_file(file_path,repo_root)
            if file_node:
                if dir_path and dir_path in module_cache:
                    module_cache[dir_path].files.append(file_node)
                elif not dir_path and os.path.basename(file_path) != '__init__.py':
                    # Handle individual files in root directory
                    is_package = False
                    root_module = ModuleNode(
                        name=os.path.basename(file_path).replace('.py', ''),
                        path=root_path,
                        is_package=is_package
                    )
                    root_module.files.append(file_node)
                    root_modules.append(root_module)
                    
        return root_modules
    
    def _extract_code_relationships(self):
        """Extract relationships between code elements"""
        relationships = []
        
        # Extract inheritance relationships
        # for cls in tqdm(self.repository_structure.classes, desc="_extract_code_relationships: Analyzing class inheritance"):
        def process_class(cls):
            cls_relationships = []
            # Use content from code node of class definition instead of reopening file
            if hasattr(cls, 'relative_code') and cls.relative_code:
                try:
                    content = cls.relative_code.code
                    tree = ast.parse(content)

                    # Find class definition node
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef) and node.name == cls.name:
                            # Extract parent classes
                            for base in node.bases:
                                if isinstance(base, ast.Name):
                                # Simple parent class names
                                    parent_class = base.id
                                    relationship = CodeRelationship(
                                        source_type="class",
                                        source_id=cls.name,
                                        target_type="class",
                                        target_id=parent_class,
                                        relationship_type="inherits"
                                    )
                                    relationships.append(relationship)
                                elif isinstance(base, ast.Attribute):
                            # Complex parent class references, like module.Class
                                    parent_class = self._get_attribute_call(base)
                                    relationship = CodeRelationship(
                                        source_type="class",
                                        source_id=cls.name,
                                        target_type="class",
                                        target_id=parent_class,
                                        relationship_type="inherits"
                                    )
                                    relationships.append(relationship)
                except Exception as e:
                    print(f"Error extracting inheritance relationship for class {cls.name}: {str(e)}")
            return cls_relationships
        
        max_workers_class = min(32, len(self.repository_structure.classes))
        with ThreadPoolExecutor(max_workers=max_workers_class) as executor:
            # Use list comprehension to submit all tasks
            futures = [executor.submit(process_class, cls) for cls in self.repository_structure.classes]
            # Use tqdm to show progress
            for future in tqdm(as_completed(futures), total=len(futures),desc="_extract_code_relationships: Analyzing class inheritance"):
                relationships.extend(future.result())
       
        def process_function(func):
            func_relationships = []
            for call in func.calls:
                # Check if called function is in known function list
                target_func = next((f for f in self.repository_structure.functions if f.name == call), None)
                if target_func:
                    relationship = CodeRelationship(
                        source_type="function",
                        source_id=func.name,
                        target_type="function",
                        target_id=call,
                        relationship_type="calls"
                    )
                    func_relationships.append(relationship)
            return func_relationships
        
        max_workers_function = min(32, len(self.repository_structure.functions))
        with ThreadPoolExecutor(max_workers=max_workers_function) as executor:
            # Use list comprehension to submit all tasks
            futures = {executor.submit(process_function, func): func for func in self.repository_structure.functions}
            # Use tqdm to show progress
            for future in tqdm(as_completed(futures), total=len(futures), desc="_extract_code_relationships: Analyzing function calls"):
                relationships.extend(future.result())

        self.repository_structure.relationships = relationships
        return relationships
    
    def _extract_variables(self, node: ast.Assign, file_path: str, scope: str, class_name: Optional[str], function_name: Optional[str], repo_root: str, content: str):
        """Extract variable definition information"""
        file_node = self.analyze_file(file_path, repo_root)
        
        for target in node.targets:
            # Handle simple variable assignment
            if isinstance(target, ast.Name):
                var_name = target.id
                
                # Check if it's a constant (all uppercase naming)
                is_constant = var_name.isupper() and "_" in var_name
                
                # Get string representation of variable value
                value = None
                try:
                    value = ast.unparse(node.value)
                except:
                    try:
                        value = str(ast.literal_eval(node.value))
                    except:
                        pass

                # Create CodeNode
                code_node = None
                if file_node:
                    code_segment = ast.get_source_segment(content, node)
                    if code_segment:
                        code_node = CodeNode(
                            start_line=node.lineno,
                            end_line=node.end_lineno,
                            belongs_to=file_node,
                            relative_function=[function_name] if function_name else [],
                            code=code_segment
                        )
                
                # Try to extract type hints (if any)
                type_hint = None
                if hasattr(target, 'annotation') and target.annotation:
                    try:
                        type_hint = ast.unparse(target.annotation)
                    except:
                        pass
                
                var_def = VariableDefinition(
                    name=var_name,
                    scope=scope,
                    function_name=function_name,
                    class_name=class_name,
                    type_hint=type_hint,
                    value=value,
                    is_constant=is_constant,
                    relative_code=code_node,
                    references=[]  # Will be filled in subsequent analysis
                )
                
                self.repository_structure.variables.append(var_def)

            # Handle tuple assignment, like a, b = 1, 2
            elif isinstance(target, ast.Tuple):
                for elt in target.elts:
                    if isinstance(elt, ast.Name):
                        var_name = elt.id
                        
                        # Check if it's a constant
                        is_constant = var_name.isupper() and "_" in var_name
                        
                        # Create CodeNode
                        code_node = None
                        if file_node:
                            code_segment = ast.get_source_segment(content, node)
                            if code_segment:
                                code_node = CodeNode(
                                    start_line=node.lineno,
                                    end_line=node.end_lineno,
                                    belongs_to=file_node,
                                    relative_function=[function_name] if function_name else [],
                                    code=code_segment
                                )
                        
                        var_def = VariableDefinition(
                            name=var_name,
                            scope=scope,
                            function_name=function_name,
                            class_name=class_name,
                            value=None,  # Tuple assignment difficult to determine specific value
                            is_constant=is_constant,
                            relative_code=code_node,
                            references=[]
                        )
                        
                        self.repository_structure.variables.append(var_def)

    def _extract_function_variables(self, node: ast.FunctionDef, file_path: str, repo_root: str, content:str):
        """Extract all variables in function"""
        function_name = node.name
        class_name = None
        
        # Find the class this function belongs to (if any)
        for cls in self.repository_structure.classes:
            for method in cls.methods:
                if method.name == function_name and method.is_method:
                    class_name = cls.name
                    break
            if class_name:
                break
        
        # Recursively traverse function body to find variable definitions
        for item in node.body:
            if isinstance(item, ast.Assign):
                self._extract_variables(item, file_path, "local", class_name, function_name, repo_root, content)
            elif isinstance(item, ast.For):
                # Handle for loop variables
                if isinstance(item.target, ast.Name):
                    var_name = item.target.id
                    self._add_for_loop_variable(var_name, item, file_path, class_name, function_name, repo_root, content)
    
    def _add_for_loop_variable(self, var_name: str, node: ast.For, file_path: str, class_name: Optional[str], function_name: str, repo_root: str, content:str):
        """Add variables in for loop"""
        file_node = self.analyze_file(file_path, repo_root)
        
        # Create CodeNode
        code_node = None
        if file_node:
            code_segment = ast.get_source_segment(content, node)
            if code_segment:
                code_node = CodeNode(
                    start_line=node.lineno,
                    end_line=node.end_lineno,
                    belongs_to=file_node,
                    relative_function=[function_name] if function_name else [],
                    code=code_segment
                )
        
        var_def = VariableDefinition(
            name=var_name,
            scope="local",
            function_name=function_name,
            class_name=class_name,
            relative_code=code_node,
            references=[]
        )
        
        self.repository_structure.variables.append(var_def)

    def process_variable(self, index: int) :
        """Handle single variable, find functions that reference it"""
        functions = self.repository_structure.functions
        for func in functions:
            if not func.relative_code or not func.relative_code.code:
                continue

            pattern = r'\b' + re.escape(self.repository_structure.variables[index].name) + r'\b'
            if re.search(pattern, func.relative_code.code):
                self.repository_structure.variables[index].references.append(func.name)

    def _link_variables_to_references(self):
        """Link variables with functions that reference them"""
        # For each variable, find functions that reference it
        # Output number of variables and functions
        variable_num = len(self.repository_structure.variables)
        print(f"Linking variables with functions that reference them, total { variable_num } variables, {len(self.repository_structure.functions)} functions")

        max_workers = min(32, variable_num)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.process_variable, idx) for idx in range(variable_num)]
            for future in tqdm(as_completed(futures), total=len(futures), desc="_link_variables_to_references: Linking variables to references"):
                try:
                    _ = future.result()
                except Exception as e:
                    print(f"[ERROR] Task exception: {e}")

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import sys
import argparse
from pathlib import Path
import time
from datetime import datetime

# Add project root directory to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from repo_parser import CodeAnalyzerSimple
from models.data_models import Repository, ModuleNode

def analyze_repository(repo_path: str,repo_root: str):
    """
    Analyze code repository structure and return Repository object
    
    Args:
        repo_path: Path to the code repository
        
    Returns:
        Repository: Object containing repository structure
    """
    print(f"Starting repository analysis: {repo_path}")
    start_time = time.time()
    
    # Create code analyzer
    analyzer = CodeAnalyzerSimple()
    
    # Analyze entire repository
    repository = analyzer.analyze_repository(repo_path,repo_root)
    
    elapsed_time = time.time() - start_time
    print(f"Repository analysis completed, time taken: {elapsed_time:.2f} seconds")
    
    return repository

def save_repository_data(repository: Repository, output_dir: str, repo_name: str | None):
    """
    Save repository data in multiple formats
    
    Args:
        repository: Repository object
        output_dir: Output directory
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Current time as filename prefix
    if not repo_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        timestamp = repo_name

    # Save complete data as JSON
    full_json_path = os.path.join(output_dir, f"{timestamp}_repo_full.json")
    with open(full_json_path, 'w', encoding='utf-8') as f:
        json.dump(repository.model_dump(), f, ensure_ascii=False, indent=2)
    print(f"Complete repository data saved to: {full_json_path}")
    
    # Save class definition summary
    classes_json_path = os.path.join(output_dir, f"{timestamp}_classes.json")
    classes_data = [{
        "name": cls.name,
        "docstring": cls.docstring,
        "methods": [m.name for m in cls.methods] if hasattr(cls, 'methods') else [],
        "attributes": [a.name for a in cls.attributes] if hasattr(cls, 'attributes') else [],
        "code_location": {
            "file": cls.relative_code.belongs_to.file_name if cls.relative_code and cls.relative_code.belongs_to else None,
            "path": cls.relative_code.belongs_to.upper_path if cls.relative_code and cls.relative_code.belongs_to else None,
            "start_line": cls.relative_code.start_line if cls.relative_code else None,
            "end_line": cls.relative_code.end_line if cls.relative_code else None
        } if hasattr(cls, 'relative_code') and cls.relative_code else None,
        "code_snippet": cls.relative_code.code[:500] + "..." if hasattr(cls, 'relative_code') and cls.relative_code and len(cls.relative_code.code) > 500 else cls.relative_code.code if hasattr(cls, 'relative_code') and cls.relative_code else None
    } for cls in repository.structure.classes]
    
    with open(classes_json_path, 'w', encoding='utf-8') as f:
        json.dump(classes_data, f, ensure_ascii=False, indent=2)
    print(f"Class definition summary saved to: {classes_json_path}")
    
    # Save function definition summary
    functions_json_path = os.path.join(output_dir, f"{timestamp}_functions.json")
    functions_data = [{
        "name": func.name,
        "docstring": func.docstring,
        "is_method": func.is_method,
        "class_name": func.class_name,
        "parameters": func.parameters,
        "calls": func.calls,
        "code_location": {
            "file": func.relative_code.belongs_to.file_name if func.relative_code and func.relative_code.belongs_to else None,
            "path": func.relative_code.belongs_to.upper_path if func.relative_code and func.relative_code.belongs_to else None,
            "start_line": func.relative_code.start_line if func.relative_code else None,
            "end_line": func.relative_code.end_line if func.relative_code else None
        } if hasattr(func, 'relative_code') and func.relative_code else None,
        "code_snippet": func.relative_code.code[:500] + "..." if hasattr(func, 'relative_code') and func.relative_code and len(func.relative_code.code) > 500 else func.relative_code.code if hasattr(func, 'relative_code') and func.relative_code else None
    } for func in repository.structure.functions]
    
    with open(functions_json_path, 'w', encoding='utf-8') as f:
        json.dump(functions_data, f, ensure_ascii=False, indent=2)
    print(f"Function definition summary saved to: {functions_json_path}")
    
    # Save code node information
    code_nodes_json_path = os.path.join(output_dir, f"{timestamp}_code_nodes.json")
    code_nodes = []
    
    # Extract code nodes from classes and functions
    for cls in repository.structure.classes:
        if hasattr(cls, 'relative_code') and cls.relative_code:
            code = cls.relative_code.code
            truncated_code = code[:32000] + "..." if len(code) > 32000 else code
            code_nodes.append({
                "type": "class",
                "name": cls.name,
                "file": cls.relative_code.belongs_to.file_name if cls.relative_code.belongs_to else None,
                "path": cls.relative_code.belongs_to.upper_path if cls.relative_code.belongs_to else None,
                "start_line": cls.relative_code.start_line,
                "end_line": cls.relative_code.end_line,
                "code": truncated_code
            })
        
        # If code was truncated, output notification
            if len(code) > 32000:
                print(f"Warning: Code for class {cls.name} in file {cls.relative_code.belongs_to.file_name} was truncated.")
        
    for func in repository.structure.functions:
        if hasattr(func, 'relative_code') and func.relative_code:
            code = func.relative_code.code
            truncated_code = code[:32000] + "..." if len(code) > 32000 else code
            code_nodes.append({
                "type": "function",
                "name": func.name,
                "class_name": func.class_name,
                "file": func.relative_code.belongs_to.file_name if func.relative_code.belongs_to else None,
                "path": func.relative_code.belongs_to.upper_path if func.relative_code.belongs_to else None,
                "start_line": func.relative_code.start_line,
                "end_line": func.relative_code.end_line,
                "code": truncated_code
            })
        
            # If code was truncated, output notification
            if len(code) > 32000:
                print(f"Warning: Code for function {func.name} in file {func.relative_code.belongs_to.file_name} was truncated.")

    with open(code_nodes_json_path, 'w', encoding='utf-8') as f:
        json.dump(code_nodes, f, ensure_ascii=False, indent=2)
    print(f"Code node information saved to: {code_nodes_json_path}")

def _extract_module_structure(modules: list):
    """Recursively extract module structure"""
    result = []
    for module in modules:
        module_data = {
            "name": module.name,
            "is_package": module.is_package,
            "files": [f.file_name for f in module.files],
            "sub_modules": _extract_module_structure(module.sub_modules)
        }
        result.append(module_data)
    return result

def print_repository_summary(repository: Repository):
    """
    Print repository structure summary
    
    Args:
        repository: Repository object
    """
    print("\n===== Repository Structure Summary =====")
    print(f"Repository name: {repository.name}")
    print(f"Repository ID: {repository.id}")
    
    # Output structure statistics
    structure = repository.structure
    print(f"\nTotal:")
    print(f"  Class definitions: {len(structure.classes)}")
    print(f"  Function definitions: {len(structure.functions)}")
    print(f"  Class attributes: {len(structure.attributes)}")
    print(f"  Code relationships: {len(structure.relationships)}")
    
    # Output module structure
    print("\nModule structure:")
    for module in structure.root_modules:
        _print_module(module, "  ")
    
    # Output main classes
    print("\nMain classes:")
    for cls in sorted(structure.classes, key=lambda c: len(c.methods) if hasattr(c, 'methods') else 0, reverse=True)[:5]:
        method_count = len(cls.methods) if hasattr(cls, 'methods') else 0
        attr_count = len(cls.attributes) if hasattr(cls, 'attributes') else 0
        print(f"  {cls.name}: {method_count} methods, {attr_count} attributes")
        if cls.docstring:
            print(f"    Documentation: {cls.docstring[:100]}..." if len(cls.docstring) > 100 else f"    Documentation: {cls.docstring}")
    
    # Output core functionality overview
    if structure.core_functionality:
        print("\nCore functionality overview:")
        print(f"  {structure.core_functionality[:500]}..." if len(structure.core_functionality) > 500 else structure.core_functionality)

def _print_module(module, indent=""):
    """Recursively print module structure"""
    print(f"{indent}Module: {module.name} ({'package' if module.is_package else 'module'})")
    
    if module.files:
        print(f"{indent}  {len(module.files)} files")
    
    for sub_module in module.sub_modules:
        _print_module(sub_module, indent + "  ")

def main():
    repos = [
    "./swebench-repos/pylint",
    "./swebench-repos/pytest",
    "./swebench-repos/requests",
    "./swebench-repos/matplotlib", 
    "./swebench-repos/sphinx",
    "./swebench-repos/sqlfluff",
    "./swebench-repos/xarray",
    "./swebench-repos/scikit-learn",
    "./swebench-repos/flask",
    "./swebench-repos/django",
    "./swebench-repos/sympy",
    "./swebench-repos/astropy",
    # Add more paths
    ]

    output_base_dir = "repo_analysis/full_code_for_embedding"
    for repo_path in repos:

        try:    
            print(f"Starting analysis: {repo_path}")
            # Extract repository name as output folder name
            repo_name = repo_path.strip("/").split("/")[-1]
            output_dir = f"{output_base_dir}/{repo_name}"
            repository = analyze_repository(repo_path=repo_path, repo_root=repo_path)
            print_repository_summary(repository)
            # Save analysis results
            save_repository_data(repository, output_dir, repo_name)
            print(f"Analysis completed: {repo_path}\n")
        except Exception as e:
            print(f"Analysis failed: {repo_path}, error: {e}")
            continue

if __name__ == "__main__":
    main() 
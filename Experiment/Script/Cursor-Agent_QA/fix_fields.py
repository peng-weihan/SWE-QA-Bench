#!/usr/bin/env python3
"""
Batch rename fields in JSONL files:
1. Rename the original 'question' field to 'origin_question'
2. Rename 'rewritten_question' field to 'question' and place it as the first field
"""

import json
import os
from pathlib import Path

def process_jsonl_file(file_path):
    """Process a single JSONL file"""
    output_lines = []
    modified = False
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                file_modified = False
                
                # Processing logic:
                # 1. If 'question' field exists and 'origin_question' does not, rename 'question' to 'origin_question'
                # 2. If 'rewritten_question' field exists, rename it to 'question' and place it as the first field
                # 3. If 'rewritten_question' does not exist but 'question' is already rewritten (and 'origin_question' exists), keep unchanged
                
                # Step 1: Process original question -> origin_question
                if 'question' in data and 'origin_question' not in data:
                    # Rename question to origin_question
                    data['origin_question'] = data.pop('question')
                    file_modified = True
                
                # Step 2: Process rewritten_question -> question (place as first field)
                if 'rewritten_question' in data:
                    rewritten_value = data.pop('rewritten_question')
                    # Create new dict with question as first field
                    new_data = {'question': rewritten_value}
                    # Add other fields
                    for key, value in data.items():
                        new_data[key] = value
                    data = new_data
                    file_modified = True
                elif 'question' not in data and 'origin_question' in data:
                    # If only origin_question exists without question, it means question has been renamed to origin_question
                    # In this case, if rewritten_question doesn't exist, we might need to get it from elsewhere
                    # But based on current file structure, this case should not occur
                    pass
                
                # Ensure question field is in the first position (if it exists)
                if 'question' in data:
                    question_value = data.pop('question')
                    new_data = {'question': question_value}
                    new_data.update(data)
                    data = new_data
                    file_modified = True
                
                if file_modified:
                    modified = True
                
                output_lines.append(json.dumps(data, ensure_ascii=False))
                
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line in {file_path}: {e}")
                output_lines.append(line)  # Keep original line
    
    # If file was modified, write back to file
    if modified:
        with open(file_path, 'w', encoding='utf-8') as f:
            for line in output_lines:
                f.write(line + '\n')
        return True
    
    return False

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Batch rename fields in JSONL files")
    parser.add_argument("--input-dir", type=str, default="./output", help="Directory containing JSONL files to process")
    args = parser.parse_args()
    
    base_dir = Path(args.input_dir)
    
    if not base_dir.exists():
        print(f"Error: Directory {base_dir} does not exist")
        return
    
    jsonl_files = list(base_dir.glob("*.jsonl"))
    
    if not jsonl_files:
        print(f"No JSONL files found in {base_dir}")
        return
    
    print(f"Found {len(jsonl_files)} JSONL files to process")
    
    for file_path in jsonl_files:
        print(f"Processing {file_path.name}...")
        try:
            modified = process_jsonl_file(file_path)
            if modified:
                print(f"  ✓ Modified {file_path.name}")
            else:
                print(f"  - No changes needed for {file_path.name}")
        except Exception as e:
            print(f"  ✗ Error processing {file_path.name}: {e}")
    
    print("\nDone!")

if __name__ == "__main__":
    main()









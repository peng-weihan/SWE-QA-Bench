# Top ten repositories
import os
import sys

from openai import OpenAI
from tqdm import tqdm
import ast
import json
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import numpy as np
from methods import CodeNode, QAPair
from methods import format_code_from_list
from voyageai import Client
from voyageai.error import InvalidRequestError
import faiss
import pickle
from dotenv import load_dotenv
# Get project root directory (SWE-QA/SWE-QA)
PROJECT_ROOT = Path(__file__).parent.parent.parent
# Load environment variables from .env file
load_dotenv()

# Configuration from environment variables
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
VOYAGE_MODEL = os.getenv("VOYAGE_MODEL", "voyage-code-3")
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "You are a professional code analysis assistant, you are good at explaining code and answering programming questions.")

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("MODEL", "DeepSeek-V3")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))

# Additional configuration (keeping hardcoded values)
EMBEDDING_BATCH_SIZE = 64
EMBEDDING_MAX_LENGTH = 32768
FAISS_NLIST = 100
FAISS_NPROBE = 10

# Validate required environment variables
if not VOYAGE_API_KEY:
    raise ValueError("VOYAGE_API_KEY environment variable is required")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")

class VoyageEmbeddingModel:
    """Voyage AI embedding model wrapper"""
    
    def __init__(self):
        self.client = Client(api_key=VOYAGE_API_KEY)
    
    def encode(self, texts, input_type="document"):
        if isinstance(texts, str):
            texts = [texts]
        try:
            # print(f"Trying to encode batch of size {len(texts)}")  # Print current batch size
            return self.client.embed(
                model=VOYAGE_MODEL,
                texts=texts,
                input_type=input_type,
                truncation=True,
            ).embeddings
        except InvalidRequestError as e:
            print(f"Batch of size {len(texts)} failed: {e}")  # Print error
            if len(texts) == 1:
                print(f"Single text failed, raising error: {texts[0]}")
                raise  # If batch size is already less than 10 we expect batchs to be abnormaly large and raise the error

            mid = len(texts) // 2
            first_half = texts[:mid]
            second_half = texts[mid:]
            embeddings_first = self.encode(first_half, input_type)
            embeddings_second = self.encode(second_half, input_type)
            return embeddings_first + embeddings_second
    
class FuncChunkRAG():
    """Use RAG technology in prompts to add relevant code content and answer user questions"""
    
    def __init__(self, save_path: str):
        self.llm_client = OpenAI(
            base_url=OPENAI_BASE_URL,
            api_key=OPENAI_API_KEY
        )
        self.save_path = save_path
        self.faiss_index_path = save_path.replace('.json', '_faiss.index')
        self.metadata_path = save_path.replace('.json', '_metadata.pkl')
        super().__init__()
        
        def read_data_from_json(filepath: str):
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
              
        # 初始化FAISS索引和元数据
        self.faiss_index = None
        self.code_metadata = []
        
        self.embed_model = VoyageEmbeddingModel()
        self._build_embeddings()
    
    def read_code_snippet(self, code_location: dict) -> str:
        """
        Read source code file content for corresponding lines based on code_location
        
        Args:
            code_location: Dictionary containing file information
                Example:
                {
                    "file": "helpers.py",
                    "path": "/data3/pwh/swebench-repos/flask/src/flask",
                    "start_line": 27,
                    "end_line": 32
                }
        
        Returns:
            str: Code snippet content
        """
        file_path = os.path.join(code_location["path"], code_location["file"])
        start = code_location["start_line"]
        end = code_location["end_line"]

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                # Note: Python lists start from 0, start_line starts from 1
                snippet = "".join(lines[start-1:end])
                return snippet
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return ""
        except Exception as e:
            print(f"Failed to read file: {e}")
            return ""
    
    def _build_embeddings(self, load_if_exists=True):
        """Build embeddings for all code elements using FAISS for storage and retrieval"""

        # Load existing FAISS index if available and allowed
        if load_if_exists and os.path.exists(self.faiss_index_path) and os.path.exists(self.metadata_path):
            print(f"Loading existing FAISS index: {self.faiss_index_path}")
            self.faiss_index = faiss.read_index(self.faiss_index_path)
            with open(self.metadata_path, 'rb') as f:
                self.code_metadata = pickle.load(f)
            print(f"Successfully loaded FAISS index with {self.faiss_index.ntotal} vectors")
            return
    
        # Prepare all texts that need to be encoded
        self.elements = []
        self.code_metadata = []  # Store code metadata
        
        for function in self.functions:
            function["type"] = "function"
            code_location = function["code_location"]
            function_full_code:str= self.read_code_snippet(code_location)
            if len(function_full_code) > EMBEDDING_MAX_LENGTH:
                print(f"function_full_code length: {len(function_full_code)}")
                function_full_code = function_full_code[:EMBEDDING_MAX_LENGTH]
            self.elements.append(function_full_code)
            # Storage append/remove 
            function["code_snippet"] = function_full_code
            function.pop("docstring", None)
            self.code_metadata.append(function)

        for class_ in self.classes:
            class_["type"] = "class"
            element = "class_name: " + class_["name"] + "\n" + "class_docstring: " + class_["docstring"]

            if len(element) > EMBEDDING_MAX_LENGTH:
                element = element[:EMBEDDING_MAX_LENGTH]
            self.elements.append(element)
            # Storage append/remove 
            class_.pop("code_snippet", None)
            self.code_metadata.append(class_)

        print(f"len(self.elements): {len(self.elements)}")
        
        # Calculate embeddings
        def encode_with_fallback(embed_model, elements, batch_size=EMBEDDING_BATCH_SIZE):
            if not elements:
                return np.array([])

            all_embeddings = []
            n_batches = int(np.ceil(len(elements) / batch_size))

            for i in tqdm(range(n_batches), desc="Embedding Batches"):
                batch = elements[i * batch_size : (i + 1) * batch_size]
                try:
                    emb = embed_model.encode(batch)
                    all_embeddings.append(emb)
                except Exception as e:
                    print(f"Batch {i+1}/{n_batches} failed, batch_size={len(batch)}, error: {e}")
                    raise e

            return np.vstack(all_embeddings)
                
        if self.elements:
            embeddings = encode_with_fallback(self.embed_model, self.elements)
        else:
            embeddings = np.array([])

        # Create FAISS index
        if len(embeddings) > 0:
            dimension = embeddings.shape[1]
            print(f"Embedding dimension: {dimension}")
            
            # Use IVFFlat index, suitable for medium-scale datasets
            if len(embeddings) > 1000:
                # For large datasets, use IVFFlat
                nlist = min(FAISS_NLIST, len(embeddings) // 10)  # Number of cluster centers
                quantizer = faiss.IndexFlatIP(dimension)
                self.faiss_index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
                
                # Train index
                print("Training FAISS index...")
                self.faiss_index.train(embeddings.astype('float32'))
            else:
                # For small datasets, use Flat index
                self.faiss_index = faiss.IndexFlatIP(dimension)
            
            # Add vectors to index
            print("Adding vectors to FAISS index...")
            self.faiss_index.add(embeddings.astype('float32'))
            
            # Save FAISS index and metadata
            faiss.write_index(self.faiss_index, self.faiss_index_path)
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.code_metadata, f)
            
            print(f"Saved FAISS index to: {self.faiss_index_path}")
            print(f"Saved metadata to: {self.metadata_path}")
        else:
            print("No embedding data to save")
            
    def find_relevant_code(self, query: str, top_k: int = 10) -> List[CodeNode]:
        """
        Use FAISS to find the most relevant code elements for the query
        
        Args:
            query: Query text
            top_k: Number of most relevant elements to return
            
        Returns:
            List of CodeNode
        """
        if self.faiss_index is None or self.faiss_index.ntotal == 0:
            print("FAISS index not initialized or empty")
            return []
        
        # Calculate query embedding
        query_embedding = self.embed_model.encode(query, input_type="query")[0] # Voyage AI returns list, take first element
        query_embedding = np.array(query_embedding).reshape(1, -1).astype('float32')
        
        # Use FAISS for similarity search
        try:
            # For IVF index, need to set nprobe parameter
            if hasattr(self.faiss_index, 'nprobe'):
                self.faiss_index.nprobe = min(FAISS_NPROBE, self.faiss_index.nlist)
            
            # Execute search
            print(self.faiss_index.d)
            similarities, indices = self.faiss_index.search(query_embedding, top_k)
            
            results = []
            for i, idx in enumerate(indices[0]):
                if idx == -1:  # FAISS returns -1 when no result found
                    continue
                element = self.code_metadata[idx]
                similarity = similarities[0][i]
                # Read relevant code
                file_path = os.path.join(element["code_location"]["path"], element["code_location"]["file"])
                results.append(element)
                print(f"Found relevant code, similarity: {similarity:.4f}, file: {file_path}, type: {element['type']}, name: {element['name']}")
            return results
            
        except Exception as e:
            print(f"FAISS search error: {str(e)}")
            return []
        
    def process_qa_pair(self, qa_pair: QAPair) -> QAPair:
        """
        Process a QA pair by finding relevant code and using LLM to answer the question
        
        Args:
            qa_pair: QAPair object containing question and answer
            
        Returns:
            Updated QAPair object with new answer
        """
        if not qa_pair.relative_code_list:
            relevant_code_list = self.find_relevant_code(qa_pair.question)
        else:
            relevant_code_list = qa_pair.relative_code_list
        answer = self.process_answer(qa_pair.question, relevant_code_list)
        qa_pair.answer = answer
        return qa_pair
    
    def process_answer(self, question: str, relevant_code_list: List[CodeNode]) -> str:
        """
        Process a question by generating an answer using LLM with relevant code context
        
        Args:
            question: The question to be answered
            relevant_code_list: List of relevant code elements
            
        Returns:
            Generated answer string
        """
        if not relevant_code_list:
            # If no relevant code found, return default message
            return "No relevant code found. No sufficient information to answer the question."
        # 2. Build prompt for LLM
        prompt = self._build_llm_prompt(question, relevant_code_list)
        # 3. Call LLM to get answer
        answer = self.llm_client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=TEMPERATURE
        )
        return answer.choices[0].message.content

    def _build_llm_prompt(self, question: str, relevant_code_list: List[CodeNode]) -> str:
        """
        Build the prompt to be submitted to the LLM
        
        Args:
            question: User's question
            relevant_code_list: List of relevant code elements
            
        Returns:
            Complete prompt text
        """
        prompt = "You are a professional code analysis assistant. Please answer the question based on the following code snippets.\n\n"
        prompt += f"Question: {question}\n\n"
        prompt += format_code_from_list(relevant_code_list)
        return prompt
    
    def process_qa_pairs(self, qa_pairs: List[QAPair]) -> List[QAPair]:
        """
        Process multiple QA pairs in batch
        
        Args:
            qa_pairs: List of QAPair objects
            
        Returns:
            Updated list of QAPair objects
        """
        updated_pairs = []
        for qa_pair in qa_pairs:
            updated_pair = self.process_qa_pair(qa_pair)
            updated_pairs.append(updated_pair)
        return updated_pairs
     
    def get_index_stats(self) -> Dict:
        """
        Get FAISS index statistics
        
        Returns:
            Dictionary containing index statistics
        """
        if self.faiss_index is None:
            return {"error": "FAISS index not initialized"}
        
        stats = {
            "total_vectors": self.faiss_index.ntotal,
            "dimension": self.faiss_index.d,
            "is_trained": self.faiss_index.is_trained if hasattr(self.faiss_index, 'is_trained') else True
        }
        
        if hasattr(self.faiss_index, 'nlist'):
            stats["nlist"] = self.faiss_index.nlist
        
        return stats
    
    def clear_index(self) -> None:
        """
        Clear the FAISS index and remove associated files
        """
        if self.faiss_index is not None:
            self.faiss_index.reset()
            self.code_metadata = []
            
            # Delete index files
            if os.path.exists(self.faiss_index_path):
                os.remove(self.faiss_index_path)
            if os.path.exists(self.metadata_path):
                os.remove(self.metadata_path)
            
            print("FAISS index cleared") 


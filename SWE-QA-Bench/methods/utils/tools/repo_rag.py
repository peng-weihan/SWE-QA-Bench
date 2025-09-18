import json
import sys
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import requests
import os
import numpy as np
from voyageai import Client
from dotenv import load_dotenv
import faiss
import pickle
from voyageai.error import InvalidRequestError
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
load_dotenv()

SYSTEM_PROMPT = "You are a professional code analysis assistant, you are good at explaining code and answering programming questions."

class VoyageEmbeddingModel:
    """Voyage AI embedding model wrapper"""
    
    def __init__(self, api_key: str = None):
        self.client = Client(api_key=os.getenv("VOYAGE_API_KEY"))
    
    def encode(self, texts, input_type="document"):
        if isinstance(texts, str):
            texts = [texts]
        try:
            # print(f"Trying to encode batch of size {len(texts)}")  # Print current batch size
            return self.client.embed(
                model=os.getenv("VOYAGE_MODEL"),
                texts=texts,
                input_type=input_type,
                truncation=True,
            ).embeddings
        except InvalidRequestError as e:
            print(f"Batch of size {len(texts)} failed: {e}")  # Print error
            if len(texts) == 1:
                print(f"Single text failed, raising error: {texts[0]}")
                raise  # If batch size is already less than 10 we expect batches to be abnormally large and raise the error

            mid = len(texts) // 2
            first_half = texts[:mid]
            second_half = texts[mid:]
            embeddings_first = self.encode(first_half, input_type)
            embeddings_second = self.encode(second_half, input_type)
            return embeddings_first + embeddings_second
    
    def get_dimension(self) -> int:
        """
        Get embedding dimension
        
        Returns:
            embedding dimension
        """
        return self.dimension

class FuncChunkRAG():
    """Use RAG technology in prompts to add relevant code content and answer user questions"""
    
    def __init__(self,save_path: str):
        self.save_path = save_path
        self.faiss_index_path = save_path.replace('.json', '_faiss.index')
        self.metadata_path = save_path.replace('.json', '_metadata.pkl')
        super().__init__()
        
        def read_data_from_json(filepath: str):
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        
        # Initialize FAISS index and metadata
        self.faiss_index = None
        self.code_metadata = []
 
        self.embed_model = VoyageEmbeddingModel()
        self._build_embeddings()
   
    def read_code_snippet(self, code_location: dict) -> str:
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
        """Build embeddings for all code elements, using FAISS for storage and retrieval"""

        # If FAISS index file already exists and loading is allowed
        if load_if_exists and os.path.exists(self.faiss_index_path) and os.path.exists(self.metadata_path):
            print(f"Loading existing FAISS index: {self.faiss_index_path}")
            self.faiss_index = faiss.read_index(self.faiss_index_path)
            with open(self.metadata_path, 'rb') as f:
                self.code_metadata = pickle.load(f)
            print(f"Successfully loaded FAISS index with {self.faiss_index.ntotal} vectors")
            return
    
    def find_relevant_code(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Use FAISS to find code elements most relevant to the query
        
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
                self.faiss_index.nprobe = min(10, self.faiss_index.nlist)
            
            # Execute search
            print(self.faiss_index.d)
            similarities, indices = self.faiss_index.search(query_embedding, top_k)
            
            results = []
            for i, idx in enumerate(indices[0]):
                if idx == -1:  # FAISS returns -1 indicating no result found
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
        
def repo_search_rag(query: str, repo_name: str) -> str:
    """
    RAG tool
    
    Args:
        query: Query question
        repo_name: Repository name
        
    Returns:
        Raw search result string
    """
    try:
        rag = FuncChunkRAG(save_path=f"/data2/raymone/voyage_faiss_func_chunk/{repo_name}_embeddings.json")
        # rag = FuncChunkRAG(save_path=f"{PROJECT_ROOT}/datasets/faiss/func_chunk/{repo_name}_embeddings.json")
        results = rag.find_relevant_code(query, top_k=5)
        if not results:
            return f"No code found related to the question '{query}'."
        answer: str = "Find the following classes or functions that are related to the query: \n"
        for result in results:
            answer += str(result)
            answer += "\n"
        print(answer)
        return answer
    except Exception as e:
        return f"RAG search error: {str(e)}"
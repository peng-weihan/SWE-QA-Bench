from asyncio import as_completed
from concurrent.futures import ThreadPoolExecutor
import os
import json
from typing import List, Dict
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
import faiss
import pickle
from voyageai import Client
from voyageai.error import InvalidRequestError
from openai import OpenAI
from methods import CodeNode, QAPair, format_code_from_list
load_dotenv()

SYSTEM_PROMPT = "You are a professional code analysis assistant, you are good at explaining code and answering programming questions."

VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
VOYAGE_MODEL = os.getenv("VOYAGE_MODEL", "voyage-code-3")

OPENAI_URL = os.getenv("OPENAI_URL")
OPENAI_KEY = os.getenv("OPENAI_KEY")

MODEL = os.getenv("MODEL", "gpt-4o")
TEMPERATURE = int(os.getenv("TEMPERATURE", "0"))

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
                raise  # If batch size is already less than 10 we expect batches to be abnormally large and raise the error

            mid = len(texts) // 2
            first_half = texts[:mid]
            second_half = texts[mid:]
            embeddings_first = self.encode(first_half, input_type)
            embeddings_second = self.encode(second_half, input_type)
            return embeddings_first + embeddings_second
  
class RAGSlidingWindowsCodeQA():
    """Use RAG technology in prompts to add relevant code content and answer user questions"""
    
    def __init__(self, save_path: str):
        self.llm_client = OpenAI(
            base_url= OPENAI_URL,
            api_key= OPENAI_KEY
        )
        self.save_path = save_path
        self.faiss_index_path = save_path.replace('.json', '_faiss.index')
        self.metadata_path = save_path.replace('.json', '_metadata.pkl')
        super().__init__()
        
        def read_code_nodes_from_jsonl(filepath: str):
            data = []
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line))
            return data
        
        # Initialize FAISS index and metadata
        self.faiss_index = None
        self.code_metadata = []

        self.embed_model = VoyageEmbeddingModel()
        self._build_embeddings()
        
    def _build_embeddings(self, load_if_exists=True):
        """Build embeddings for all code elements, use FAISS for storage and retrieval"""

        # If FAISS index file already exists and loading is allowed
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
        
        # Calculate embeddings
        # def encode_with_fallback(embed_model, elements, batch_sizes=[32, 24, 16, 8, 4]):
        #     if not elements:
        #         return np.array([])
            
        #     for batch_size in batch_sizes:
        #         try:
        #             embeddings = embed_model.encode(elements, batch_size=batch_size, show_progress_bar=True)
        #             print(f"Successfully used batch_size={batch_size}")
        #             return embeddings
        #         except Exception as e:
        #             print(f"batch_size={batch_size} failed, preparing to retry with smaller batch, error: {e}")

        #     raise RuntimeError("All batch_size attempts failed, unable to encode.")
        def encode_with_fallback(embed_model, elements, batch_size=32):
            if not elements:
                return np.array([])
            
            all_embeddings = []
            batches = [elements[i:i+batch_size] for i in range(0, len(elements), batch_size)]

            # with ThreadPoolExecutor(max_workers=1) as executor:
            #     futures = {executor.submit(embed_model.encode, batch, input_type="document"): batch for batch in batches}

            #     with tqdm(total=len(elements), desc="Encoding all elements") as pbar:
            #         for future in as_completed(futures):
            #             batch_embeddings = future.result()
            #             all_embeddings.extend(batch_embeddings)
            #             for _ in batch_embeddings:
            #                 pbar.update(1)

            # return np.array(all_embeddings)
        
            with tqdm(total=len(elements), desc="Encoding all elements") as pbar:
                for i in range(0, len(elements), batch_size):
                    batch = elements[i:i+batch_size]
                    embeddings = embed_model.encode(batch, input_type="document")
                    all_embeddings.extend(embeddings)
                    pbar.update(len(batch))  # Update global progress bar

            return np.array(all_embeddings)
        
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
                nlist = min(100, len(embeddings) // 10)  # Number of cluster centers
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
            
    def find_relevant_code(self, query: str, top_k: int = 5) -> List[CodeNode]:
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
        query_embedding = self.embed_model.encode(query, input_type="query")[0]  # Voyage AI returns list, take first element
        query_embedding = np.array(query_embedding).reshape(1, -1).astype('float32')
        
        # Use FAISS for similarity search
        try:
            # For IVF index, need to set nprobe parameter
            if hasattr(self.faiss_index, 'nprobe'):
                self.faiss_index.nprobe = min(10, self.faiss_index.nlist)
            
            # Execute search
            similarities, indices = self.faiss_index.search(query_embedding, top_k)
            
            results = []
            for i, idx in enumerate(indices[0]):
                if idx == -1:  # FAISS returns -1 when no result found
                    continue
                    
                element = self.code_metadata[idx]
                similarity = similarities[0][i]
                
                # Read relevant code
                try:

                    # Create file node
                    file_node = {
                        "file_name": element["metadata"]["filename"].split("/")[-1],
                        "upper_path":  "/".join(element["metadata"]["filename"].split("/")[:-1]),
                        "module": "",
                        "define_class": [],
                        "imports": []
                    }
                    
                    # Create code node
                    code_node = CodeNode(
                        start_line=element["metadata"]["start"],
                        end_line=element["metadata"]["end"],
                        belongs_to=file_node,
                        relative_function=[],
                        code=element["text"]
                    )
                    
                    results.append(code_node)
                    print(f"Found relevant code, similarity: {similarity:.4f}, file: {element['metadata']['filename']}")
                    
                except Exception as e:
                    print(f"Warning: Error occurred: {str(e)}")
                    continue
                    
            return results
            
        except Exception as e:
            print(f"FAISS search error: {str(e)}")
            return []
    
    def make_question_prompt(self, question: str) -> str:
        """
        Assemble question with code to pass to LLM
        
        Args:
            question: User's question
            
        Returns:
            Answer content
        """
        # Find relevant code elements
        relevant_code = self.find_relevant_code(question)
        
        if not relevant_code:
            return "Sorry, no relevant code information found."
            
        # Build answer
        answer = "Based on code analysis, here is the relevant information:\n\n"
        
        for code_content, similarity, element_type in relevant_code:
            answer += f"Relevance: {similarity:.2f}\n"
            answer += f"Type: {element_type}\n"
            answer += "Relevant code:\n```python\n"
            answer += code_content
            answer += "\n```\n\n"
            
        return answer
    
    def process_qa_pair(self, qa_pair: QAPair) -> QAPair:
        """
        Process QA Pair, find relevant code and use LLM to answer questions
        
        Args:
            qa_pair: QA Pair object containing question and answer
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
        Process QA Pair, find relevant code and use LLM to answer questions
        
        Args:
            qa_pair: QA Pair object containing question and answer
            
        Returns:
            Updated QA Pair object containing new answer
        """

        if not relevant_code_list:
            return "No relevant code found. No sufficient information to answer the question."
        prompt = self._build_llm_prompt(question, relevant_code_list)

        try:
            response = self.llm_client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=TEMPERATURE
            )
            answer = response.choices[0].message.content
            return answer
        except Exception as e:
            print(f"LLM call failed: {str(e)}")
            print(f"Model used: {MODEL}")
            print(f"URL used: {OPENAI_URL}")
            print(f"API Key used: {OPENAI_KEY[:10]}...")
            print(f"Prompt length: {len(prompt)}")
            raise e

    def _build_llm_prompt(self, question: str, relevant_code_list: List[CodeNode]) -> str:
        """
        Build prompt to submit to LLM
        
        Args:
            question: User's question
            relevant_code_list: List of relevant code, each element is a tuple of (code content, similarity, element type)
            
        Returns:
            Complete prompt text
        """
        prompt = "You are a professional code analysis assistant. Please answer the question based on the following code snippets.\n\n"
        prompt += f"Question: {question}\n\n"
        prompt += format_code_from_list(relevant_code_list)
        return prompt
    
    def process_qa_pairs(self, qa_pairs: List[QAPair]) -> List[QAPair]:
        """
        Batch process multiple QA Pairs
        
        Args:
            qa_pairs: List of QA Pairs
            
        Returns:
            Updated list of QA Pairs
        """
        updated_pairs = []
        for qa_pair in qa_pairs:
            updated_pair = self.process_qa_pair(qa_pair)
            updated_pairs.append(updated_pair)
        return updated_pairs
    
    def add_code_to_index(self, code_nodes: List[Dict]) -> None:
        """
        Add new code snippets to FAISS index
        
        Args:
            code_nodes: List of new code nodes
        """
        if self.faiss_index is None:
            print("FAISS index not initialized")
            return
        
        # Prepare new code text
        new_elements = []
        new_metadata = []
        
        for code_node in code_nodes:
            new_elements.append(code_node["text"][:4096])
            new_metadata.append(code_node)
        
        if not new_elements:
            return
        
        # Calculate embeddings for new code
        new_embeddings = self.embed_model.encode(new_elements)
        
        # Add to FAISS index
        self.faiss_index.add(new_embeddings.astype('float32'))
        
        # Update metadata
        self.code_metadata.extend(new_metadata)
        
        # Save updated index and metadata
        faiss.write_index(self.faiss_index, self.faiss_index_path)
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.code_metadata, f)
        
        print(f"Successfully added {len(new_elements)} code snippets to FAISS index")
    
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
        Clear FAISS index
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


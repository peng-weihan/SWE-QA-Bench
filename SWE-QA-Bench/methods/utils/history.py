from typing import List, Dict, Any
from config import Config
from langchain_core.messages import AnyMessage


class ConversationHistory:
    """Conversation history management class"""

    def __init__(self, max_history: int | None = None):
        """
        Initialize history manager

        Args:
            max_history: Maximum number of history rounds to keep, defaults to config value
        """
        self.max_history = max_history or Config.HISTORY_WINDOW
        self.history: List[
            List[AnyMessage]
        ] = []  # Each item is a list of messages from one interaction
        self.rag_cache: Dict[int, str] = {}  # RAG query cache, key is query hash, value is result

    def add_interaction(self, messages: List[AnyMessage]):
        """
        Add one round of interaction to history

        Args:
            messages: List of messages from the interaction
        """

        self.history.append(messages)

        # Keep history length within maximum limit
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history :]

    def flatten(self):
        return [message for interaction in self.history for message in interaction]

    def add_rag_result(self, query: str, result: str):
        """
        Add RAG query result to cache
        
        Args:
            query: Query string
            result: Query result
        """
        query_hash = hash(query)
        self.rag_cache[query_hash] = result

    def get_rag_result(self, query: str) -> str | None:
        """
        Get RAG query result
        
        Args:
            query: Query string
            
        Returns:
            Result string if found for the same query, otherwise None
        """
        query_hash = hash(query)
        return self.rag_cache.get(query_hash)

    def clear_history(self):
        """Clear history records"""
        self.history.clear()
        self.rag_cache.clear()

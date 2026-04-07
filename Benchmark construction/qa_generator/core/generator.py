
import ast
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import numpy as np
import os
from openai import OpenAI
import dotenv
import logging

dotenv.load_dotenv()

class BaseGenerator:
    """Use RAG technology in prompts to add relevant code content and answer user questions"""
    
    def __init__(self, baseurl: str = None, apikey: str = None):
        self.llm_client = OpenAI(
            base_url="",
            api_key=""
        )

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """
        Call LLM to get response
        
        Args:
            system_prompt: System prompt text submitted to LLM
            user_prompt: User prompt text submitted to LLM
            
        Returns:
            LLM response
        """
        if not self.llm_client:
            return "Unable to call LLM, please ensure LLM client is properly configured."
        
        try:
            # Call LLM
            response = self.llm_client.chat.completions.create(
                model="",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                stream=False,
                temperature=0.5
            )
            # Extract response
            return response.choices[0].message.content.strip("```json").strip("```")
        except Exception as e:
            return f"Error occurred when calling LLM: {str(e)}"
    

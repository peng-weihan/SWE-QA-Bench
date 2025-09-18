import os
from dotenv import load_dotenv

load_dotenv(override=True)  # priority: .env > global


class Config:
    """Project configuration class"""
    # OpenAI configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")
    OPENAI_TEMPERATURE = os.getenv("OPENAI_TEMPERATURE", 0)
    
    # Agent configuration
    MAX_ITERATIONS = 5 # Reduce maximum iterations for efficiency-first strategy
    HISTORY_WINDOW = 5  # Keep last 5 rounds of interactions

    # Tool configuration
    ALLOWED_COMMANDS = ["tree", "ls", "grep", "cat"]
    SEARCH_RESULTS_LIMIT = 20
    GREP_WINDOW_SIZE = 100  # Code window size when using grep search

    # File configuration
    PROMPTS_DIR = "prompts"
    TOOLS_DIR = "tools"

    @classmethod
    def validate(cls):
        """Validate that necessary configurations exist"""
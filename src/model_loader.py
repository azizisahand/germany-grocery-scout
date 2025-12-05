"""
Model Loader Module.

This module is responsible for initializing and returning instances of the 
Language Model (LLM) and the Embedding Model. It handles environment variable 
validation and model-specific configuration.
"""

import os
import logging
from typing import Optional

from dotenv import load_dotenv
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from src.config import (
    EMBEDDING_MODEL_NAME,
    EMBEDDING_CACHE_PATH,
    LLM_MODEL,
    # Add other config variables here if you decide to use them (e.g., TEMPERATURE)
)

# Load environment variables once at module level
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


<<<<<<< HEAD
def initialise_llm() -> Gemini:
=======
def initialise_llm() -> GoogleGenAI:
>>>>>>> parent of 15bc4e3 (stramlit error)
    """
    Initialises the Google Gemini LLM using credentials from the environment.

    Returns:
        GoogleGenAI: An instance of the Google Gemini model.

    Raises:
        ValueError: If GOOGLE_API_KEY is missing from environment variables.
    """
    api_key: Optional[str] = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        logger.error("Attempted to initialize LLM without an API Key.")
        raise ValueError(
            "GOOGLE_API_KEY not found. Please ensure it is set in your .env file."
        )

    logger.info(f"Initializing LLM: {LLM_MODEL}")
    
    # Note: If you want to configure temperature or tokens, 
    # add them to src/config.py and pass them here.
    return GoogleGenAI(
        model=LLM_MODEL,
        api_key=api_key
    )


def get_embedding_model() -> HuggingFaceEmbedding:
    """
    Initialises and returns the HuggingFace embedding model.
    
    This function ensures the local cache directory exists before attempting
    to load the model. This prevents downloading the model on every run.

    Returns:
        HuggingFaceEmbedding: The loaded embedding model.
    """
    # Ensure cache directory exists
    if not EMBEDDING_CACHE_PATH.exists():
        logger.info(f"Creating embedding cache directory at: {EMBEDDING_CACHE_PATH}")
        EMBEDDING_CACHE_PATH.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading Embedding Model: {EMBEDDING_MODEL_NAME}...")
    
    return HuggingFaceEmbedding(
        model_name=EMBEDDING_MODEL_NAME,
        cache_folder=EMBEDDING_CACHE_PATH.as_posix()
    )
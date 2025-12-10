from pathlib import Path

"""
Configuration Management for the Prospekt RAG Agent.

This file centralizes all model parameters, prompt engineering rules, 
and file path definitions. It is tuned specifically for parsing 
unstructured, multi-column grocery brochures (Prospekte).
"""

# --- LLM Model Configuration ---
# We use Google's Gemini Flash model for its speed and cost-efficiency.
LLM_MODEL: str = "gemini-2.5-flash"

# Increased limit (1024) to allow the AI to generate long lists of products 
# when comparing prices across multiple stores.
LLM_MAX_NEW_TOKENS: int = 1024

# Temperature set to 0.0 to enforce strict factual extraction.
# We do not want the AI to be "creative" with prices.
LLM_TEMPERATURE: float = 0.0

LLM_TOP_P: float = 0.95
LLM_REPETITION_PENALTY: float = 1.05

# The System Prompt defines the AI's persona and error-correction logic.
# It includes specific instructions to handle OCR artifacts common in 
# German brochures (e.g., "169" instead of "1.69").
LLM_SYSTEM_PROMPT: str = (
    "You are a savvy German shopping assistant reading grocery brochures (Prospekte). "
    "Rules for answering:"
    "1. PRICE INTERPRETATION: If you see a price like '169' or '129' in a table context, "
    "   it almost always means '1.69 €' or '1.29 €'. Assume the last two digits are cents."
    "2. SEARCH BROADLY: If the user asks for 'Butter', look for 'Markenbutter', 'Streichfett', or specific brands like 'Kerrygold'."
    "3. NO HALLUCINATION: If a price column says 'AKTION' but has no number, say 'Price not listed'."
    "4. ALWAYS list the Store Name found in the context."
)


# --- Embedding Model Configuration ---
# We use a multilingual model because brochures often contain mixed German/English terms.
# 'paraphrase-multilingual-MiniLM-L12-v2' performs significantly better on German text
# than the standard English-only MiniLM models.
EMBEDDING_MODEL_NAME: str = "paraphrase-multilingual-MiniLM-L12-v2"


# --- RAG / VectorStore Configuration ---

# Number of text chunks to retrieve from the database per query.
# Set to 8 (higher than default) to ensure we capture a wide variety of offers.
SIMILARITY_TOP_K: int = 50

# Size of each text chunk in tokens.
# 512 is chosen to be large enough to contain a whole "Product Box" (Name + Description + Price).
CHUNK_SIZE: int = 512

# Overlap between chunks.
# Critical for brochures: Prevents splitting a product name from its price 
# if they happen to fall on the boundary of two chunks.
CHUNK_OVERLAP: int = 50


# --- Chat Memory Configuration ---
# Limit the conversation history to prevent hitting the LLM context window limit.
CHAT_MEMORY_TOKEN_LIMIT: int = 3900


# --- Persistent Storage Paths ---
# Uses pathlib to automatically handle OS-specific separators (Windows/Linux/Mac).
ROOT_PATH: Path = Path(__file__).parent.parent
DATA_PATH: Path = ROOT_PATH / "data/"
EMBEDDING_CACHE_PATH: Path = ROOT_PATH / "local_storage/embedding_model/"
VECTOR_STORE_PATH: Path = ROOT_PATH / "local_storage/vector_store/"
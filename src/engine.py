"""
Core RAG Engine Module.

This module handles the creation and management of the Vector Store (indexing)
and the Chat Engine (retrieval and generation). It supports a hybrid parsing strategy:
1. LlamaParse: For high-fidelity extraction (tables, layout) using an LLM.
2. Standard: Fallback using basic PDF text extraction.

It also handles 'Metadata Injection', ensuring filenames (Store Names) are 
embedded directly into the text chunks to improve retrieval accuracy.
"""

import os
import logging
from typing import List

from llama_index.core import (
    StorageContext,
    SimpleDirectoryReader,
    VectorStoreIndex,
    load_index_from_storage,
    Document,
    Settings
)
from llama_index.core.chat_engine.types import BaseChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.node_parser import SentenceSplitter, MarkdownNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.gemini import Gemini
from llama_parse import LlamaParse

# Internal imports
from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_PATH,
    LLM_SYSTEM_PROMPT,
    SIMILARITY_TOP_K,
    VECTOR_STORE_PATH,
    CHAT_MEMORY_TOKEN_LIMIT,
)
from src.model_loader import (
    get_embedding_model,
    initialise_llm
)

# --- Configuration ---
# Toggle this to False if you want to save API credits and use standard parsing.
USE_LLAMAPARSE = True

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _create_new_vector_store(embed_model: HuggingFaceEmbedding) -> VectorStoreIndex:
    """
    Creates a fresh Vector Store from documents in the DATA_PATH.

    This function handles:
    1. Parsing (LlamaParse for Markdown tables OR Standard PDF text).
    2. Metadata Injection (inserting Store Names into the text body).
    3. Indexing (embedding the text and saving to disk).

    Args:
        embed_model (HuggingFaceEmbedding): The loaded embedding model instance.

    Returns:
        VectorStoreIndex: The fully built and indexed vector store.

    Raises:
        ValueError: If no documents are found in the data directory.
    """
    logger.info("Creating new vector store from 'data' directory...")

    # ---------------------------------------------------------
    # 1. LOAD DOCUMENTS (Strategy Pattern)
    # ---------------------------------------------------------
    documents: List[Document] = []

    # Check if LlamaParse is enabled and the API key is present
    if USE_LLAMAPARSE and os.getenv("LLAMA_CLOUD_API_KEY"):
        logger.info("🚀 Strategy: LlamaParse (High-Quality Brochure Extraction)")
        
        # Specific instructions help the LLM understand the grid layout of brochures
        parsing_instructions = """
        The provided document is a grocery store brochure (Prospekt) with a grid layout.
        1. Extract all products and their prices into Markdown tables.
        2. IMPORTANT: Prices often lack a decimal point (e.g., "169" means "1.69"). 
           Always format prices with a decimal point.
        3. Add the Store Name to the beginning of every table caption.
        """
        
        parser = LlamaParse(
            result_type="markdown", 
            verbose=True,
            language="de",
            parsing_instruction=parsing_instructions
        )
        
        file_extractor = {".pdf": parser}
        documents = SimpleDirectoryReader(
            input_dir=DATA_PATH, 
            file_extractor=file_extractor
        ).load_data()
        
    else:
        logger.warning("⚠️ Strategy: Standard PDF Reader (Layouts might be messy)")
        documents = SimpleDirectoryReader(input_dir=DATA_PATH).load_data()

    if not documents:
        error_msg = f"No documents found in {DATA_PATH}."
        logger.error(error_msg)
        raise ValueError(error_msg)

    # ---------------------------------------------------------
    # 2. METADATA INJECTION
    # ---------------------------------------------------------
    logger.info(f"💉 Injecting Store Names into {len(documents)} document chunks...")
    
    # We create a new list because modifying 'documents' in-place can be buggy 
    # if the reader returns generator-like objects.
    augmented_documents = []
    
    for doc in documents:
        # Extract filename to determine store (e.g., "aldi.pdf" -> "ALDI")
        filename = doc.metadata.get("file_name", "Unknown")
        store_name = filename.replace(".pdf", "").replace(".md", "").upper()
        
        # Prepend the Store Name to the text content.
        # This is crucial for RAG: When the vector search looks for "Milk",
        # it will now find "ALDI... Milk", associating the product with the store.
        original_text = doc.get_content()
        new_content = f"🛒 STORE OFFER FROM: {store_name}\n\n{original_text}"
        
        # Create a FRESH Document object to ensure clean state
        new_doc = Document(
            text=new_content, 
            metadata=doc.metadata 
        )
        
        # Explicitly allow the LLM and Embedder to see this metadata if needed elsewhere
        new_doc.metadata["store_name"] = store_name
        new_doc.excluded_llm_metadata_keys = [] 
        new_doc.excluded_embed_metadata_keys = []
        
        augmented_documents.append(new_doc)

    # Swap references
    documents = augmented_documents

    # ---------------------------------------------------------
    # 3. CREATE NODES & INDEX
    # ---------------------------------------------------------
    if USE_LLAMAPARSE:
        logger.info("✨ Processing: Using MarkdownNodeParser")
        # Since LlamaParse returns Markdown, we use a parser that respects headers and tables.
        # This prevents splitting a price table in the middle.
        node_parser = MarkdownNodeParser()
        nodes = node_parser.get_nodes_from_documents(documents)
        
        index = VectorStoreIndex(
            nodes=nodes,
            embed_model=embed_model
        )
    else:
        logger.info("Processing: Using Standard SentenceSplitter")
        # Standard overlapping window splitter for raw text
        text_splitter = SentenceSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        index = VectorStoreIndex.from_documents(
            documents,
            transformations=[text_splitter],
            embed_model=embed_model
        )

    # ---------------------------------------------------------
    # 4. PERSISTENCE
    # ---------------------------------------------------------
    logger.info(f"Persisting vector store to: {VECTOR_STORE_PATH}")
    index.storage_context.persist(persist_dir=VECTOR_STORE_PATH.as_posix())
    
    return index


def get_vector_store(embed_model: HuggingFaceEmbedding) -> VectorStoreIndex:
    """
    Retrieves the Vector Store. 
    
    Logic:
    1. Checks if the storage directory exists and is populated.
    2. If yes, loads the index from disk (Fast).
    3. If no (or if loading fails), calls `_create_new_vector_store` (Slow).

    Args:
        embed_model (HuggingFaceEmbedding): The embedding model.

    Returns:
        VectorStoreIndex: The loaded or created index.
    """
    VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)

    # Check if directory contains files
    if any(VECTOR_STORE_PATH.iterdir()):
        logger.info("Loading existing vector store from disk...")
        try:
            storage_context = StorageContext.from_defaults(
                persist_dir=VECTOR_STORE_PATH.as_posix()
            )
            return load_index_from_storage(
                storage_context,
                embed_model=embed_model
            )
        except Exception as e:
            # Fallback mechanism: If index is corrupted, rebuild it.
            logger.error(f"Error loading existing index: {e}. Rebuilding...")
            return _create_new_vector_store(embed_model)
    else:
        return _create_new_vector_store(embed_model)


def get_chat_engine(
        llm: Gemini,
        embed_model: HuggingFaceEmbedding
) -> BaseChatEngine:
    """
    Constructs the Chat Engine with memory and context.

    Args:
        llm (Gemini): The Large Language Model.
        embed_model (HuggingFaceEmbedding): The Embedding Model.

    Returns:
        BaseChatEngine: An interactive chat engine ready for the loop.
    """
    vector_index = get_vector_store(embed_model)
    
    # Configure conversation memory (rolling window buffer)
    memory = ChatMemoryBuffer.from_defaults(
        token_limit=CHAT_MEMORY_TOKEN_LIMIT
    )

    # Initialize the engine
    # chat_mode="context": Retrieves relevant nodes and inserts them into system prompt.
    # This is preferred over "condense_question" for strict fact-based QA.
    chat_engine = vector_index.as_chat_engine(
        chat_mode="context", 
        memory=memory,
        llm=llm,
        system_prompt=LLM_SYSTEM_PROMPT,
        similarity_top_k=SIMILARITY_TOP_K,
    )
    return chat_engine


def main_chat_loop() -> None:
    """
    Entry point for running the chat loop directly from this file.
    """
    print("--- Initialising models... ---")
    llm = initialise_llm()
    embed_model = get_embedding_model()

    chat_engine = get_chat_engine(
        llm=llm,
        embed_model=embed_model
    )
    print("--- RAG Chatbot Initialised. Type 'exit' to quit. ---")
    
    # Start the interactive REPL provided by LlamaIndex
    chat_engine.chat_repl()


if __name__ == "__main__":
    main_chat_loop()
"""
Debug Viewer for Vector Store.

This utility script allows developers to inspect the contents of the persisted 
Vector Store without running the full RAG pipeline. It is useful for verifying:
1. That documents were actually loaded and indexed.
2. That chunking is working as expected (checking chunk sizes).
3. That metadata (like store names) is correctly attached to the nodes.

Usage:
    Run this script directly from the terminal:
    python debug_view.py
"""

import logging
from typing import List

from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.schema import BaseNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Import config to ensure we use the exact same settings as the main app
from src.config import (
    VECTOR_STORE_PATH, 
    EMBEDDING_CACHE_PATH, 
    EMBEDDING_MODEL_NAME
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def view_chunks() -> None:
    """
    Loads the persistent vector store and prints a sample of the indexed chunks.
    
    This function:
    1. Initializes the embedding model (required to load the index).
    2. Reconstructs the index from the storage directory.
    3. Retrieves all nodes from the document store.
    4. Prints metadata and content for the first 3 nodes.
    """
    logger.info(f"📂 Loading vector store from: {VECTOR_STORE_PATH}")
    
    # 1. Load Embedding Model
    # We must load the SAME embedding model used for creation to successfully load the index.
    embed_model = HuggingFaceEmbedding(
        model_name=EMBEDDING_MODEL_NAME,
        cache_folder=EMBEDDING_CACHE_PATH.as_posix()
    )
    
    # 2. Load Index
    try:
        storage_context = StorageContext.from_defaults(
            persist_dir=VECTOR_STORE_PATH.as_posix()
        )
        index = load_index_from_storage(
            storage_context, 
            embed_model=embed_model
        )
    except FileNotFoundError:
        logger.error(f"Could not find vector store at {VECTOR_STORE_PATH}. Have you run engine.py?")
        return
    except Exception as e:
        logger.error(f"Failed to load index: {e}")
        return
    
    # 3. Peek at the documents
    # The docstore contains all the chunks (nodes) referenced by the index.
    all_nodes: List[BaseNode] = list(index.docstore.docs.values())
    
    logger.info(f"✅ Total Chunks found: {len(all_nodes)}")
    print("-" * 50)
    
    # 4. Print the first 3 chunks to check quality
    if not all_nodes:
        logger.warning("Index is empty! No chunks to display.")
        return

    sample_size = min(3, len(all_nodes))
    for i, node in enumerate(all_nodes[:sample_size]):
        # Extract metadata safely
        file_name = node.metadata.get('file_name', 'Unknown File')
        store_name = node.metadata.get('store_name', 'Unknown Store')
        
        print(f"\n📄 CHUNK {i+1}")
        print(f"   Source File: {file_name}")
        print(f"   Store Name:  {store_name}")
        print(f"   Node ID:     {node.node_id}")
        print("   Preview:")
        print(f"   {node.get_content()[:500]}...") # Print first 500 chars to avoid flooding terminal
        print("-" * 50)


if __name__ == "__main__":
    view_chunks()
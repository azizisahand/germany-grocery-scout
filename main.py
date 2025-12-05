"""
Main Application Entry Point.

This module serves as the Command Line Interface (CLI) entry point for the 
Prospekt AI application. It initializes the environment and triggers the 
interactive chat loop defined in the engine.

Usage:
    Run directly from the terminal:
    python main.py
"""

import sys
import logging

# Internal imports
from src.engine import main_chat_loop

# Configure logging to ensure we see startup messages from imported modules
# We set the format to show timestamps, which is helpful for debugging delays.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main() -> None:
    """
    Main execution function.

    This function:
    1. Prints a startup banner.
    2. Delegates control to the core chat loop in src.engine.
    3. Handles clean exits (KeyboardInterrupt).
    """
    print("--- 🤖 Main Application Starting ---")
    logger.info("Initializing application components...")

    try:
        # Start the main interactive chat session.
        # This function blocks until the user types 'exit' or the process is killed.
        main_chat_loop()
        
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully so it doesn't print a messy traceback
        print("\n\n--- 🛑 Application stopped by user. Goodbye! ---")
        sys.exit(0)
    except Exception as e:
        # Catch unexpected crashes and log them before exiting
        logger.critical(f"Application crashed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
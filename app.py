"""
Prospekt AI - Main Streamlit Application.

This module serves as the frontend entry point for the German Grocery Scout application.
It provides a Streamlit-based user interface that allows users to:
1. Input their Google Gemini API Key securely.
2. Interact with the RAG (Retrieval-Augmented Generation) Chat Engine.
3. View the AI's responses in real-time (streaming).
4. Inspect the source documents (brochure chunks) used to generate answers.

The UI features a custom starfield background animation implemented via raw CSS/HTML injection.
"""

import os
import random
import time
import logging
from typing import List, Dict, Any

import streamlit as st
from llama_index.core.chat_engine.types import StreamingAgentChatResponse

# Internal imports
from src.engine import get_chat_engine
from src.model_loader import get_embedding_model, initialise_llm

# --- CONFIGURATION ---
PAGE_TITLE = "🛒 Prospekt AI"
PAGE_ICON = "🛒"
LAYOUT = "centered"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- PAGE SETUP ---
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=LAYOUT
)


def generate_star_shadows(n: int, max_x: int = 2000, max_y: int = 2000) -> str:
    """
    Generates a CSS box-shadow string to simulate stars.
    
    Args:
        n (int): Number of stars to generate.
        max_x (int): Maximum X coordinate.
        max_y (int): Maximum Y coordinate.

    Returns:
        str: A long string of comma-separated box-shadow values (x y color).
    """
    shadows = []
    for _ in range(n):
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)
        shadows.append(f"{x}px {y}px #FFF")
    return ", ".join(shadows)


def inject_custom_css() -> None:
    """
    Injects custom CSS to create the animated starfield background and 
    style the chat interface transparency.
    """
    shadows_small = generate_star_shadows(700)
    shadows_medium = generate_star_shadows(200)
    shadows_big = generate_star_shadows(100)

    st.markdown(
        f"""
        <style>
        /* 1. RESET & BASIC SETUP */
        * {{
            box-sizing: border-box;
        }}
        
        /* 2. MAIN CONTAINER TRANSPARENCY */
        /* Make Streamlit containers transparent to reveal the stars */
        .stApp, 
        [data-testid="stAppViewContainer"],
        [data-testid="stHeader"],
        [data-testid="stToolbar"],
        .main, 
        .block-container {{
            background: transparent !important;
            background-color: transparent !important;
        }}

        /* 3. STARFIELD WRAPPER */
        /* Fixed position background that sits behind all content */
        .star-bg-container {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            z-index: -1;
            overflow: hidden;
            background: radial-gradient(ellipse at bottom, #1B2735 0%, #090A0F 100%);
            pointer-events: none;
        }}

        /* 4. STAR ANIMATIONS */
        #stars {{
            width: 1px;
            height: 1px;
            background: transparent;
            box-shadow: {shadows_small};
            animation: animStar 50s linear infinite;
        }}
        #stars:after {{
            content: " ";
            position: absolute;
            top: 2000px;
            width: 1px;
            height: 1px;
            background: transparent;
            box-shadow: {shadows_small};
        }}

        #stars2 {{
            width: 2px;
            height: 2px;
            background: transparent;
            box-shadow: {shadows_medium};
            animation: animStar 100s linear infinite;
        }}
        #stars2:after {{
            content: " ";
            position: absolute;
            top: 2000px;
            width: 2px;
            height: 2px;
            background: transparent;
            box-shadow: {shadows_medium};
        }}

        #stars3 {{
            width: 3px;
            height: 3px;
            background: transparent;
            box-shadow: {shadows_big};
            animation: animStar 150s linear infinite;
        }}
        #stars3:after {{
            content: " ";
            position: absolute;
            top: 2000px;
            width: 3px;
            height: 3px;
            background: transparent;
            box-shadow: {shadows_big};
        }}

        @keyframes animStar {{
            from {{ transform: translateY(0px); }}
            to {{ transform: translateY(-2000px); }}
        }}

        /* 5. TEXT & UI COLOR FIXES */
        /* Ensure text is white against the dark space background */
        h1, h2, h3, p, .stMarkdown, div, span, label {{
            color: #f0f0f0 !important;
        }}
        
        /* Style Inputs */
        .stTextInput > div > div > input {{
            background-color: rgba(255, 255, 255, 0.1);
            color: white;
            border: 1px solid rgba(255,255,255,0.2);
        }}
        
        /* Style Chat Bubbles */
        .stChatMessage {{
            background-color: rgba(0,0,0,0.3);
            border-radius: 10px;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    # Inject the HTML divs that hold the stars
    st.markdown(
        """
        <div class="star-bg-container">
            <div id='stars'></div>
            <div id='stars2'></div>
            <div id='stars3'></div>
        </div>
        """,
        unsafe_allow_html=True
    )


@st.cache_resource(show_spinner="Loading AI Models & Index...")
def load_rag_engine():
    """
    Initializes the RAG engine.
    Cached to prevent reloading the heavy models on every user interaction.
    """
    # Note: os.environ["GOOGLE_API_KEY"] must be set before calling this
    llm = initialise_llm()
    embed_model = get_embedding_model()
    return get_chat_engine(llm=llm, embed_model=embed_model)


def main():
    """Main application loop."""
    
    # 1. Setup UI
    inject_custom_css()
    st.title("🛒 German Grocery Scout")
    st.markdown("Enter your API Key to start chatting with your *Prospekte*.")

    # 2. Authentication (API Key Input)
    # We use a password input so the key isn't visible on screen
    user_api_key = st.text_input("🔑 Enter Google Gemini API Key:", type="password")

    if not user_api_key:
        st.warning("Please enter your API Key to continue.")
        st.stop()  # Halt execution until key is provided

    # Set the key in environment for the model loader to find
    os.environ["GOOGLE_API_KEY"] = user_api_key

    # 3. Load Logic
    try:
        chat_engine = load_rag_engine()
    except Exception as e:
        logger.error(f"Failed to load engine: {e}")
        st.error(f"Error loading models. Please check your API Key.\nDetails: {e}")
        st.stop()

    # 4. Chat State Management
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hallo! Ich habe die Prospekte gelesen. Was suchst du heute?"}
        ]

    # 5. Display Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 6. Handle New User Input
    if prompt := st.chat_input("Ex: Wo gibt es günstig Butter?"):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate Assistant Response
        with st.chat_message("assistant"):
            with st.spinner("Searching stars and stores..."):
                # Use streaming for better UX
                streaming_response: StreamingAgentChatResponse = chat_engine.stream_chat(prompt)
                
                response_container = st.empty()
                full_response = ""
                
                # Stream tokens
                for token in streaming_response.response_gen:
                    full_response += token
                    response_container.markdown(full_response + "▌")
                
                # Finalize response
                response_container.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})

                # 7. Source Inspector (Transparency)
                # Show which documents were used to generate the answer
                with st.expander("🔍 View Sources"):
                    if hasattr(streaming_response, 'source_nodes'):
                        for node in streaming_response.source_nodes:
                            store = node.metadata.get("store_name", "Unknown Store")
                            score = f"{node.score:.2f}" if node.score else "N/A"
                            
                            st.markdown(f"**🏪 {store}** (Relevance: {score})")
                            # Show first 300 chars of the source text
                            st.caption(node.get_content()[:300] + "...")
                            st.divider()


if __name__ == "__main__":
    main()
# 🛒 German Grocery Scout

**German Grocery Scout** is a Retrieval-Augmented Generation (RAG) tool that helps you find the best deals in German supermarket brochures (*Prospekte*).

Meet **AngeBOT** 🤖 (from *Angebot* meaning "offer"), your AI shopping assistant. You can ask AngeBOT questions like *"Wo gibt es diese Woche günstig Butter?"* or *"Welche Angebote hat Lidl?"*, and it will scan the provided PDF brochures to give you accurate answers with source references.

## 🔒 Privacy & Security Notice

**This repository does NOT contain any API keys.** We value privacy and security. To use AngeBOT, you must provide your own API keys.

- **Web App:** You will be prompted to enter your **Gemini API Key** securely in the browser sidebar. It is *never* stored, logged, or saved to our servers.
- **Local Parsing:** If you run the ingestion pipeline locally, keys are loaded from a `.env` file which is excluded from version control via `.gitignore`.

## 🔑 API Requirements

This project uses two distinct APIs. You will need to obtain keys for them if you want to use the full functionality:

1. **Google Gemini API (Required):**
   - Used for the Chat Intelligence (LLM) and Reasoning.
   - [Get a free Gemini API key here](https://aistudio.google.com/app/apikey).

2. **LlamaCloud API (Optional):**
   - Used for **Advanced Parsing** (extracting tables and complex layouts from PDFs).
   - If you skip this, the system falls back to a standard PDF reader (which is less accurate for grid layouts).
   - [Get a LlamaCloud key here](https://cloud.llamaindex.ai/).

## 📂 Project Structure

```
├── data/                  # Place your PDF brochures here (e.g., aldi.pdf, lidl.pdf)
├── local_storage/         # (Generated) Stores the searchable Vector Index
├── src/
│   ├── config.py          # Settings (Chunk sizes, model names)
│   ├── engine.py          # Core Logic: LlamaParse indexing & Metadata injection
│   ├── model_loader.py    # Loads Gemini LLM & HuggingFace Embeddings
│   └── __init__.py
├── app.py                 # Streamlit Web Interface (for End Users)
├── main.py                # CLI Terminal App (for Developers)
├── debug_view.py          # Debug tool to inspect extracted chunks
├── environment.yml        # Dependencies list
├── .env                   # (User Created) Secure storage for local keys
└── README.md              # Documentation
```

## 🚀 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/german-grocery-scout.git
cd german-grocery-scout
```

### 2. Install Dependencies

We recommend using Conda to manage environments:

```bash
conda env create -f environment.yml
conda activate german-grocery-scout
```

### 3. Configure Local Environment

Create a file named `.env` in the root folder to store your LlamaCloud key (for parsing) and your Gemini key (for local testing).

**File: `.env`**

```env
# Required for Local CLI usage.


# Optional: Only needed if you want high-quality table extraction
LLAMA_CLOUD_API_KEY=your_llama_key_here
```

## 🖥️ Usage

### Option A: The Web Interface (Streamlit)

This is the standard way to use **AngeBOT** with the visual interface and starfield background.

1. Run the app:
   ```bash
   streamlit run app.py
   ```

2. Your browser will open `http://localhost:8501`.

3. Enter your **Google Gemini API Key** in the sidebar password field.

4. Start chatting!

### Option B: Local Terminal (CLI)

For developers who want to test the backend logic without the web UI.

1. Ensure your `.env` file is set up.

2. Run the script:
   ```bash
   python main.py
   ```

3. AngeBOT will start in your terminal window.

### Option C: Debugging & Data Inspection

Want to see how the AI breaks down a PDF? Use the debug viewer to print the first few "chunks" of text and verify that the store names are being correctly detected.

```bash
python debug_view.py
```

## 📊 How It Works (The Pipeline)

1. **Ingestion:** You drop PDFs into the `data/` folder.

2. **Preprocessing & Metadata:**
   - The system scans filenames (e.g., `aldi.pdf`) and injects the store name directly into the text chunks (e.g., *"🛒 STORE OFFER FROM: ALDI..."*).
   - This helps the AI distinguish between offers from different stores.

3. **Parsing (LlamaParse):**
   - If enabled, LlamaParse converts the PDF images/grids into Markdown tables, ensuring prices like "1.99" stay associated with the correct product "Butter".

4. **Indexing:** The processed text is converted into numbers (vectors) and saved in `local_storage/`.

5. **Retrieval:** When you ask a question, AngeBOT finds the most relevant chunks and uses Gemini to formulate a helpful answer.

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **Orchestration:** LlamaIndex
- **LLM:** Google Gemini Flash
- **Embeddings:** HuggingFace (all-MiniLM-L6-v2)
- **Parser:** LlamaParse (Multimodal)

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contact

For any questions or feedback, feel free to reach out:

<p align="center">
  <a href="https://www.linkedin.com/in/sahandazizi/">
    <img 
      src="https://user-images.githubusercontent.com/74038190/235294012-0a55e343-37ad-4b0f-924f-c8431d9d2483.gif" 
      alt="LinkedIn"
      width="60"
    >
  </a>
  &nbsp;&nbsp;

  <a href="https://github.com/azizisahand">
    <img 
      src="https://user-images.githubusercontent.com/74038190/212257468-1e9a91f1-b626-4baa-b15d-5c385dfa7ed2.gif"
      alt="GitHub"
      width="50"
    >
  </a>
</p>

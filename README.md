# Agent-4-HelpSites
This project implements an AI-powered agent capable of crawling help documentation websites, processing their content, and answering natural language questions based on the information found.
## ⚙️ Setup Instructions

1. **Clone the repo**  
   ```bash
   git clone git@github.com:BUVANESHWARAN-B/Agent-4-HelpSites
   cd Agent-4-HelpSites

2. **Install Dependencies**
   pip install -r requirements.txt
   Install Playwright Browsers:
    Crawl4AI uses Playwright for browser automation. Install the necessary browser binaries:
    ```bash
    playwright install
    ```

3.Configure API key

Create a file named .env in the project root:
In the File 
GOOGLE_API_KEY=your_google_api_key_here

🚀Usage Examples 
python qa_agent.py https://help.instagram.com
Question:About AI on Instagram

Answer:
AIs on Instagram can answer questions, help you be more productive, and keep you entertained. With AIs, you can:

*   Start conversations with an AI
*   Use Meta AI across Instagram
*   Write or use voice with Meta AI
*   Edit your image or generate an image in chats with Meta AI

[Learn about how Meta uses information for generative AI models](https://www.facebook.com/privacy/genai).


[https://help.instagram.com/828618801985471/?helpref=topq](https://help.instagram.com/828618801985471/?helpref=topq)

Sources:
https://help.instagram.com/
https://help.instagram.com/1417489251945243/?helpref=hc_global_nav
https://help.instagram.com/478880589321969/?helpref=topq
https://help.instagram.com/828618801985471/?helpref=topq

Type quit or exit to end the session.


🏗Design Decisions

1.Crawler

Uses crawl4ai for URL discovery, rate-limiting, basic Markdown conversion.

Custom URL filtering ensures we stay on the target domain.

2.Parsing & Chunking

Markdown-header splitter preserves logical sections; fallback to recursive splitter for flat content.

Token-aware splits (via character counts) with overlap to respect LLM context limits.

3.Indexing

HuggingFaceEmbeddings + FAISS for dense vector search.

BM25Retriever and EnsembleRetriever for hybrid sparse + dense retrieval.

4.RAG Loop

Top-k retrieval → assemble prompt with inline citations → ChatGoogleGenerativeAI generates the final answer.

“No-info” behavior can be customized via distance thresholds before LLM call.






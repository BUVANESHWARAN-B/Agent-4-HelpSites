import asyncio
import sys # Import sys to read terminal arguments
import os # Import os to access environment variables
from urllib.parse import urldefrag, urlparse

# --- Import dotenv to load environment variables from a .env file ---
# You will need to install this library: pip install python-dotenv
try:
    from dotenv import load_dotenv
    load_dotenv() # Load variables from .env file if it exists
    print(".env file loaded.")
except ImportError:
    print("python-dotenv not installed. Skipping .env file loading.")
    print("Please install it (`pip install python-dotenv`) to use .env files.")
except Exception as e:
    print(f"Error loading .env file: {e}")


from crawl4ai import (
    AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode,
    MemoryAdaptiveDispatcher,
    PruningContentFilter # Import a content filter if needed
)

# Import necessary libraries for text splitting, embedding, vector database, LLM, and hybrid retrieval
# You will need to install these libraries:
# pip install langchain-text-splitters langchain-community google-generativeai faiss-cpu langchain-google-genai rank_bm25
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document # Import Document for splitter output
from langchain_google_genai import ChatGoogleGenerativeAI # Import the Google Generative AI LLM
from langchain_core.prompts import PromptTemplate # For creating the prompt for the LLM
from langchain_core.runnables import RunnablePassthrough, RunnableParallel # For building the RAG chain
from langchain_core.output_parsers import StrOutputParser # For parsing the LLM output

# Imports for Hybrid Retrieval
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever



# --- Load the Google API Key ---
# The load_dotenv() call above will load the key from a .env file
# if it exists. We then read it from the environment variables.
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize the embedding model (using embed-text-v1.5)
# Ensure you have the GOOGLE_API_KEY environment variable set
# os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY" # Uncomment and replace with your key or set as env var
if not GOOGLE_API_KEY:
    print("Error: GOOGLE_API_KEY environment variable not set.")
    print("Please set it or create a .env file with GOOGLE_API_KEY=YOUR_API_KEY")
    embedding_model = None
else:
    try:
        # Pass the API key explicitly or rely on the environment variable
        embedding_model = HuggingFaceEmbeddings(
         model_name="intfloat/e5-small-v2",
         model_kwargs={"trust_remote_code": True}
        ) # embed-text-v1.5 is mapped to embedding-001
        print("Embedding model initialized successfully.")
    except Exception as e:
        print(f"Error initializing embedding model: {e}")
        embedding_model = None # Set to None if initialization fails


# Initialize the Google LLM (Using gemini-1.5-flash-latest as a potential Flash 2.0 equivalent)
# Ensure you have the GOOGLE_API_KEY environment variable set
# Note: Model availability can vary. If gemini-1.5-flash-latest doesn't work, try "gemini-1.0-pro"
if not GOOGLE_API_KEY:
     llm = None # LLM cannot be initialized without the API key
else:
    try:
        # Pass the API key explicitly or rely on the environment variable
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3, google_api_key=GOOGLE_API_KEY) # Adjust temperature as needed
        print("LLM (gemini-1.5-flash-latest) initialized successfully.")
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        print("Could not initialize gemini-1.5-flash-latest. Please check model availability or try 'gemini-1.0-pro'.")
        llm = None # Set to None if initialization fails


# Initialize a global FAISS index variable
# This will be built in memory for each script execution
vector_db = None
# Also need a global variable to store the raw chunks for BM25
all_chunks_for_bm25 = []


async def process_and_store_content(url: str, markdown_content: str, base_url: str):
    """
    Processes the extracted markdown content by chunking it, generating embeddings,
    and storing them in a FAISS index in memory. Also stores chunks for BM25.
    """
    print(f"Processing content for: {url}")

    if not markdown_content:
        print("  - No markdown content to process.")
        return

    # --- Chunking ---
    # Use a text splitter to break the markdown content into smaller chunks.
    # Splitting by headers is often effective for documentation to maintain context.
    # Define the headers to split by. Adjust these based on typical documentation structure.
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
        ("#####", "Header 5"),
        ("######", "Header 6"),
    ]

    # Use MarkdownHeaderTextSplitter
    try:
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        # The splitter expects a single string, not a Document object directly
        chunks = markdown_splitter.split_text(markdown_content)
        # Add metadata to the generated chunks
        for chunk in chunks:
            chunk.metadata["source_url"] = url
            chunk.metadata["base_url"] = base_url
        print(f"  - Split using MarkdownHeaderTextSplitter.")
    except Exception as e:
        print(f"  - MarkdownHeaderTextSplitter failed or not suitable: {e}. Falling back to RecursiveCharacterTextSplitter.")
        # Fallback to RecursiveCharacterTextSplitter if header splitting fails or is not desired
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, # Adjust chunk size as needed
            chunk_overlap=200 # Adjust overlap as needed
        )
        # Split the original markdown string and add metadata
        chunks = text_splitter.create_documents([markdown_content], metadatas=[{"source_url": url, "base_url": base_url}])
        print(f"  - Split using RecursiveCharacterTextSplitter.")


    print(f"  - Original Markdown length: {len(markdown_content)} chars")
    print(f"  - Number of chunks generated: {len(chunks)}")

    # --- Embedding and Storage ---
    # Generate embeddings for each chunk and add to the FAISS index in memory.
    global vector_db # Use the global vector_db variable
    global all_chunks_for_bm25 # Use the global chunks variable for BM25

    if embedding_model is None:
        print("  - Embedding model not initialized. Skipping embedding and storage.")
        return

    if not chunks:
        print("  - No chunks to embed and store.")
        return

    try:
        # Add chunks to the FAISS index
        if vector_db is None:
            # If the index doesn't exist (first content processed in this run), create it
            print(f"  - Creating new FAISS index in memory.")
            vector_db = FAISS.from_documents(chunks, embedding_model)
        else:
            # If the index exists in memory, add the new chunks to it
            print("  - Adding chunks to existing FAISS index in memory.")
            vector_db.add_documents(chunks)

        # Store chunks for later use with BM25
        all_chunks_for_bm25.extend(chunks)

        print(f"  - Successfully processed and prepared {len(chunks)} chunks for storage.")


    except Exception as e:
        print(f"  - Error during embedding or FAISS storage: {e}")


async def crawl_recursive_batch(start_urls, max_depth=3, max_concurrent=10):
    """
    Recursively crawls a list of starting URLs up to a specified depth,
    processes the content, and builds the FAISS index and collects chunks for BM25 in memory.
    """
    # Configure the browser (headless is good for performance)
    browser_config = BrowserConfig(headless=True, verbose=False)

    # Configure the crawler run
    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS, # Bypass cache to get fresh content
        stream=False, # Process results after each batch
        # You can configure a content filter here to remove potentially irrelevant
        # blocks based on heuristics *before* markdown generation.
        # Example: Remove blocks with less than 50 words.
        # markdown_generator=DefaultMarkdownGenerator(content_filter=PruningContentFilter(threshold=50, threshold_type='fixed'))
    )

    # Configure the dispatcher for managing concurrency and resources
    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=70.0,     # Don't exceed 70% memory usage
        check_interval=1.0,                # Check memory every second
        max_session_permit=max_concurrent  # Max parallel browser sessions
    )

    # Track visited URLs to prevent revisiting and infinite loops (ignoring fragments)
    visited = set()
    def normalize_url(url):
        # Remove fragment (part after #) and ensure consistent structure
        parsed_url = urlparse(url)
        # Consider normalizing scheme (http/https) and trailing slashes if needed
        normalized = parsed_url._replace(fragment="").geturl()
        # You might want to remove trailing slash unless it's the root
        if normalized.endswith('/') and len(parsed_url.path) > 1:
             normalized = normalized[:-1]
        return normalized

    # Determine the base domain(s) to restrict internal links
    base_domains = {urlparse(u).netloc for u in start_urls}
    def is_internal(url):
        try:
            # Also check if the URL starts with one of the base URLs path
            # This helps restrict crawling to subdirectories of the starting URLs
            parsed_url = urlparse(url)
            if parsed_url.netloc not in base_domains:
                return False
            # Check if the path starts with any of the base URL paths
            for base_url in start_urls:
                base_path = urlparse(base_url).path.rstrip('/')
                if parsed_url.path.rstrip('/').startswith(base_path):
                    return True
            return False

        except Exception as e:
            print(f"Error parsing URL {url}: {e}")
            return False # Handle potential URL parsing errors


    current_urls = set([normalize_url(u) for u in start_urls])

    async with AsyncWebCrawler(config=browser_config) as crawler:
        for depth in range(max_depth):
            print(f"\n=== Crawling Depth {depth+1} ===")

            # Only crawl URLs we haven't seen yet (ignoring fragments)
            urls_to_crawl = [url for url in current_urls if url not in visited]

            if not urls_to_crawl:
                print(f"No new URLs to crawl at depth {depth+1}. Stopping.")
                break

            # Add URLs to visited set *before* crawling to prevent duplicates in the current batch
            # This prevents adding the same URL multiple times if found via different paths
            for url in urls_to_crawl:
                 visited.add(url)

            print(f"Crawling {len(urls_to_crawl)} URLs...")

            # Batch-crawl all URLs at this depth in parallel
            results = await crawler.arun_many(
                urls=urls_to_crawl,
                config=run_config,
                dispatcher=dispatcher
            )

            next_level_urls = set()

            for result in results:
                norm_url = normalize_url(result.url)

                if result.success:
                    print(f"[OK] {result.url}")
                    # Process and store the content in the in-memory index and collect chunks
                    # Pass the first URL from start_urls as the base source identifier
                    await process_and_store_content(result.url, result.markdown, start_urls[0])

                    # Collect all new internal links for the next depth
                    if result.links: # Ensure links attribute exists
                        # Filter internal links based on the base domain and path
                        internal_links = [normalize_url(link["href"]) for link in result.links.get("internal", []) if is_internal(link["href"])]
                        for next_url in internal_links:
                            # Only add internal links that haven't been visited
                            if next_url not in visited:
                                next_level_urls.add(next_url)
                else:
                    print(f"[ERROR] {result.url}: {result.error_message}")

            # Move to the next set of URLs for the next recursion depth
            current_urls = next_level_urls

    # The FAISS index is built in memory for this run, no need to save it
    print("\nCrawling and indexing complete. FAISS index built in memory.")


async def answer_question(query: str, vector_db: FAISS):
    """
    Answers a user question using a hybrid retriever (BM25 + FAISS) and an LLM.
    """
    global all_chunks_for_bm25 # Access the global list of chunks

    if vector_db is None or not all_chunks_for_bm25:
        return "The knowledge base has not been built yet for this website. Please ensure the crawl completed successfully."

    if llm is None:
        return "The language model is not initialized. Cannot answer questions."

    print(f"\nSearching for relevant documents for query: '{query}' using hybrid retrieval.")

    try:
        # 1. Create the retrievers
        # Dense retriever (FAISS)
        faiss_retriever = vector_db.as_retriever(search_kwargs={"k": 5}) # Adjust k as needed

        # Sparse retriever (BM25) - requires the raw document chunks
        bm25_retriever = BM25Retriever.from_documents(all_chunks_for_bm25)
        bm25_retriever.k = 5 # Adjust k as needed for BM25

        # Create the ensemble retriever
        # Adjust weights based on desired emphasis (e.g., 0.3 for BM25, 0.7 for FAISS)
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever],
            weights=[0.3, 0.7]
        )

        # 2. Retrieve relevant documents using the ensemble retriever
        retrieved_docs = ensemble_retriever.invoke(query)

        print(f"Found {len(retrieved_docs)} relevant documents.")

        # 3. Prepare the prompt for the LLM
        # Define the prompt template
        template = """You are an AI assistant specialized in answering questions based on provided documentation.
        Use the following pieces of retrieved context to answer the question.
        If you don't know the answer, simply state that you cannot find the information in the documentation.
        Include the source URL(s) of the documents you used to formulate the answer at the end of the response.

        Context:
        {context}

        Question:
        {question}

        Answer:
        """
        prompt = PromptTemplate.from_template(template)

        # 4. Create the RAG chain
        # This chain retrieves documents, formats the context, and passes it to the LLM
        # Use the ensemble_retriever here
        rag_chain = (
            {"context": ensemble_retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser() # Parse the output to a string
        )

        # 5. Invoke the chain to get the answer
        answer = rag_chain.invoke(query)

        # 6. Extract and format source references
        # Get unique source URLs from the retrieved documents' metadata
        source_urls = set()
        if retrieved_docs:
            for doc in retrieved_docs:
                if 'source_url' in doc.metadata:
                    source_urls.add(doc.metadata['source_url'])

        sources_text = "\n\nSources:\n" + "\n".join(sorted(list(source_urls))) if source_urls else ""

        return answer + sources_text

    except Exception as e:
        print(f"Error during question answering: {e}")
        return "An error occurred while trying to answer your question."


# Main execution block
if __name__ == "__main__":
    # Check if a URL was provided as a command-line argument
    if len(sys.argv) < 2:
        print("Usage: python your_script_name.py <help_website_url>")
        sys.exit(1) # Exit if no URL is provided

    start_url = sys.argv[1] # Get the URL from the command-line argument
    start_urls = [start_url]

    # Define default crawling parameters
    max_depth = 2 # Adjust depth based on the size of the site
    max_concurrent = 5 # Adjust concurrency based on your system resources

    # --- Perform the crawl and build the index and collect chunks in memory ---
    print(f"Starting processing for: {start_url}")
    print(f"Max depth: {max_depth}, Max concurrent sessions: {max_concurrent}")

    # Ensure embedding_model is initialized before crawling and indexing
    if embedding_model:
         # Run the crawling and indexing process
         asyncio.run(crawl_recursive_batch(start_urls, max_depth=max_depth, max_concurrent=max_concurrent))

         # The vector_db and all_chunks_for_bm25 are now populated in memory
         loaded_vector_db = vector_db # Use the in-memory index for querying

    else:
         print("Embedding model not initialized. Cannot perform crawl and indexing.")
         loaded_vector_db = None # Cannot answer questions without indexing


    # --- Start the question answering loop ---
    # Use the in-memory loaded_vector_db and the collected chunks
    if loaded_vector_db is not None and all_chunks_for_bm25 and llm is not None:
        print("\nKnowledge base built in memory. Ask me a question about the website content.")
        print("Type 'quit' or 'exit' to end the session.")

        while True:
            query = input("\nYour question: ")
            if query.lower() in ["quit", "exit"]:
                break

            # Answer the question using the hybrid retriever and LLM
            # Pass the loaded_vector_db (FAISS) and implicitly use all_chunks_for_bm25 (BM25)
            answer = asyncio.run(answer_question(query, loaded_vector_db))
            print("\nAnswer:")
            print(answer)
    else:
        print("\nAgent cannot answer questions due to missing knowledge base, chunks for BM25, or LLM.")

    print("\nSession ended.")


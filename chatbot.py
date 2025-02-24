import os
import re
import io
import fitz  # PyMuPDF
import chromadb
import cloudinary
import google.generativeai as genai
from PIL import Image
from imagehash import phash
from typing import List, Dict, Set, Optional
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
import cloudinary.uploader
import tiktoken
import json
from dotenv import load_dotenv
load_dotenv()


# Environment variables
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Cloudinary configuration
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-pro')

# ChromaDB multi-modal setup
CHROMA_DB_PATH = "chroma_db"
PROCESSED_FILES_PATH = os.path.join(CHROMA_DB_PATH, "processed_files.json")
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
embedding_function = OpenCLIPEmbeddingFunction()
data_loader = ImageLoader()

print('All imports done')
PROCESSED_FILES = {}
IMAGE_HASH_CACHE = {}

def load_processed_files():
    """Load the record of processed files and their image hashes"""
    global PROCESSED_FILES
    if os.path.exists(PROCESSED_FILES_PATH):
        try:
            with open(PROCESSED_FILES_PATH, 'r') as f:
                PROCESSED_FILES = json.load(f)
                print(f"Loaded {len(PROCESSED_FILES)} processed file records.")
        except Exception as e:
            print(f"Error loading processed files: {e}")
            PROCESSED_FILES = {}
    else:
        PROCESSED_FILES = {}

def save_processed_files():
    """Save the record of processed files and their image hashes"""
    try:
        os.makedirs(os.path.dirname(PROCESSED_FILES_PATH), exist_ok=True)
        with open(PROCESSED_FILES_PATH, 'w') as f:
            json.dump(PROCESSED_FILES, f)
        print(f"Saved {len(PROCESSED_FILES)} processed file records.")
    except Exception as e:
        print(f"Error saving processed files: {e}")

def get_file_hash(file_path):
    """Get a simple hash of the file to detect changes"""
    import hashlib
    with open(file_path, 'rb') as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    return file_hash

def process_pdf(pdf_path):
    """Processes PDF page by page with incremental processing support"""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at path: {pdf_path}")
    
    # Load previously processed files
    load_processed_files()
    
    # Check if file has been processed before
    file_hash = get_file_hash(pdf_path)
    file_key = os.path.basename(pdf_path)
    
    if file_key in PROCESSED_FILES and PROCESSED_FILES[file_key]["hash"] == file_hash:
        print(f"File {file_key} has already been processed and hasn't changed.")
        # Load the image hash cache
        global IMAGE_HASH_CACHE
        IMAGE_HASH_CACHE = PROCESSED_FILES[file_key].get("image_hashes", {})
        return
    
    # Initialize or get collection (without deleting existing data)
    collection = chroma_client.get_or_create_collection(
        name="multimodal_rag",
        embedding_function=embedding_function,
        data_loader=data_loader
    )
    
    # Process the PDF
    all_text_chunks = []
    all_chunk_images = []
    
    with fitz.open(pdf_path) as doc:
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            
            # Extract text from the page
            page_text = page.get_text("text").strip()
            if not page_text:
                continue  # Skip blank pages
                
            # Extract and process images from the page
            page_images = process_images_on_page(page, page_num)
            
            # Chunk text
            text_chunks = semantic_chunk(page_text)
            
            # Link images to text chunks
            chunk_images = []
            for _ in text_chunks:
                chunk_images.append([img["url"] for img in page_images])
            
            all_text_chunks.extend(text_chunks)
            all_chunk_images.extend(chunk_images)
    
    # Store content in ChromaDB
    if all_text_chunks:
        store_content_in_chroma(all_text_chunks, all_chunk_images,file_hash)
    
    # Update processed files record
    PROCESSED_FILES[file_key] = {
        "hash": file_hash,
        "last_processed": import_timestamp(),
        "image_hashes": IMAGE_HASH_CACHE
    }
    save_processed_files()

def import_timestamp():
    """Get current timestamp string"""
    from datetime import datetime
    return datetime.now().isoformat()

def process_images_on_page(page, page_num):
    """Extract and process images from a given page with duplicate detection across sessions"""
    global IMAGE_HASH_CACHE
    
    images = []
    image_list = page.get_images(full=True)
    
    for img_index, img in enumerate(image_list):
        xref = img[0]
        base_image = page.parent.extract_image(xref)
        image_bytes = base_image["image"]
        image_ext = base_image.get("ext", "png").lower()
        img_stream = io.BytesIO(image_bytes)
        
        try:
            img = Image.open(img_stream)
            img_hash = str(phash(img))
            
            # Skip duplicates - check in current session and previous sessions
            if img_hash in IMAGE_HASH_CACHE:
                images.append({"url": IMAGE_HASH_CACHE[img_hash], "page": page_num + 1})
                continue
                
        except Exception as e:
            print(f"Error processing image: {e}")
            continue
            
        # Upload to Cloudinary if not already cached
        img_stream.seek(0)  # Reset stream for upload
        try:
            upload_result = cloudinary.uploader.upload(img_stream, format=image_ext)
            image_url = upload_result["secure_url"]
            
            # Add to cache
            IMAGE_HASH_CACHE[img_hash] = image_url
            
            images.append({
                "url": image_url,
                "page": page_num + 1,
                "format": image_ext
            })
        except Exception as e:
            print(f"Cloudinary upload failed: {e}")
            
    return images

def semantic_chunk(text: str, chunk_size: int = 512, overlap: int = 50, max_chunks: int = None) -> List[str]:
    """Context-aware text chunking by paragraphs while respecting token limits."""
    enc = tiktoken.get_encoding("cl100k_base")

    # Split by paragraphs
    paragraphs = re.split(r'\n\s*\n', text.strip())

    chunks = []
    current_chunk = []
    current_length = 0

    for paragraph in paragraphs:
        paragraph_tokens = enc.encode(paragraph)
        paragraph_length = len(paragraph_tokens)

        if paragraph_length > chunk_size:
            # If a paragraph is too large, split it into sentences
            sentences = re.split(r'(?<=[.!?])\s+|(?<=\n)\s*', paragraph)
            for sentence in sentences:
                sentence_tokens = enc.encode(sentence)
                sentence_length = len(sentence_tokens)

                if current_length + sentence_length > chunk_size:
                    chunks.append(enc.decode(current_chunk))
                    if max_chunks and len(chunks) >= max_chunks:
                        return chunks  # Stop if max_chunks is reached
                    current_chunk = current_chunk[-overlap:] if overlap else []
                    current_length = len(current_chunk)

                current_chunk.extend(sentence_tokens)
                current_length += sentence_length
        else:
            if current_length + paragraph_length > chunk_size:
                chunks.append(enc.decode(current_chunk))
                if max_chunks and len(chunks) >= max_chunks:
                    return chunks
                current_chunk = current_chunk[-overlap:] if overlap else []
                current_length = len(current_chunk)

            current_chunk.extend(paragraph_tokens)
            current_length += paragraph_length

    if current_chunk:
        chunks.append(enc.decode(current_chunk))

    return chunks
def store_content_in_chroma(text_chunks: List[str], chunk_images: List[List[str]], file_hash: str):
    """Store content in ChromaDB with guaranteed unique IDs"""
    collection = chroma_client.get_or_create_collection(
        name="multimodal_rag",
        embedding_function=embedding_function,
        data_loader=data_loader
    )

    # 1. Delete existing entries from this PDF using source tracking
    collection.delete(where={"source": file_hash})

    # 2. Generate unique deterministic IDs using file hash + content hash
    text_ids = [
        f"text_{file_hash}_{hash(chunk) & 0xFFFFFFFF}"
        for chunk in text_chunks
    ]

    # 3. Create metadata with proper typing
    metadatas = [{
        "images": ",".join(images) if images else "",
        "source": str(file_hash)
    } for images in chunk_images]

    # 4. Upsert with collision-resistant IDs
    collection.upsert(
        ids=text_ids,
        documents=text_chunks,
        metadatas=metadatas
    )

    # 5. Process images with content-based IDs
    unique_image_urls = []
    seen_urls = set()
    for images in chunk_images:
        for url in images:
            if url not in seen_urls:
                seen_urls.add(url)
                unique_image_urls.append(url)

    if unique_image_urls:
        image_embeddings = embedding_function(unique_image_urls)
        image_ids = [f"img_{file_hash}_{hash(url) & 0xFFFFFFFF}" for url in unique_image_urls]
        image_metadatas = [{"source": str(file_hash)} for _ in unique_image_urls]
        
        collection.upsert(
            ids=image_ids,
            uris=unique_image_urls,
            embeddings=image_embeddings,
            metadatas=image_metadatas
        )
        
def query_rag(query: str, n_results: int = 5) -> List[Dict]:
    """Enhanced multi-modal RAG query with linked images"""
    try:
        collection = chroma_client.get_collection(
            name="multimodal_rag",
            embedding_function=embedding_function,
            data_loader=data_loader
        )
        
        # Query text first
        text_results = collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        combined_results = []
        
        # Process text results and their linked images
        if "documents" in text_results and text_results["documents"]:
            for i, doc in enumerate(text_results["documents"][0]):
                result = {
                    "type": "text",
                    "content": doc,
                    "distance": text_results["distances"][0][i] if "distances" in text_results else None
                }
                
                # Add linked images if available
                if "metadatas" in text_results and text_results["metadatas"][0]:
                    metadata = text_results["metadatas"][0][i]
                    if metadata and "images" in metadata and metadata["images"]:
                        result["images"] = metadata["images"].split(",") if metadata["images"] else []
                
                combined_results.append(result)
        
        # Query for images directly too
        image_results = collection.query(
            query_texts=[query],
            n_results=3,  # Fewer image results to balance with text
            include=["uris", "distances"]  # Changed from "urls" to "uris"
        )
        
        # Add image results that weren't already included
        if "uris" in image_results and image_results["uris"]:  # Changed from "urls" to "uris"
            for i, uri in enumerate(image_results["uris"][0]):  # Changed from "urls" to "uris"
                # Check if this image is already included in a text result
                already_included = False
                for result in combined_results:
                    if "images" in result and uri in result["images"]:  # Changed 'url' to 'uri'
                        already_included = True
                        break
                
                if not already_included:
                    combined_results.append({
                        "type": "image",
                        "content": uri,  # Changed from 'url' to 'uri'
                        "distance": image_results["distances"][0][i] if "distances" in image_results else None
                    })
        
        # Sort by relevance (distance)
        combined_results.sort(key=lambda x: x["distance"] if x["distance"] is not None else float('inf'))
        
        return combined_results
    except ValueError as e:
        print(f"Error in query_rag: {e}")  # Added error logging
        return []
def generate_response(query: str) -> str:
    """Generate response using Gemini with multi-modal context"""
    results = query_rag(query)
    
    if not results:
        return "I couldn't find relevant information in the documents."
    
    context_parts = []
    image_urls = []
    
    for item in results:
        if item["type"] == "text":
            context_parts.append(f"Text excerpt: {item['content']}")
            # Collect linked images
            if "images" in item and item["images"]:
                image_urls.extend(item["images"])
        elif item["type"] == "image":
            image_urls.append(item["content"])
    
    # Remove duplicate images while preserving order
    unique_image_urls = []
    seen = set()
    for url in image_urls:
        if url not in seen:
            seen.add(url)
            unique_image_urls.append(url)
    
    # Add image references to context
    for i, url in enumerate(unique_image_urls[:3]):  # Limit to first 3 images
        context_parts.append(f"Image {i+1}: {url}")
    
    context = "\n\n".join(context_parts)
    
    prompt = f"""Analyze this multi-modal context and answer the query:
    
    Context:
    {context}
    
    Query: {query}
    
    Provide a comprehensive answer considering both text and visual elements:"""
    
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error generating response: {str(e)}"

if __name__ == "__main__":
    try:
        pdf_path = r"C:\Users\Megh\Desktop\2nd yr\4th sem\WE\Experiment 1_WE_Megh_Dave.pdf"
        process_pdf(pdf_path)
        
        print("Multi-modal RAG system ready. Ask me anything about the document!")
        while True:
            query = input("\nYour question: ")
            if query.lower() in ('exit', 'quit'):
                break
            response = generate_response(query)
            print(f"\nAssistant: {response}")
    except Exception as e:
        print(f"Error: {str(e)}")
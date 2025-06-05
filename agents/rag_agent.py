from typing import Dict, Any, List, Optional, Tuple
import os
import tempfile
import logging
import json
import uuid
import PyPDF2
import re
import shutil
from .base import BaseAgent
import google.generativeai as genai
import chromadb
import numpy as np

logger = logging.getLogger(__name__)

class GeminiEmbeddingFunction:
    """Custom embedding function for Gemini API that follows ChromaDB interface"""
    
    def __init__(self, api_key: str, model_name: str = "models/embedding-001", embedding_dim: int = 768):
        self.api_key = api_key
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name)
        
    def __call__(self, input: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts (reduced dimension)
        
        Args:
            input: List of text strings to embed (parameter name must be 'input' for ChromaDB compatibility)
            
        Returns:
            List of embedding vectors
        """
        if not input:
            return []
        
        try:
            embeddings = []
            # Process in batches to avoid rate limits
            batch_size = 10
            
            for i in range(0, len(input), batch_size):
                batch = input[i:i+batch_size]
                batch_embeddings = []
                
                for text in batch:
                    if not text or not text.strip():
                        # Handle empty text with zero embedding
                        batch_embeddings.append([0.0] * self.embedding_dim)
                        continue
                    
                    # Generate embedding for the text
                    try:
                        embedding_result = genai.embed_content(
                            model=self.model_name,
                            content=text,
                            task_type="retrieval_document"
                        )
                        
                        # Use only the first embedding_dim elements
                        if hasattr(embedding_result, "embedding"):
                            emb = embedding_result.embedding[:self.embedding_dim]
                        elif isinstance(embedding_result, dict) and "embedding" in embedding_result:
                            emb = embedding_result["embedding"][:self.embedding_dim]
                        else:
                            logger.warning(f"Unexpected embedding response format: {embedding_result}")
                            emb = [0.0] * self.embedding_dim
                        batch_embeddings.append(emb)
                    
                    except Exception as e:
                        logger.error(f"Error generating embedding: {str(e)}")
                        batch_embeddings.append([0.0] * self.embedding_dim)  # Use zero embedding on error
                
                embeddings.extend(batch_embeddings)
            
            return embeddings
        
        except Exception as e:
            logger.error(f"Error in embedding function: {str(e)}")
            # Return zero embeddings on error
            return [[0.0] * self.embedding_dim] * len(input)

class RAGAgent(BaseAgent):
    """Retrieval-Augmented Generation agent using ChromaDB and Gemini without LangChain"""
    
    def __init__(self, config: Dict[str, Any], system_instruction: str):
        super().__init__(config, system_instruction)
        self.api_key = config.get('api_key')
        if not self.api_key:
            raise ValueError("API key is required for RAGAgent")
        self.model_name = config.get('model', 'gemini-1.5-flash')
        self.temperature = float(config.get('temperature', 0.7))
        self.max_tokens = int(config.get('max_tokens', 1024))
        self.top_p = float(config.get('top_p', 0.95))
        self.top_k = int(config.get('top_k', 40))
        self.chunk_size = int(config.get('chunk_size', 1024))
        self.chunk_overlap = int(config.get('chunk_overlap', 128))
        self.num_results = int(config.get('num_results', 5))

        agent_id = config.get('agent_id')
        if not agent_id:
            raise ValueError("agent_id is required for RAGAgent")

        # Persistent ChromaDB path per agent
        self.collection_name = f"rag_collection_{agent_id}"
        
        self.collection_path = os.path.join("db", "chroma", str(agent_id))
        os.makedirs(self.collection_path, exist_ok=True)
        print(f"Using ChromaDB collection path: {self.collection_path}", flush=True)
        # Embedding function
        self.embedding_function = GeminiEmbeddingFunction(
            api_key=self.api_key,
            model_name="models/embedding-001"
        )
        # Persistent ChromaDB client
        self.chroma_client = chromadb.PersistentClient(path=self.collection_path)


        print(f"All collections in {self.collection_path}:", flush=True)
        try:
            all_collections = self.chroma_client.list_collections()
            for col in all_collections:
                print(f"  - Collection: {col.name}, Count: {col.count()}", flush=True)
        except Exception as e:
            print(f"Error listing collections: {e}", flush=True)

        # Create or get collection
        try:
            self.collection = self.chroma_client.get_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
            print(f"Using existing ChromaDB collection: {self.collection_name}", flush=True)
            print(f"Collection metadata: {self.collection.get_metadata()}", flush=True)
        except Exception as e:
            print(f"Failed to get collection '{self.collection_name}': {str(e)}", flush=True)
            # Try to get collection without embedding function first
            try:
                self.collection = self.chroma_client.get_collection(name=self.collection_name)
                print(f"Got existing collection without embedding function: {self.collection_name}", flush=True)
            except Exception:
                # Collection truly doesn't exist, create it
                print(f"Collection '{self.collection_name}' does not exist, creating new one...", flush=True)
                try:
                    self.collection = self.chroma_client.create_collection(
                        name=self.collection_name,
                        embedding_function=self.embedding_function,
                        metadata={"description": "RAG document collection"}
                    )
                except Exception as create_error:
                    # Handle case where collection exists but with different config
                    if "already exists" in str(create_error).lower():
                        print(f"Collection exists but with different config, getting existing one...", flush=True)
                        self.collection = self.chroma_client.get_collection(name=self.collection_name)
                    else:
                        raise create_error
        
        # Generate safety settings
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_ONLY_HIGH"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_ONLY_HIGH"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_ONLY_HIGH"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_ONLY_HIGH"
            }
        ]
        
        # Initialize model
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config={
                "temperature": self.temperature,
                "max_output_tokens": self.max_tokens,
                "top_p": self.top_p,
                "top_k": self.top_k
            },
            safety_settings=safety_settings
        )
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks of specified size with overlap
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        chunks = []
        
        # Strip excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # If text is shorter than chunk size, just return it
        if len(text) <= self.chunk_size:
            return [text]
        
        start = 0
        while start < len(text):
            # Get a chunk of text
            end = start + self.chunk_size
            if end > len(text):
                end = len(text)
            
            # Try to find a good breaking point (sentence or paragraph)
            if end < len(text):
                # Look for paragraph break
                paragraph_break = text.rfind('\n\n', start, end)
                if paragraph_break != -1 and paragraph_break > start + 200:  # Don't make chunks too small
                    end = paragraph_break
                else:
                    # Look for sentence break
                    sentence_break = max(
                        text.rfind('. ', start, end),
                        text.rfind('! ', start, end),
                        text.rfind('? ', start, end)
                    )
                    if sentence_break != -1 and sentence_break > start + 100:
                        end = sentence_break + 1  # Include the period
            
            # Add chunk to list
            chunks.append(text[start:end].strip())
            
            # Move start position accounting for overlap
            start = end - self.chunk_overlap
        
        return chunks
    
    def extract_text_from_pdf(self, file_path: str) -> Dict[str, str]:
        """Extract text from PDF file
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Dictionary mapping page numbers to page text
        """
        pages = {}
        
        try:
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for i, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    if text and text.strip():
                        pages[i + 1] = text.strip()  # 1-indexed page numbers
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
        
        return pages
    
  
   

    async def process(self, input_text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        print(f"RAGAgent processing input: {input_text}", flush=True)
        context = context or {}
        print(context, flush=True)
        
        # Ensure we're using the correct collection for this agent
        try:
            self.collection = self.chroma_client.get_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
        except Exception:
            # Collection doesn't exist, create it
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata={"description": "RAG document collection"}
            )
        
        # If document provided, process it (only once)
        if "document_content" in context and "filename" in context:
            doc_result = await self.process_document(context["document_content"], context["filename"])
            if doc_result["status"] == "error":
                return {"status": "error", "error": f"Failed to process document: {doc_result['error']}"}
            # If this is just a document upload with empty query, return success
            if not input_text:
                return {
                    "status": "success",
                    "message": f"Document '{context['filename']}' processed successfully",
                    "document": doc_result["document"]
                }
        # Query ChromaDB using the correct collection
        try:
            query_results = self.collection.query(query_texts=[input_text], n_results=self.num_results)
            # DEBUG: Log the raw ChromaDB query results
            logger.info(f"RAGAgent ChromaDB query for: '{input_text}' => {query_results}")
            # Extract content from relevant documents
            if not query_results["documents"] or not query_results["documents"][0]:
                context_text = "No relevant information found in the documents."
                sources = []
                debug_chunks = []
                debug_metadatas = []
            else:
                # Extract the text from the most relevant chunks
                chunks = query_results["documents"][0]
                metadatas = query_results["metadatas"][0]
                context_text = "\n\n".join([
                    f"[Document: {meta.get('filename', 'Unknown')}, Page {meta.get('page', 'Unknown')}]:\n{chunk}"
                    for chunk, meta in zip(chunks, metadatas)
                ])
                # Prepare sources info for response
                sources = []
                for i, (chunk, meta) in enumerate(zip(chunks[:3], metadatas[:3])):  # Top 3 sources
                    source_info = {
                        "page": meta.get("page", "Unknown"),
                        "source": meta.get("source", f"Page {meta.get('page', 'Unknown')} of {meta.get('filename', 'Unknown')}") ,
                        "excerpt": chunk[:100] + "..." if len(chunk) > 100 else chunk
                    }
                    sources.append(source_info)
                debug_chunks = chunks
                debug_metadatas = metadatas
            from datetime import date
            today_str = date.today().strftime("%B %d, %Y")
            # Build structured prompt with RAG context to prevent prompt injection
            system_content = f"""You are an AI assistant with access to the following document information:

{context_text}

Please be accurate, helpful, and ONLY use information from the provided context when answering specific questions about the document content.
If you don't know or the information isn't in the context, say "The document does not provide that information.". You are not allowed to use your own knowledge or make up information.
If you ever need it, this is the today's date: {today_str}.

{self.system_instruction}"""

            # Create structured messages to prevent prompt injection
            system_message = {"role": "system", "content": system_content}
            user_message = {"role": "user", "content": input_text}
            
            # Format the messages properly for Gemini API
            rag_prompt = f"[{system_message['role']}]: {system_message['content']}\n\n[{user_message['role']}]: {user_message['content']}"
            # Generate response using Gemini
            response = self.model.generate_content(rag_prompt)
            # Extract the response text
            response_text = response.text if hasattr(response, "text") else str(response)
            # Calculate token usage (approximate since Gemini doesn't always return this)
            # Estimation: ~4 chars per token
            prompt_tokens = len(rag_prompt) // 4
            completion_tokens = len(response_text) // 4
            total_tokens = prompt_tokens + completion_tokens
            token_usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            }
            # Return debug info for troubleshooting
            return {
                "output": response_text,
                "status": "success",
                "response": response_text,
                "token_usage": token_usage,
                "sources": sources,
                "debug_chunks": debug_chunks,
                "debug_metadatas": debug_metadatas,
                "rag_prompt": rag_prompt
            }
        except Exception as e:
            logger.error(f"Error with RAG processing: {str(e)}")
            return {
                "status": "error",
                "error": f"Error with RAG processing: {str(e)}"
            }
            
    def __del__(self):
        """Cleanup temporary files when object is deleted (do NOT delete ChromaDB collection!)"""
        # Only clean up temp files if you use any, but do NOT delete persistent ChromaDB collection
        pass
        # If you want to clean up temp files, do it here, but do not touch ChromaDB collections
        # Example:
        # if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
        #     try:
        #         shutil.rmtree(self.temp_dir)
        #     except Exception as e:
        #         logger.warning(f"Failed to delete temporary directory: {str(e)}")

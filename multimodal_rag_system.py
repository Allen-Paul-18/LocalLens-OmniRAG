"""
Multimodal RAG System - Backend Implementation
Supports offline operation with local models and vector database
"""

import os
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np
import re
import math
from collections import Counter

# Document processing
from docx import Document
import PyPDF2
from PIL import Image
import pytesseract

# Audio processing
# import speech_recognition as sr  # Replaced by faster_whisper
from faster_whisper import WhisperModel
from pydub import AudioSegment

# Embedding models (all can run offline)
from sentence_transformers import SentenceTransformer
try:
    from sentence_transformers.cross_encoder import CrossEncoder
except Exception:
    CrossEncoder = None
import torch
from transformers import CLIPProcessor, CLIPModel

# Local vector database
import chromadb

import json as _json

# LLM for generation (can use local models)
from transformers import AutoTokenizer, AutoModelForCausalLM


@dataclass
class DocumentChunk:
    """Represents a chunk of content from any modality"""
    id: str
    source_file: str
    content: str
    modality: str  # 'text', 'image', 'audio'
    metadata: Dict[str, Any]
    embedding: List[float] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class MultimodalRAGSystem:
    """
    Complete RAG system supporting PDF, DOCX, Images, and Audio
    Designed for offline operation with local models
    """
    
    def __init__(self,
                 storage_dir: str = "./rag_storage",
                 use_gpu: bool = True,
                 embedding_strategy: str = "minilm",
                 llm_path: str = None,
                 llm_backend: str = "local",
                 llm_n_gpu_layers: int = 0,
                 llm_base_url: str = "http://localhost:1234/v1",
                 llm_api_key: str = "lm-studio"):
        """
        Initialize the RAG system with local models

        Args:
            storage_dir: Directory for storing vector database and cache
            use_gpu: Whether to use GPU acceleration if available
            embedding_strategy: Strategy for embeddings ('minilm', 'clip', etc.)
            llm_path: Path to local GGUF model file (llama-cpp backend)
            llm_backend: "local" for llama-cpp or "external" for an OpenAI-compatible server
            llm_n_gpu_layers: Number of GGUF layers to offload; 0=CPU, -1=all possible layers
            llm_base_url: Base URL of the external LLM server, e.g. http://localhost:1234/v1
            llm_api_key: API key for the external server (LM Studio accepts a placeholder)
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.embedding_strategy = embedding_strategy.lower()
        print(f"Initializing on device: {self.device} (strategy={self.embedding_strategy})")

        # Initialize embedding models
        print("Loading embedding models...")
        self._init_embedding_models()

        # Initialize vector database (ChromaDB)
        print("Setting up local ChromaDB vector store...")
        self._init_vector_db()
        
        # Initialize the selected LLM backend. Only one backend is active at a time.
        self.llm_backend = (llm_backend or "local").strip().lower()
        self.llm_model = None
        self.llm_tokenizer = None
        self.llm_client = None
        self.llm_external_model = None

        if self.llm_backend == "external":
            self._init_external_llm(llm_base_url, llm_api_key)
        elif self.llm_backend == "local":
            if llm_path:
                self._init_llm(llm_path, n_gpu_layers=llm_n_gpu_layers)
        else:
            raise ValueError("llm_backend must be either 'local' or 'external'")
        
        # Document cache
        self.documents = {}
        # File-level embedding aggregates
        self.file_emb_sum = {}
        self.file_emb_count = {}
        self.file_emb_mean = {}
        # BM25 / lexical stats
        self.df = {}  # document frequency per term
        self.doc_count = 0
        self.doc_len = {}  # doc_id -> length
        self.doc_len_sum = 0
        self.doc_terms = {}  # doc_id -> set(terms)
        self.doc_term_freqs = {}  # doc_id -> Counter(tokens)
        
        # Initialize Whisper model
        print("Loading Whisper model...")
        try:
            from faster_whisper import WhisperModel
            # 'small' model is a good balance. It will download the binary automatically if not present.
            # compute_type="float16" is standard for GPU
            compute_type = "float16" if self.device == "cuda" else "int8"
            self.whisper_model = WhisperModel("small", device=self.device, compute_type=compute_type)
        except Exception as e:
            print(f"Failed to load Whisper model: {e}")
            self.whisper_model = None
        
    def _init_embedding_models(self):
        """Initialize models for different modalities"""
        # Text embeddings - sentence-transformers works offline
        self.text_model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        # Target embedding dimension (from text model)
        try:
            self.embedding_dim = self.text_model.get_sentence_embedding_dimension()
        except Exception:
            # Fallback to 384 which is the dimension for all-MiniLM-L6-v2
            self.embedding_dim = 384
        
        if self.embedding_strategy == 'minilm':
             # Still load CLIP for search (but maybe skip indexing with it?)
             # Or just allow it. The user wants search_by_image to work now.
             pass

        # Image embeddings - CLIP for multimodal (always load for cross-modal search)
        try:
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model.to(self.device)

            # Create a small projection to align CLIP output dim to text embedding dim
            # Many CLIP models output 512-d vectors while the text model above is 384-d.
            clip_out_dim = None
            try:
                clip_out_dim = getattr(self.clip_model.config, 'projection_dim', None)
            except Exception:
                clip_out_dim = None

            if clip_out_dim is None:
                # Try some other likely config fields, fallback to 512
                clip_out_dim = getattr(self.clip_model.config, 'hidden_size', 512)

            if clip_out_dim != self.embedding_dim:
                self.clip_projection = torch.nn.Linear(int(clip_out_dim), int(self.embedding_dim)).to(self.device)
                # Keep in eval mode and don't train
                self.clip_projection.eval()
            else:
                self.clip_projection = None
        except Exception as e:
            print(f"Error loading CLIP: {e}")
            self.clip_model = None
            self.clip_processor = None
            self.clip_projection = None

        # Try to load a cross-encoder reranker for better re-ranking (optional)
        if CrossEncoder is not None:
            try:
                self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=self.device)
            except Exception:
                self.reranker = None
        else:
            self.reranker = None
        
    # ChromaDB collections
    _CHROMA_COLLECTIONS = {
        "text": "rag_text",
        "image": "rag_image",
        "audio": "rag_audio",
    }


    def _init_vector_db(self):
        """Initialize local persistent ChromaDB."""

        db_path = self.storage_dir / "chroma"

        print(f"Opening ChromaDB at: {db_path}")

        self._chroma_client = chromadb.PersistentClient(
            path=str(db_path)
        )

        self._collections = {}

        for modality, collection_name in self._CHROMA_COLLECTIONS.items():

            collection = self._chroma_client.get_or_create_collection(
                name=collection_name,
                metadata={
                    "hnsw:space": "cosine"
                }
            )

            self._collections[modality] = collection

            print(
                f"  Collection ready: "
                f"{collection_name} "
                f"({collection.count()} items)"
            )


    def _chroma_metadata(self, chunk: DocumentChunk):
        """Convert chunk metadata into Chroma-compatible metadata."""

        safe_metadata = self._sanitize_metadata(
            chunk.metadata
        )

        return {
            "source_file": str(chunk.source_file),
            "modality": str(chunk.modality),
            "metadata_json": json.dumps(safe_metadata),
        }


    def _result_from_chroma(
        self,
        modality,
        item_id,
        document,
        metadata,
        distance
    ):
        """Convert Chroma result to the existing application format."""

        try:
            meta = json.loads(
                metadata.get("metadata_json", "{}")
            )
        except Exception:
            meta = {}

        return {
            "id": item_id,
            "content": document or "",
            "metadata": meta,
            "distance": float(distance),
            "relevance": max(
                0.0,
                1.0 - float(distance)
            ),
            "modality": modality,
        }


    def drop_database(self):
        """Delete all ChromaDB collections."""

        for collection_name in self._CHROMA_COLLECTIONS.values():

            try:
                self._chroma_client.delete_collection(
                    collection_name
                )

                print(
                    f"Deleted collection: {collection_name}"
                )

            except Exception as e:

                print(
                    f"Could not delete collection "
                    f"{collection_name}: {e}"
                )

        self._collections = {}

        self._init_vector_db()

        # Reset internal caches

        self.documents.clear()

        self.doc_count = 0
        self.doc_len_sum = 0

        self.df.clear()
        self.doc_len.clear()
        self.doc_terms.clear()
        self.doc_term_freqs.clear()

        self.file_emb_sum.clear()
        self.file_emb_count.clear()
        self.file_emb_mean.clear()
    
    def _generate_doc_id(self, filepath: str, chunk_index: int = 0) -> str:
        """Generate unique document ID"""
        content = f"{filepath}_{chunk_index}_{datetime.now().isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()
    
    # ==================== TEXT PROCESSING ====================
    
    def extract_text_from_pdf(self, filepath: str) -> List[DocumentChunk]:
        """Extract text from PDF with page information"""
        chunks = []
        
        with open(filepath, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                
                if text.strip():
                    # Split into smaller chunks if needed
                    text_chunks = self._split_text(text, max_length=500)
                    
                    for chunk_idx, chunk_text in enumerate(text_chunks):
                        chunk = DocumentChunk(
                            id=self._generate_doc_id(filepath, page_num * 100 + chunk_idx),
                            source_file=os.path.basename(filepath),
                            content=chunk_text,
                            modality='text',
                            metadata={
                                'page': page_num + 1,
                                'chunk_index': chunk_idx,
                                'file_type': 'pdf',
                                'file_path': filepath
                            }
                        )
                        chunks.append(chunk)
        
        return chunks
    
    def extract_text_from_docx(self, filepath: str) -> List[DocumentChunk]:
        """Extract text from DOCX documents"""
        chunks = []
        doc = Document(filepath)
        
        full_text = []
        for para in doc.paragraphs:
            if para.text.strip():
                full_text.append(para.text)
        
        text = '\n'.join(full_text)
        text_chunks = self._split_text(text, max_length=500)
        
        for chunk_idx, chunk_text in enumerate(text_chunks):
            chunk = DocumentChunk(
                id=self._generate_doc_id(filepath, chunk_idx),
                source_file=os.path.basename(filepath),
                content=chunk_text,
                modality='text',
                metadata={
                    'chunk_index': chunk_idx,
                    'file_type': 'docx',
                    'file_path': filepath
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def _split_text(self, text: str, max_length: int = 500) -> List[str]:
        """Split text into chunks while preserving sentences"""
        sentences = text.split('. ')
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            if current_length + sentence_length > max_length and current_chunk:
                chunks.append('. '.join(current_chunk) + '.')
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        if current_chunk:
            chunks.append('. '.join(current_chunk))
        
        return chunks
    
    # ==================== IMAGE PROCESSING ====================
    
    def extract_text_from_image(self, filepath: str) -> List[DocumentChunk]:
        """Extract text from images using OCR and generate image embeddings"""
        chunks = []

        # Load image
        image = Image.open(filepath)

        # OCR text extraction
        try:
            ocr_text = pytesseract.image_to_string(image)
        except Exception:
            ocr_text = ""

        # Get image metadata
        metadata = {
            'file_type': 'image',
            'format': image.format,
            'size': image.size,
            'mode': image.mode,
            'ocr_available': bool(ocr_text.strip())
        }

        # If strategy allows visual embeddings, keep an image chunk
        if self.embedding_strategy != 'minilm':
            img_chunk = DocumentChunk(
                id=self._generate_doc_id(filepath, 0),
                source_file=os.path.basename(filepath),
                content=f"Image: {os.path.basename(filepath)}",
                modality='image',
                metadata=metadata
            )
            img_chunk.metadata['image_path'] = filepath
            img_chunk.metadata['file_path'] = filepath
            chunks.append(img_chunk)

        # If OCR text exists, also add a text chunk so lexical/semantic text search finds it
        if ocr_text and ocr_text.strip():
            text_meta = {
                'file_type': 'image_ocr',
                'ocr_source': os.path.basename(filepath),
                'image_path': filepath
            }
            text_chunk = DocumentChunk(
                id=self._generate_doc_id(filepath, 1),
                source_file=os.path.basename(filepath),
                content=ocr_text,
                modality='text',
                metadata=text_meta
            )
            text_chunk.metadata['file_path'] = filepath
            chunks.append(text_chunk)

        return chunks
    
    def embed_image(self, image_path: str) -> np.ndarray:
        """Generate CLIP embedding for an image"""
        image = Image.open(image_path)
        inputs = self.clip_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)

        # Some transformer versions return a ModelOutput-like object
        # which doesn't have .cpu(). Handle both cases robustly.
        tensor = self._unwrap_model_output(image_features)
        # Project if needed
        if getattr(self, 'clip_projection', None) is not None:
            with torch.no_grad():
                tensor = self.clip_projection(tensor)

        arr = tensor.cpu().numpy()[0]
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return arr
    
    # ==================== AUDIO PROCESSING ====================
    
    def extract_text_from_audio(self, filepath: str) -> List[DocumentChunk]:
        """Transcribe audio files to text using local Whisper model"""
        chunks = []

        try:
            # Transcribe with faster-whisper (handles segmentation internally)
            # beam_size=5 is standard for accuracy
            segments, info = self.whisper_model.transcribe(filepath, beam_size=5)
            
            print(f"Detected language '{info.language}' with probability {info.language_probability}")

            chunk_idx = 0
            # faster-whisper returns a generator, so we iterate
            for segment in segments:
                if segment.text.strip():
                    chunk = DocumentChunk(
                        id=self._generate_doc_id(filepath, chunk_idx),
                        source_file=os.path.basename(filepath),
                        content=segment.text.strip(),
                        modality='audio',
                        metadata={
                            'chunk_index': chunk_idx,
                            'file_type': 'audio',
                            'start_ms': int(segment.start * 1000),
                            'end_ms': int(segment.end * 1000),
                            'duration': segment.end - segment.start,
                            'file_path': filepath
                        }
                    )
                    chunks.append(chunk)
                    chunk_idx += 1
                    
        except Exception as e:
            print(f"Error transcribing audio: {e}")

        return chunks
    
    # ==================== EMBEDDING & INDEXING ====================
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for text"""
        emb = self.text_model.encode(text, convert_to_numpy=True)
        # Normalize to unit-length
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb
    
    def embed_text_with_clip(self, text: str) -> np.ndarray:
        """Generate CLIP embedding for text (for cross-modal search)"""
        inputs = self.clip_processor(text=text, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**inputs)

        tensor = self._unwrap_model_output(text_features)
        if getattr(self, 'clip_projection', None) is not None:
            with torch.no_grad():
                tensor = self.clip_projection(tensor)

        arr = tensor.cpu().numpy()[0]
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return arr

    def _unwrap_model_output(self, output):
        """Return a torch.Tensor from various transformer ModelOutput shapes.

        Supports raw tensors, tuples/lists, and ModelOutput objects with
        common attributes like 'image_embeds', 'text_embeds',
        'pooler_output', or 'last_hidden_state'.
        """
        # If it's already a tensor-like object with .cpu(), return it.
        if hasattr(output, 'cpu'):
            return output

        # Common attribute names for CLIP/transformers outputs
        for attr in ('image_embeds', 'text_embeds', 'pooler_output', 'last_hidden_state'):
            if hasattr(output, attr):
                val = getattr(output, attr)
                # If last_hidden_state is returned, pool to a vector (take CLS/first token)
                if attr == 'last_hidden_state':
                    try:
                        return val[:, 0]
                    except Exception:
                        return val
                return val

        # If it's a tuple/list, assume first element is the tensor
        if isinstance(output, (tuple, list)) and len(output) > 0:
            return output[0]

        raise ValueError('Unable to extract tensor from model output')

    def _sanitize_metadata(self, obj):
        """Recursively convert metadata values to Chroma-safe types.

        Converts tuples to lists, numpy scalars to Python scalars, Path to str,
        and falls back to string for unknown types.
        """
        # dict
        if isinstance(obj, dict):
            return {str(k): self._sanitize_metadata(v) for k, v in obj.items()}

        # list/tuple/set -> list
        if isinstance(obj, (list, tuple, set)):
            return [self._sanitize_metadata(v) for v in obj]

        # numpy scalar
        try:
            if isinstance(obj, np.generic):
                return obj.item()
        except Exception:
            pass

        # Path -> str
        try:
            from pathlib import Path as _Path
            if isinstance(obj, _Path):
                return str(obj)
        except Exception:
            pass

        # Primitive types
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj

        # Fallback: convert to string
        try:
            return str(obj)
        except Exception:
            return None
    
    def index_document(self, chunk: DocumentChunk):
        """Add document chunk to vector database"""
        # Generate appropriate embedding based on modality
        if chunk.modality == 'image':
            # In MiniLM-only mode, index image as text (OCR or filename)
            if self.embedding_strategy == 'minilm':
                chunk.embedding = self.embed_text(chunk.content).tolist()
                # mark metadata to indicate converted image->text
                chunk.metadata = dict(chunk.metadata)
                chunk.metadata['converted_from_image'] = True
            else:
                if 'image_path' in chunk.metadata and self.clip_model is not None:
                    chunk.embedding = self.embed_image(chunk.metadata['image_path']).tolist()
                else:
                    chunk.embedding = self.embed_text_with_clip(chunk.content).tolist()
        else:
            chunk.embedding = self.embed_text(chunk.content).tolist()

        # Ensure embedding is normalized (numpy -> list)
        emb = np.array(chunk.embedding, dtype=float)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        chunk.embedding = emb.tolist()

        # Store chunk in ChromaDB

        safe_metadata = self._chroma_metadata(chunk)

        collection = self._collections.get(
            chunk.modality,
            self._collections["text"]
        )

        collection.upsert(
            ids=[chunk.id],
            embeddings=[chunk.embedding],
            documents=[chunk.content],
            metadatas=[safe_metadata],
        )

        # Store in cache
        self.documents[chunk.id] = chunk

        # Update file-level aggregate embedding
        key = chunk.source_file
        try:
            emb = np.array(chunk.embedding, dtype=float)
        except Exception:
            emb = None

        if emb is not None:
            if key in self.file_emb_sum:
                self.file_emb_sum[key] += emb
                self.file_emb_count[key] += 1
            else:
                self.file_emb_sum[key] = emb.copy()
                self.file_emb_count[key] = 1

            mean = self.file_emb_sum[key] / float(self.file_emb_count[key])
            norm = np.linalg.norm(mean)
            if norm > 0:
                mean = mean / norm
            self.file_emb_mean[key] = mean

        # --- BM25 bookkeeping ---
        try:
            tokens = self._tokenize(chunk.content)
            tset = set(tokens)
            # avoid double counting if same chunk re-indexed
            if chunk.id not in self.doc_terms:
                for t in tset:
                    self.df[t] = self.df.get(t, 0) + 1
                self.doc_count += 1
                self.doc_terms[chunk.id] = tset
                tf = Counter(tokens)
                self.doc_term_freqs[chunk.id] = tf
                self.doc_len[chunk.id] = len(tokens)
                self.doc_len_sum += len(tokens)
        except Exception:
            pass
    
    # ==================== DOCUMENT INGESTION ====================
    
    def ingest_document(self, filepath: str) -> List[DocumentChunk]:
        """
        Main ingestion function - detects file type and processes accordingly
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        print(f"Ingesting: {filepath.name}")
        
        # Determine file type and extract
        suffix = filepath.suffix.lower()
        
        if suffix == '.pdf':
            chunks = self.extract_text_from_pdf(str(filepath))
        elif suffix in ['.docx', '.doc']:
            chunks = self.extract_text_from_docx(str(filepath))
        elif suffix in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']:
            chunks = self.extract_text_from_image(str(filepath))
        elif suffix in ['.mp3', '.wav', '.m4a', '.flac']:
            chunks = self.extract_text_from_audio(str(filepath))
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
        
        # Index all chunks
        for chunk in chunks:
            self.index_document(chunk)
        
        print(f"Indexed {len(chunks)} chunks from {filepath.name}")
        return chunks
    
    # ==================== CROSS-MODAL SEARCH ====================

    def search_by_image(
        self,
        image_path: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:

        """
        Search all ChromaDB collections using an image query.
        """

        query_embedding = self.embed_image(
            image_path
        ).tolist()

        results = []

        initial_n = max(20, top_k)

        for modality, collection in self._collections.items():

            count = collection.count()

            if count == 0:
                continue

            try:

                data = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=min(initial_n, count),
                    include=[
                        "documents",
                        "metadatas",
                        "distances"
                    ],
                )

                for item_id, document, metadata, distance in zip(
                    data["ids"][0],
                    data["documents"][0],
                    data["metadatas"][0],
                    data["distances"][0],
                ):

                    results.append(
                        self._result_from_chroma(
                            modality,
                            item_id,
                            document,
                            metadata or {},
                            distance,
                        )
                    )

            except Exception as e:

                print(
                    f"Error searching ChromaDB "
                    f"{collection.name}: {e}"
                )

        # Deduplicate

        best = {}

        for result in results:

            if (
                result["id"] not in best
                or result["relevance"]
                > best[result["id"]]["relevance"]
            ):
                best[result["id"]] = result

        results = list(best.values())

        # Filter

        results = [
            r for r in results
            if r["relevance"] >= 0.3
        ]

        results.sort(
            key=lambda x: x["relevance"],
            reverse=True
        )

        return results[:top_k]

    def search_by_audio(self, audio_path: str, top_k: int = 5) -> Tuple[List[Dict[str, Any]], str]:
        """
        Search using audio input.
        Transcribes audio using Whisper then performs text search.
        Returns: (results, transcript)
        """
        print(f"Transcribing audio query: {audio_path}")
        
        # Transcribe with Whisper
        segments, _ = self.whisper_model.transcribe(audio_path, beam_size=5)
        
        # Combine segments into single query text
        transcript = " ".join([segment.text for segment in segments]).strip()
        print(f"Transcript: {transcript}")
        
        if not transcript:
            return [], ""
            
        return self.search(transcript, top_k=top_k), transcript


    
    def search(self, 
               query: str, 
               modality: str = None,
               top_k: int = 5,
               cross_modal: bool = True) -> List[Dict[str, Any]]:
        """
        Search across all modalities
        
        Args:
            query: Search query (text)
            modality: Specific modality to search ('text', 'image', 'audio', or None for all)
            top_k: Number of results to return
            cross_modal: Whether to use CLIP for cross-modal search
        """
        results = []

        # Generate query embedding
        query_embedding = self.embed_text(
            query
        ).tolist()

        # Select collections
        if modality:

            collection = self._collections.get(
                modality
            )

            selected_collections = [
                (modality, collection)
            ]

        else:

            selected_collections = list(
                self._collections.items()
            )

        initial_n = max(20, top_k)

        # Search ChromaDB
        for modality_name, collection in selected_collections:

            if collection is None:
                continue

            count = collection.count()

            if count == 0:
                continue

            try:

                data = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=min(initial_n, count),
                    include=[
                        "documents",
                        "metadatas",
                        "distances"
                    ],
                )

                for item_id, document, metadata, distance in zip(
                    data["ids"][0],
                    data["documents"][0],
                    data["metadatas"][0],
                    data["distances"][0],
                ):

                    results.append(
                        self._result_from_chroma(
                            modality_name,
                            item_id,
                            document,
                            metadata or {},
                            distance,
                        )
                    )

            except Exception as e:

                print(
                    f"Error searching ChromaDB "
                    f"{collection.name}: {e}"
                )

        # Deduplicate

        best = {}

        for result in results:

            if (
                result["id"] not in best
                or result["relevance"]
                > best[result["id"]]["relevance"]
            ):
                best[result["id"]] = result

        results = list(best.values())

        # Pre-rank: take top candidates then rerank using MiniLM text similarity
        results.sort(key=lambda x: x['relevance'], reverse=True)
        candidates = results[:max(20, top_k)]

        q_emb = query_embedding

        # If a cross-encoder reranker is available, use it; otherwise use dot-product similarity
        if getattr(self, 'reranker', None) is not None:
            try:
                pairs = [(query, r['content']) for r in candidates]
                scores = self.reranker.predict(pairs)
                # Normalize cross-encoder scores to 0-1
                smin = float(np.min(scores))
                smax = float(np.max(scores))
                score01 = []
                for s in scores:
                    if smax > smin:
                        score01.append((float(s) - smin) / (smax - smin))
                    else:
                        score01.append(0.5)

                for r, s in zip(candidates, score01):
                    r['rerank_score'] = s
            except Exception:
                # Fallback to text-model similarity
                for r in candidates:
                    try:
                        chunk_emb = self.text_model.encode(r['content'], convert_to_numpy=True)
                        norm = np.linalg.norm(chunk_emb)
                        if norm > 0:
                            chunk_emb = chunk_emb / norm
                        sim = float(np.dot(q_emb, chunk_emb))
                    except Exception:
                        sim = r.get('relevance', 0.0)
                    r['rerank_score'] = (sim + 1.0) / 2.0
        else:
            for r in candidates:
                try:
                    chunk_emb = self.text_model.encode(r['content'], convert_to_numpy=True)
                    norm = np.linalg.norm(chunk_emb)
                    if norm > 0:
                        chunk_emb = chunk_emb / norm
                    sim = float(np.dot(q_emb, chunk_emb))
                except Exception:
                    sim = r.get('relevance', 0.0)
                # convert similarity (-1..1) to 0..1
                r['rerank_score'] = (sim + 1.0) / 2.0

        # Compute BM25 scores for candidates (lexical matching)
        def bm25_for_doc(query_tokens, doc_id, k1=1.5, b=0.75):
            N = max(1, self.doc_count)
            avgdl = (self.doc_len_sum / N) if N > 0 else 1.0
            score = 0.0
            tf = self.doc_term_freqs.get(doc_id, Counter())
            for term in query_tokens:
                df = self.df.get(term, 0)
                idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
                f = tf.get(term, 0)
                denom = f + k1 * (1 - b + b * (self.doc_len.get(doc_id, 0) / avgdl))
                if denom > 0:
                    score += idf * ((f * (k1 + 1)) / denom)
            return float(score)

        # Combine rerank score with file-level aggregate score and BM25 (small boost)
        for r in candidates:
            file_key = r['metadata'].get('source_file')
            file_score01 = 0.0
            if file_key and file_key in self.file_emb_mean:
                try:
                    femb = self.file_emb_mean[file_key]
                    file_sim = float(np.dot(q_emb, femb))
                    file_score01 = (file_sim + 1.0) / 2.0
                except Exception:
                    file_score01 = 0.0
            r['file_score01'] = file_score01

            # BM25 lexical score (normalized across candidates)
            try:
                q_tokens = self._tokenize(query)
                bm25_raw = bm25_for_doc(q_tokens, r['id'])
            except Exception:
                bm25_raw = 0.0
            r['bm25_raw'] = bm25_raw

        # Normalize BM25 across candidates
        bm25_vals = [c.get('bm25_raw', 0.0) for c in candidates]
        max_b = max(bm25_vals) if bm25_vals else 0.0
        for r in candidates:
            bm25_norm = (r.get('bm25_raw', 0.0) / max_b) if max_b > 0 else 0.0
            # Weighted combine: 70% reranker, 15% file-level, 15% BM25
            r['final_score'] = 0.70 * float(r.get('rerank_score', 0.0)) + 0.15 * float(r.get('file_score01', 0.0)) + 0.15 * float(bm25_norm)

        candidates.sort(key=lambda x: x['final_score'], reverse=True)
        
        # Filter poor matches
        filtered = [c for c in candidates if c['final_score'] >= 0.3]
        
        # Overwrite the base vector 'relevance' so the UI displays the true final score
        for c in filtered:
            c['relevance'] = c['final_score']
            
        return filtered[:top_k]
    
    # ==================== LOCAL LLM (GGUF) ====================

    def _init_llm(self, model_path: str, n_gpu_layers: int = -1, n_ctx: int = 4096):
        """Initialize local GGUF model"""

        if not model_path or not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"GGUF model file not found: {model_path}"
            )

        try:
            from llama_cpp import Llama
            print(f"Loading local LLM from {model_path}...")
            # llama-cpp CUDA support is independent of PyTorch CUDA support.
            # The UI probes llama_supports_gpu_offload() and passes the chosen mode.
            gpu_layers = max(0, n_gpu_layers) if n_gpu_layers >= 0 else -1
            if gpu_layers != 0:
                from llama_cpp import llama_cpp
                if not llama_cpp.llama_supports_gpu_offload():
                    print("llama-cpp CUDA offload requested, but this build has no GPU offload support. Falling back to CPU.")
                    gpu_layers = 0

            self.llm_model = Llama(
                model_path=model_path,
                n_gpu_layers=gpu_layers,
                n_ctx=n_ctx,
                verbose=True
            )
            print("Local LLM loaded successfully")
        except Exception as e:
            print(f"Failed to load local LLM: {e}")
            self.llm_model = None

    def _init_external_llm(self, base_url: str, api_key: str = "lm-studio"):
        """Connect to an OpenAI-compatible local server such as LM Studio."""
        import json
        import urllib.request
        import urllib.error

        self.llm_base_url = (base_url or "http://localhost:1234/v1").rstrip("/")
        self.llm_api_key = api_key or "lm-studio"

        try:
            request = urllib.request.Request(
                f"{self.llm_base_url}/models",
                headers={"Authorization": f"Bearer {self.llm_api_key}"},
                method="GET",
            )
            with urllib.request.urlopen(request, timeout=8) as response:
                payload = json.loads(response.read().decode("utf-8"))

            models = payload.get("data", [])
            if not models:
                raise RuntimeError("The server is reachable but returned no loaded models.")

            self.llm_external_model = models[0].get("id")
            if not self.llm_external_model:
                raise RuntimeError("The server did not provide a usable model ID.")

            self.llm_client = True  # Connection is handled with the standard library.
            print(
                f"Connected to external LLM server: {self.llm_base_url} "
                f"(model={self.llm_external_model})"
            )
        except Exception as e:
            self.llm_client = None
            self.llm_external_model = None
            print(f"Failed to connect to external LLM server: {e}")

    def generate_answer(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Generate an answer from retrieved context using the selected LLM backend.
        """
        if self.llm_backend == "external" and not self.llm_client:
            return (
                "External LLM server is not connected. Start LM Studio (or another "
                "OpenAI-compatible server), load a model, and check the API URL."
            )
        if self.llm_backend == "local" and not self.llm_model:
            return "Local LLM not initialized. Please configure a valid GGUF model path."

        # Format context with numbered citations.
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            meta = chunk.get('metadata', {})
            source_info = f"[Source {i}]: {meta.get('source_file', 'Unknown')}"

            if 'page' in meta:
                source_info += f" (Page {meta['page']})"
            elif 'start_ms' in meta:
                start_sec = meta['start_ms'] / 1000.0
                end_sec = meta.get('end_ms', 0) / 1000.0
                source_info += f" ({start_sec:.1f}s - {end_sec:.1f}s)"

            context_parts.append(f"{source_info}\\n{chunk.get('content', '')}")

        context_text = "\\n\\n".join(context_parts)
        system_prompt = (
            "You are a helpful AI assistant. Answer the user's question based ONLY "
            "on the provided context. You MUST cite sources using [1], [2], etc. "
            "Every factual statement should have a citation. If the answer is not "
            "in the context, say you don't know."
        )
        user_prompt = f"Context:\\n{context_text}\\n\\nQuestion: {query}"

        try:
            if self.llm_backend == "external":
                import json
                import urllib.request

                payload = {
                    "model": self.llm_external_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.7,
                    "max_tokens": 512,
                }
                request = urllib.request.Request(
                    f"{self.llm_base_url}/chat/completions",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.llm_api_key}",
                    },
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=120) as response:
                    result = json.loads(response.read().decode("utf-8"))
                return result["choices"][0]["message"]["content"].strip()

            # Keep the existing llama-cpp GGUF path for CPU and CUDA-enabled builds.
            prompt = (
                f"<s>[INST] {system_prompt}\\n\\n{user_prompt} [/INST]"
            )
            output = self.llm_model(
                prompt,
                max_tokens=512,
                stop=["</s>"],
                echo=False,
                temperature=0.7,
            )
            return output['choices'][0]['text'].strip()
        except Exception as e:
            return f"Error generating answer: {str(e)}"

    # ==================== UTILITIES ====================
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics from ChromaDB."""

        counts = {}
        total = 0

        for modality, collection in self._collections.items():

            try:
                count = collection.count()

            except Exception:
                count = 0

            counts[modality] = count
            total += count

        return {
            "total_documents": total,
            "text_chunks": counts.get("text", 0),
            "image_chunks": counts.get("image", 0),
            "audio_chunks": counts.get("audio", 0),
            "storage_path": str(self.storage_dir),
        }
    
    def export_index(self, output_path: str):
        """Export the document index for backup"""
        index_data = {
            'documents': {doc_id: asdict(doc) for doc_id, doc in self.documents.items()},
            'statistics': self.get_statistics(),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(index_data, f, indent=2)

    def unload_llm(self):
        """Unload the local GGUF model and release memory."""
        if self.llm_model is not None:
            del self.llm_model
            self.llm_model = None

            import gc
            gc.collect()

            print("Local LLM unloaded.")


def main():
    """Example usage"""
    # Initialize system
    rag = MultimodalRAGSystem(storage_dir="./rag_data")
    
    # Ingest documents
    sample_files = [
        "sample_report.pdf",
        "screenshot.png",
        "meeting_recording.mp3"
    ]
    
    for file in sample_files:
        if os.path.exists(file):
            try:
                rag.ingest_document(file)
            except Exception as e:
                print(f"Error processing {file}: {e}")
    
    # Search example
    query = "What was discussed about international development?"
    results = rag.search(query, top_k=5)
    
    print(f"\nSearch Results for: '{query}'")
    print("=" * 50)
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Source: {result['metadata'].get('source_file', 'Unknown')}")
        print(f"   Relevance: {result['relevance']:.2%}")
        print(f"   Content: {result['content'][:200]}...")
    
    # Generate answer
    answer = rag.generate_answer(query, results)
    print(f"\nGenerated Answer:\n{answer}")
    
    # Show statistics
    stats = rag.get_statistics()
    print(f"\nSystem Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()

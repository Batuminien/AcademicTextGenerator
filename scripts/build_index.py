#!pip install -Uq sentence-transformers hnswlib tqdm
#!pip install --upgrade setuptools

import json
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
import hnswlib

#BAAI/bge-m3 is a SOTA embedding model supporting long context as 8192 token limit.
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
BASE_DIR = "/data"
CHUNKS_PATH = "f{BASE_DIR}/chunks/chunks.jsonl"
INDEX_DIR = f"{BASE_DIR}/index"

REPO_ID = "forza61/academic-rag-data"
FILENAME = "chunks.jsonl"

def download_chunks():
    """
    If chunks.jsonl doesn't exist locally, it downloads it from Hugging Face.
    """
    path = Path(CHUNKS_PATH)
    if not path.exists():
        print(f"'{CHUNKS PATH}' not found. Downloading from Hugging Face...")
        
        # Klasör yoksa oluştur
        path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            downloaded_path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=HF_FILENAME,
                repo_type="dataset",
                local_dir=path.parent, #Download to chunks folder
                local_dir_use_symlinks=False
            )
            print(f"Download successful:: {downloaded_path}")
        except Exception as e:
            print(f"An error occurred during the download: {e}")
            raise e
    else:
        print(f"The chunks file already exists: {CHUNKS_PATH}")

def load_chunks(chunks_path: str):
    """
    Loads text chunks and their metadata from a JSONL file.

    Args:
        chunks_path: Path to the .jsonl file containing the chunks.

    Returns:
        A list of text strings to be embedded.
        A list of metadata dictionaries corresponding to each text.
    """
    texts = []
    metadatas = []

    path = Path(chunks_path)
    #assert path.exists(), f"Chunks file not found: {path}"

    with path.open("r", encoding="utf-8") as f:
        #Enumerate gives us a line index, but we mainly use it for progress tracking here
        for i, line in enumerate(tqdm(f, desc="Loading chunks")):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            raw_text = obj.get("text") or ""
            text = raw_text.strip()
            
            #Skip empty chunks to maintain index quality
            if not text:
                continue

            texts.append(text)

            #Construct a comprehensive metadata object for retrieval context
            meta = {
                "idx": i, #Original line index
                "chunk_id": obj.get("chunk_id"),
                "paper_id": obj.get("paper_id"),
                "title": obj.get("title"),
                "section_title": obj.get("section_title"),
                "section_path": obj.get("section_path"),
                "para_index": obj.get("para_index"),
                "reference_ids": obj.get("reference_ids", []),
                "inline_citations": obj.get("inline_citations", []),
                "references": obj.get("references", []),
                "year": obj.get("year"),
                "url": obj.get("url"),
                "venue": obj.get("venue"),
                "authors": obj.get("authors"),
                "text": text, #Storing text in metadata is useful for retrieval display
            }
            metadatas.append(meta)

        return texts, metadatas

def build_embeddings(texts, model_name: str, batch_size: int = 256):
    """
    Generates dense vector embeddings for the given texts using a SentenceTransformer model.

    Args:
        texts: List of texts to encode.
        model_name: HuggingFace model identifier (e.g., 'BAAI/bge-m3').
        batch_size: Number of texts to process in parallel on the GPU.

    Returns:
        A numpy matrix of shape (num_texts, embedding_dim).
    """
    print(f"Loading embedding model: {model_name} on CUDA...")
    
    #trust_remote_code=True is often required for newer architectures like BGE-M3
    model = SentenceTransformer(model_name, device="cuda", trust_remote_code=True)

    #Set max sequence length to handle longer academic paragraphs
    model.max_seq_length = 8192

    print(f"Encoding {len(texts)} chunks with batch_size={batch_size}...")
    
    # normalize_embeddings=True is crucial for cosine similarity search
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return embeddings

def build_hnsw_index(
        embeddings: np.ndarray,
        index_dir: str,
        space: str = "cosine"
):
    """
    Builds and saves an HNSW (Hierarchical Navigable Small World) index for fast approximate nearest neighbor search.

    Args:
        embeddings: The vector embeddings matrix.
        index_dir: Directory where the index binary will be saved.
        space: Distance metric ('l2', 'ip', or 'cosine'). Default is 'cosine'.

    Returns:
        The array of integer labels assigned to the vectors (0 to N-1).
    """
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    dim = embeddings.shape[1]
    num_elements = embeddings.shape[0]

    print(f"Building HNSW index | dim={dim}, n={num_elements}")

    #Initialize the index
    index = hnswlib.Index(space=space, dim=dim)

    # HNSW construction parameters:
    # - M: The number of bi-directional links created for every new element during construction. 
    #      Higher M works better for high dimensional data/recall but consumes more memory.
    # - ef_construction: Controls the trade-off between index construction time and accuracy.
    #                    Higher value provides better recall but longer build time.
    index.init_index(
        max_elements=num_elements,
        ef_construction=200,
        M=32,
    )

    #Create integer labels (IDs) for the index items
    labels = np.arange(num_elements)

    #Add items to the index
    index.add_items(embeddings, labels)

    # Query time parameter:
    # - ef: Controls the trade-off between search speed and recall during querying.
    index.set_ef(96)

    index_path = index_dir / "hnsw_index.bin"
    print(f"Saving HNSW index to: {index_path}")
    index.save_index(str(index_path))

    return labels

def save_metadata(metadatas, labels, index_dir: str):
    """
    Saves the metadata, mapping the HNSW integer labels back to the full data objects.

    Args:
        metadatas: The list of metadata dictionaries.
        labels: The corresponding integer labels used in the HNSW index.
        index_dir: Directory to save the metadata file.
    """
    index_dir = Path(index_dir)
    meta_path = index_dir / "metadatas.jsonl"

    print(f"Saving metadata to: {meta_path}")

    with meta_path.open("w", encoding="utf-8") as f:
        for label, meta in zip(labels, metadatas):
            #Create a copy to avoid mutating original data
            meta_out = dict(meta)
            #Explicitly store the label so we can map Search Result -> Metadata
            meta_out["label"] = int(label)
            f.write(json.dumps(meta_out, ensure_ascii=False) + "\n")

def main():
    download_chunks()
    
    print("Step 1: Loading chunks")
    texts, metadatas = load_chunks(CHUNKS_PATH)

    print("Step 2: Building embeddings")
    #Batch size reduced to 64 to be safe on standard GPUs.
    embeddings = build_embeddings(texts, EMBEDDING_MODEL_NAME, batch_size=64)

    print(f"Embeddings shape: {embeddings.shape}")

    print("Step 3: Building HNSW index")
    labels = build_hnsw_index(embeddings, INDEX_DIR, space="cosine")

    print("Step 4: Saving metadata")
    save_metadata(metadatas, labels, INDEX_DIR)

    print("Done! Index and metadata are ready.")

if __name__ == "__main__":
    main()
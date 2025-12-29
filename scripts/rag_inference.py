#!pip install -q sentence-transformers hnswlib

import os
import json
import re
import hnswlib
from huggingface_hub import hf_hub_download
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import defaultdict
from IPython.display import Markdown, display

root_dir = Path(__file__).resolve().parents[1]  # AcademicTextGenerator/
BASE_DIR = f"{root_dir}/data"
INDEX_DIR = f"{BASE_DIR}/index"
META_PATH = f"{INDEX_DIR}/metadatas.jsonl"

os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

REPO_ID = "forza61/academic-rag-data"

def download_data():
    print("Data files are being checked...")
    
    #Download HNSW index
    if not os.path.exists(f"{INDEX_DIR}/hnsw_index.bin"):
        print("Downloading HNSW index...")
        hf_hub_download(
            repo_id=REPO_ID,
            filename="hnsw_index.bin",
            repo_type="dataset",
            local_dir=INDEX_DIR
        )

    #Download metadata
    if not os.path.exists(f"{INDEX_DIR}/metadatas.jsonl"):
        print("Downloading Metadata...")
        hf_hub_download(
            repo_id=REPO_ID,
            filename="metadatas.jsonl",
            repo_type="dataset",
            local_dir=INDEX_DIR
        )
    print("Datas are ready.")

#Embedding Model: Used for encoding the user query
EMBEDDING_MODEL_ID = "BAAI/bge-m3"
EMBEDDING_DIM = 1024
MAX_SEQ_LENGTH = 8192

# Generator Model: The LLM that answers the question
LLM_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

#We use dictionaries to act as singletons for holding loaded models to avoid reloading.
rag_components = {
    "hnsw_index": None,
    "metadatas": None,
    "emb_model": None
}

llm_components = {
    "model": None,
    "tokenizer": None
}

#Retrieval System
def init_retrieval_system():
    """
    Initializes the retrieval stack:
    1. Loads metadata (mappings from ID to text/author).
    2. Loads the HNSW index (vector database).
    3. Loads the SentenceTransformer model (query encoder).
    """
    print("Loading metadata...")
    metas = []
    if not os.path.exists(META_PATH):
        raise FileNotFoundError(f"Metadata file not found at {META_PATH}")

    with open(META_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                metas.append(json.loads(line))

    rag_components["metadatas"] = metas
    print(f"-> Loaded {len(metas)} metadata entries")

    print(f"Loading HNSW index (Dim={EMBEDDING_DIM})...")
    if not os.path.exists(INDEX_DIR):
        raise FileNotFoundError(f"Index file not found at {INDEX_DIR}")

    #Initialize HNSW index
    index = hnswlib.Index(space="cosine", dim=EMBEDDING_DIM)
    index.load_index(INDEX_DIR)
    index.set_ef(128)
    rag_components["hnsw_index"] = index
    print(f"-> Index loaded with {index.get_current_count()} elements.")

    print(f"Loading embedding model: {EMBEDDING_MODEL_ID}...")
    model = SentenceTransformer(EMBEDDING_MODEL_ID, device="cuda", trust_remote_code=True)
    model.max_seq_length = MAX_SEQ_LENGTH
    rag_components["emb_model"] = model
    print("-> Embedding model ready.")

    print("\n RETRIEVAL SYSTEM READY!")

def retrieve_documents(query: str, top_k: int = 5):
    """
    Performs semantic search to find the most relevant document chunks.

    Args:
        query: The user's question.
        top_k: Number of chunks to retrieve.

    Returns:
        A list of dictionaries containing text, metadata, and similarity scores.
    """
    model = rag_components["emb_model"]
    index = rag_components["hnsw_index"]
    metadatas = rag_components["metadatas"]

    #Encode the query into a vector
    q_emb = model.encode([query], normalize_embeddings=True)

    #Search the HNSW index
    #labels: indices of the nearest neighbors
    #distances: cosine distances
    labels, distances = index.knn_query(q_emb, k=top_k)

    results = []
    for label, dist in zip(labels[0], distances[0]):
        #Map integer label back to full metadata
        meta = metadatas[int(label)]
        results.append({
            "idx": int(label),
            "distance": float(dist),
            "chunk_id": meta.get("chunk_id"),
            "text": meta.get("text"),
            "title": meta.get("title", "Unknown Title"),
            "authors": meta.get("authors", []),
            "year": meta.get("year", ""),
            "url": meta.get("url", ""),
            "references": meta.get("references", []),
            "section": meta.get("section_title", "")
        })

    return results

#LLM System
def init_llm_system():
    """
    Loads the Large Language Model and Tokenizer.
    Uses 'bfloat16' for memory efficiency on modern GPUs.
    """
    print(f"Loading LLM: {LLM_MODEL_ID}...")

    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_ID,
        dtype=torch.bfloat16, #Optimizes memory usage
        device_map="auto", #Automatically distributes model across available GPUs
        trust_remote_code=True
    )

    llm_components["model"] = model
    llm_components["tokenizer"] = tokenizer

    print(f"\n {LLM_MODEL_ID} READY")

#Generation and Formatting Logic
def get_full_author_list(authors_list):
    """Parses author dictionaries into a readable string."""
    if not authors_list:
        return "Unknown Authors"

    names = []
    for author in authors_list:
        first = author.get("firstname", "").strip()
        last = author.get("surname", "").strip()
        full_name = f"{first} {last}".strip()
        if full_name:
            names.append(full_name)

    return ", ".join(names)

def clean_ref_text(text):
    """Normalizes whitespace in reference texts."""
    if not text: return ""
    text = text.replace('\n', ' ').replace('\t', ' ')
    return re.sub(r'\s+', ' ', text).strip()

def query_qwen(user_query: str, use_rag: bool = True, top_k: int = 6):
    """
    The core RAG pipeline function.
    
    1. Retrieves context (if RAG is enabled).
    2. Constructs a system prompt that injects the context and strict citation rules.
    3. Generates the response using the LLM.

    Args:
        user_query: The user's question.
        use_rag: Whether to use retrieved documents or just the LLM's internal knowledge.
        top_k: Number of context chunks to retrieve.

    Returns:
        The generated text and the list of source documents used.
    """
    tokenizer = llm_components["tokenizer"]
    model = llm_components["model"]

    contexts = []
    context_block = ""

    if use_rag:
        print(f"Retrieving top {top_k} contexts for: '{user_query}'...")
        contexts = retrieve_documents(user_query, top_k=top_k)

        #Build the context block string to feed into the LLM
        for i, ctx in enumerate(contexts, 1):
            auth_str = get_full_author_list(ctx['authors'])
            year = ctx.get('year') or "n.d."
            title = ctx.get('title', 'Unknown Title')
            section = ctx.get('section', 'General')
            
            #Extract internal citations found within the retrieved chunk
            internal_refs_text = ""
            raw_refs = ctx.get('references', [])

            if raw_refs:
                internal_refs_text = "\n    > Studies cited within this text:\n"
                for ref in raw_refs:
                    rid = ref.get('id')
                    clean_text = clean_ref_text(ref.get('text')) # Senin helper fonksiyonun
                    if clean_text:
                        internal_refs_text += f"    * [Ref ID: {rid}] {clean_text}\n"

            context_block += f"--- SOURCE {i} ---\n"
            context_block += f"Primary Work: {title}\n"
            context_block += f"Authors: {auth_str} ({year})\n"
            context_block += f"Content (from {section}):\n{ctx['text']}\n"
            context_block += f"{internal_refs_text}\n"

        #Prompt Engineering
        #Defines the persona, citation rules, and strict constraints to prevent hallucinations.
        system_instruction = f"""[INST] You are an expert Academic Literature Reviewer and Research Assistant.
    Your goal is to synthesize the provided academic papers into a coherent, objective, scientifically accurate, and highly readable review.

    ### I. CITATION & INDEXING PROTOCOLS (STRICTLY FOLLOW):
    1.  **SEQUENTIAL RE-INDEXING RULE (CRITICAL):**
        * You will receive sources labeled with various IDs (e.g., `--- SOURCE 5 ---`, `--- SOURCE 12 ---`).
        * **IGNORE** these original numbers for your citations.
        * **RE-NUMBER** them based on their order of appearance in the provided context:
            * The **1st** source listed in the context becomes **[1]**.
            * The **2nd** source listed in the context becomes **[2]**.
            * And so on.
        * *Example:* If the context shows `Source 10` followed by `Source 5`, cite the first one as [1] and the second as [2].

    2.  **QUALITY FILTER:**
        * If a provided source is empty, irrelevant, or lacks specific findings, **DO NOT USE IT**. Do not force a citation just to fill a quota. Only cite sources that contribute meaningful information.

    3.  **SECONDARY SOURCES:**
        * If referencing a study cited *within* a source (e.g., Smith, 2020), state: "Smith (2020, cited in [1])..."

    ### II. FORMATTING & STYLE GUIDELINES:
    * **Tone:** Objective, formal, and academic. No conversational filler.
    * **Structure:** Use **Headings (`##`)** for themes, **Bolding** for key terms, and **Bullet Points** for lists.
    * **LaTeX:** Use `$...$` for inline math (e.g., $p < 0.05$) and `$$...$$` for block equations. Do NOT use LaTeX for simple units (e.g., write "15%", not $15\%$).

    ### III. CRITICAL NEGATIVE CONSTRAINTS:
    1.  **NO HALLUCINATIONS:** If the answer is not in the sources, do not invent it.
    2.  **NO SOURCE CONFLATION:** Keep findings distinct.
    3.  **NO META-TALK:** Do not write "The provided text says...". Start the review directly.
    4.  **NO REFERENCE LIST:** DO NOT generate a "References" section at the end.

    ### IV. ONE-SHOT EXAMPLE (EMULATE THIS STYLE):

    **Context Provided:**
    --- SOURCE 25 --- (First in list)
    Content: Method A achieves 90% accuracy.
    --- SOURCE 8 --- (Second in list)
    Content: Method B is faster but less accurate.

    **Ideal Response:**
    ## Performance Comparison
    Recent studies highlight a trade-off between accuracy and speed. Method A demonstrates superior precision, achieving **90% accuracy** [1]. In contrast, Method B prioritizes computational efficiency over raw performance [2].

    ### V. EXECUTION:
    AVAILABLE CONTEXT SOURCES:
    {context_block}
    ---

    USER QUERY:
    {user_query}

    Generate the academic review response body now.
    At the very end, add a single, short, italicized "Next Step" asking if the user wants to explore a specific aspect further. [/INST]
    """
    else:
        #Fallback prompt for non-RAG mode
        system_instruction = """You are an expert Academic Researcher.
Answer the user's question using your internal knowledge base.
Maintain a formal, objective, and scientific tone."""

    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_query}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    print(f"Generating Qwen response ({'RAG' if use_rag else 'BASELINE'})...")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=4096,
            temperature=0.2, # Low temperature for more factual/deterministic output
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return answer.strip(), contexts

def sanitize_table_cell(text):
    """Escapes pipes and removes newlines for Markdown table compatibility."""
    if not text: return "N/A"
    text = str(text).replace('\n', ' ').replace('\r', ' ').replace('|', '&#124;')
    return re.sub(r'\s+', ' ', text).strip()

def clean_and_display_report_qwen(query, use_rag=True, top_k=5):
    """
    Orchestrates the query process and displays a formatted Markdown report within Jupyter.

    It creates a split view:
    1. The AI generated text.
    2. A detailed 'Bibliography' table showing exactly which sources were used and their relevance score.
    """
    generated_text, used_sources = query_qwen(query, use_rag=use_rag, top_k=top_k)

    # Remove any AI-generated reference lists to avoid duplication with our manual table
    split_pattern = r'(?i)\n\s*(References|Bibliography|Sources|Studies cited within).*$'
    parts = re.split(split_pattern, generated_text)
    clean_text = parts[0] if parts else generated_text
    clean_text = re.sub(r'^[ \t]+', '', clean_text, flags=re.MULTILINE)

    mode_title = "RAG Augmented Response (Qwen)" if use_rag else "Baseline Response (Qwen)"
    markdown_report = f"## {mode_title}\n\n"
    markdown_report += clean_text.strip() + "\n\n"
    markdown_report += "---\n\n"

    #If no RAG or no sources returned, just show the text
    if not use_rag or not used_sources:
        display(Markdown(markdown_report))
        return

    #Identify which sources were actually cited by the LLM (e.g., [1], [3])
    cited_indices = set()
    matches = re.findall(r'\[(\d+)]', clean_text)
    for m in matches:
        cited_indices.add(int(m))

    #Group chunks by paper title to avoid listing the same paper multiple times separately
    grouped_sources = defaultdict(list)
    for i, src in enumerate(used_sources, 1):
        if i not in cited_indices:
            continue

        title = src.get('title', 'Unknown Title')
        grouped_sources[title].append({
            "id": i,
            "section": src.get('section', 'General'),
            "distance": src.get('distance'),
            "authors": src.get('authors'),
            "year": src.get("year"),
            "url": src.get("url"),
            "references": src.get('references', [])
        })

    #Construct the reference table
    if not grouped_sources:
        markdown_report += "> *No sources were directly cited in the text although RAG was active.*"
    else:
        markdown_report += "## References\n\n"

        for title, chunks in grouped_sources.items():
            first_chunk = chunks[0]
            full_authors = get_full_author_list(first_chunk['authors'])

            safe_title = sanitize_table_cell(title)
            safe_authors = sanitize_table_cell(full_authors)
            year = first_chunk['year'] or "n.d."
            url = first_chunk['url'] or "n.d."
            source_ids = ", ".join([str(c['id']) for c in chunks])

            markdown_report += f"### [Source {source_ids}] {safe_title}\n"
            markdown_report += f"**Authors:** *{full_authors}* ({year})\n"
            markdown_report += f"**Url:** *{url}*\n\n"

            markdown_report += "| Ref ID | Section Used | Key Citations Inside | Score |\n"
            markdown_report += "| :---: | :--- | :--- | :---: |\n"

            for c in chunks:
                #Convert distance to similarity score (Approximate)
                score = 1 - c['distance']
                safe_section = sanitize_table_cell(c['section'])

                inner_refs_display = "-"
                if c['references']:
                    refs_list = []
                    for r in c['references']:
                        rid = r.get('id')
                        rtext = clean_ref_text(r.get('text', ''))
                        safe_rtext = sanitize_table_cell(rtext)
                        refs_list.append(f"• [{rid}] {safe_rtext}")
                    inner_refs_display = "<br>".join(refs_list)

                markdown_report += f"| **[{c['id']}]** | {safe_section} | {inner_refs_display} | **{score:.2f}** |\n"

            markdown_report += "\n<br>\n"

    display(Markdown(markdown_report))

if __name__ == "__main__":
    #Download index and metadatas
    download_data()
    #Load systems
    init_retrieval_system()
    init_llm_system()
    
    #Run a test query
    query = "Are LLM's better GNN's"
    clean_and_display_report_qwen(query, use_rag=True, top_k=20)
import os
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Iterable
from tqdm.auto import tqdm

#Matches numeric references in brackets, e.g., "[1]" or "[1, 2, 3]"
REF_PATTERN = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")

#Matches content within parentheses that likely contains a year, e.g., "(Smith, 2020)"
PAREN_CITATION_PATTERN = re.compile(r"\(([^()]*\d{4}[^()]*)\)")

#Captures author names and year from a citation string, e.g., "Smith et al., 2020"
AUTHOR_YEAR_PATTERN = re.compile(r"(.+?),\s*(\d{4}[a-z]?)")

def extract_arxiv_info(paper_id: str, link_map: Dict[str, str]):
    """
    Retrieves the PDF URL and extracts the publication year from the arXiv ID/URL.
    
    Args:
      paper_id: The identifier of the paper (filename without extension).
      link_map: A dictionary mapping filenames to PDF URLs.
      
    Returns:
      The PDF URL and the extracted year if found.
    """
    pdf_key = f"{paper_id}.pdf"
    url = link_map.get(pdf_key)

    year = None

    if url:
        #arXiv URLs usually contain the pattern /YYMM.nnnn
        match = re.search(r"/(\d{4})\.\d+", url)
        if match:
            yymm = match.group(1)
            yy = yymm[:2]

            #Convert 2-digit year to 4-digit year
            try:
                year = int("20" + yy)
            except ValueError:
                year = None

    return url, year

def extract_numeric_reference_ids(text: str) -> List[int]:
  """
  Parses the text to find numeric citation IDs like [1], [2, 3].
  
  Args:
    text: The input text paragraph.
    
  Returns:
    A sorted list of unique reference IDs found in the text.
  """
  ids: List[int] = []
  for match in REF_PATTERN.findall(text):
    #Handle multiple citations in one bracket, e.g., "1, 2"
    parts = match.split(",")
    for p in parts:
      p = p.strip()
      if p.isdigit():
        ids.append(int(p))

  return sorted(set(ids))

def extract_author_year_citations(text: str) -> List[Dict[str, Any]]:
  """
    Parses the text to find author-year style citations like (Author, 2020).

    Args:
        text: The input text paragraph.

    Returns:
        A list of dictionaries containing raw text, authors, and year.
    """
  citations: List[Dict[str, Any]] = []

  for paren_content in PAREN_CITATION_PATTERN.findall(text):
    #Handle multiple citations seperated by semicolons
    parts = re.split(r";", paren_content)
    for part in parts:
      part = part.strip()
      if not part:
        continue

      m = AUTHOR_YEAR_PATTERN.search(part)
      if m:
        authors = m.group(1).strip()
        year = m.group(2).strip()
        citations.append({
            "raw": part,
            "authors": authors,
            "year": year,
        })

  return citations

def iter_sections(
    section: Dict[str, Any],
    section_path: List[str]
) -> Iterable[Dict[str, Any]]:
  """
    Recursively iterates through nested sections and subsections to yield paragraphs.

    Args:
        section: The current section dictionary.
        section_path: The hierarchical path of section titles leading to this section.

    Yields:
        A dictionary containing path, index, and text of a paragraph.
    """
  current_path = section_path + [section.get("title", "").strip()]

  #Yield paragraphs in the current section
  for i, para in enumerate(section.get("paragraphs", [])):
    yield {
        "section_path": current_path,
        "para_index": i,
        "text": para
    }

  #Recursively process subsections
  for subsection in section.get("subsections", []):
    yield from iter_sections(subsection, current_path)

def resolve_inline_reference(
        authors: str,
        year: str,
        reference_map: Dict[int, str]
):
    """
      Attempts to match an inline citation (Author, Year) to a reference in the bibliography.

      Args:
          authors: Author string from the text.
          year: Year string from the text.
          reference_map: The parsed bibliography map {id: full_reference_text}.

      Returns:
          The matched reference dictionary or None if not found.
    """
    authors_l = authors.lower()
    year_l = year.lower()

    #Checks if author and year exist in the reference text
    for rid, full_text in reference_map.items():
        ft_l = full_text.lower()
        if authors_l in ft_l and year_l in ft_l:
            return {"id": rid, "text": full_text}

    return None


def process_single_paper(json_path: Path, link_map: Dict[str, str]) -> List[Dict[str, Any]]:
  """
    Processes a single JSON paper file, extracting chunks and resolving references.

    Args:
        json_path: Path to the input JSON file.
        link_map: Mapping for PDF links.

    Returns:
        A list of processed chunks (dictionaries) ready for export.
    """
  with json_path.open("r", encoding="utf-8") as f:
    paper = json.load(f)

  #Header information extraction
  title = paper.get("title", "").strip()
  authors = paper.get("authors", [])
  paper_id = json_path.stem
  
  # Get URL and Year (fallback to arXiv ID extraction if metadata year is missing)
  pdf_url, extracted_year = extract_arxiv_info(paper_id, link_map)
  year = paper.get("year")
  if year is None:
    year = extracted_year

  venue = paper.get("venue")

  #Build reference map (ID -> Text)
  paper_references = paper.get("references", [])
  reference_map: Dict[int, str] = {}
  for ref in paper_references:
    try:
        rid = ref.get("id")
        rtext = ref.get("text")
        if rid is not None and rtext is not None:
            reference_map[int(rid)] = rtext
    except Exception:
        continue

  chunks: List[Dict[str, Any]] = []

  #Process abstract
  abstract_raw = paper.get("abstract") or ""
  abstract_text = abstract_raw.strip()
  
  if abstract_text:
    #Extract citations
    numeric_ref_ids = extract_numeric_reference_ids(abstract_text)
    author_year_cits = extract_author_year_citations(abstract_text)

    #Resolve citations to actual bibliography entries
    resolved_numeric = [{
        "id": rid,
        "text": reference_map.get(rid)
        }
        for rid in numeric_ref_ids
        if rid in reference_map
    ]

    resolved_inline = []
    for cit in author_year_cits:
        authors_c = cit["authors"]
        year_c = cit["year"]
        ref_match = resolve_inline_reference(authors_c, year_c, reference_map)
        #Avoid duplicates if already found via numeric ID or previous inline match
        if ref_match and ref_match not in resolved_numeric and ref_match not in resolved_inline:
            resolved_inline.append(ref_match)

    resolved_references = resolved_numeric + resolved_inline

    #Create abstract chunk
    chunk_id = f"{paper_id}_abstract_p0"
    chunks.append({
        "chunk_id": chunk_id,
        "paper_id": paper_id,
        "title": title,
        "section_title": "Abstract",
        "section_path": ["Abstract"],
        "para_index": 0,
        "text": abstract_text,
        "reference_ids": numeric_ref_ids,
        "inline_citations": author_year_cits,
        "year": year,
        "venue": venue,
        "url": pdf_url,
        "authors": authors,
        "references": resolved_references,
    })

  #Process sections and subsections
  for sec_idx, sec in enumerate(paper.get("sections", [])):
    #Use generator to flatten nested subsections
    for para_info in iter_sections(sec, []):
      section_path = para_info["section_path"]
      para_index = para_info["para_index"]
      text_raw = para_info["text"] or ""
      text = text_raw.strip()
      if not text:
        continue
      
      #Extract and resolve citations for this paragraph
      numeric_ref_ids = extract_numeric_reference_ids(text)
      author_year_cits = extract_author_year_citations(text)

      resolved_numeric = [
            {"id": rid, "text": reference_map.get(rid)}
            for rid in numeric_ref_ids
            if rid in reference_map
        ]

      resolved_inline = []
      for cit in author_year_cits:
          authors_c = cit["authors"]
          year_c = cit["year"]
          ref_match = resolve_inline_reference(authors_c, year_c, reference_map)
          if ref_match and ref_match not in resolved_numeric and ref_match not in resolved_inline:
              resolved_inline.append(ref_match)

      resolved_references = resolved_numeric + resolved_inline

      section_title = section_path[-1] if section_path else ""

      chunk_id = f"{paper_id}__sec{sec_idx}_p{para_index}"

      chunks.append({
          "chunk_id": chunk_id,
          "paper_id": paper_id,
          "title": title,
          "section_title": section_title,
          "section_path": section_path,
          "para_index": para_index,
          "text": text,
          "reference_ids": numeric_ref_ids,
          "inline_citations": author_year_cits,
          "year": year,
          "venue": venue,
          "url": pdf_url,
          "authors": authors,
          "references": resolved_references,
      })

  return chunks

def build_all_chunks(data_dir: str, links_path: str, output_path: str) -> None:
  """
    Main driver function to iterate over all JSON files and write the output JSONL.

    Args:
        data_dir: Directory containing source JSON files.
        links_path: Path to the PDF links JSON mapping file.
        output_path: Destination path for the .jsonl output file.
    """
  data_dir = Path(data_dir)
  out_path = Path(output_path)
  
  #Ensure output directory exists
  out_path.parent.mkdir(parents=True, exist_ok=True)

  print(f"Loading link mapping from: {links_path}")
  with open(links_path, "r", encoding="utf-8") as f:
    link_map = json.load(f)

  json_files = sorted(data_dir.glob("*.json"))

  with out_path.open("w", encoding="utf-8") as out_f:
    #Use tqdm for a progress bar
    for json_path in tqdm(json_files, desc="Processing papers"):
      try:
        chunks = process_single_paper(json_path, link_map)
        for ch in chunks:
          #Write each chunk as a sepearte line in jsonl format
          out_f.write(json.dumps(ch, ensure_ascii=False) + "\n")
      except Exception as e:
        #Log errors but do not stop the entire process
        print(f"Error in {json_path}: {e}")

build_all_chunks(
    data_dir="/content/drive/MyDrive/NLP/codes/data/jsons",
    links_path="/content/drive/MyDrive/NLP/codes/data/pdf_links_matching.json",
    output_path="/content/drive/MyDrive/NLP/codes/data/chunks/chunks.jsonl"
  )

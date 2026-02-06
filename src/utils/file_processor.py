import os
import re
import glob
import json
from pathlib import Path
from typing import List, Dict, Any
from docx import Document
import docx2txt


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load JSONL file; one JSON object per line. Skips blank lines and invalid lines."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return data


def save_jsonl(data: List[Dict[str, Any]], file_path: str) -> None:
    """Save list of dicts to JSONL file; one JSON object per line."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def read_text_file(file_path: str) -> str:
    """Read a plain text file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='latin-1') as f:
            return f.read()


def read_markdown_file(file_path: str) -> str:
    """Read a markdown file."""
    return read_text_file(file_path)


def read_docx_file(file_path: str) -> str:
    """Read a Word document file."""
    try:
        # Try using python-docx first
        doc = Document(file_path)
        text = []
        for paragraph in doc.paragraphs:
            text.append(paragraph.text)
        return '\n'.join(text)
    except Exception:
        # Fallback to docx2txt
        try:
            return docx2txt.process(file_path)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return ""


def get_file_content(file_path: str) -> str:
    """Read content from file based on its extension."""
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.txt':
        return read_text_file(file_path)
    elif file_ext == '.md':
        return read_markdown_file(file_path)
    elif file_ext == '.docx':
        return read_docx_file(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_ext}")


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks."""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence endings
            for i in range(end, max(start + chunk_size - 100, start), -1):
                if text[i] in '.!?':
                    end = i + 1
                    break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap
        if start >= len(text):
            break
    
    return chunks


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}]', '', text)
    return text.strip()


def get_all_files(input_dir: str) -> List[str]:
    """Get all supported files from input directory."""
    supported_extensions = ['*.txt', '*.md', '*.docx']
    files = []
    
    for ext in supported_extensions:
        files.extend(glob.glob(os.path.join(input_dir, ext)))
        files.extend(glob.glob(os.path.join(input_dir, '**', ext), recursive=True))
    
    return sorted(files)


def process_files(input_dir: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
    """Process all files in input directory and return chunks with metadata."""
    files = get_all_files(input_dir)
    processed_chunks = []
    
    for file_path in files:
        try:
            content = get_file_content(file_path)
            content = clean_text(content)
            
            if not content.strip():
                continue
            
            chunks = chunk_text(content, chunk_size, overlap)
            
            for i, chunk in enumerate(chunks):
                processed_chunks.append({
                    'content': chunk,
                    'source_file': os.path.basename(file_path),
                    'file_path': file_path,
                    'chunk_index': i,
                    'total_chunks': len(chunks)
                })
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    return processed_chunks 
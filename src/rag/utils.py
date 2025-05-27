from typing import List, Optional
import re
import os
import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
    
class TextSplitter:
    def __init__(self,
                 separators: List[str] = ["\n\n", "\n", " ", ""],
                 chunk_size: int = 1024,
                 chunk_overlap: int = 128
                 ) -> None:
        
        self.splitter = RecursiveCharacterTextSplitter(
            separators=separators,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def __call__(self, documents):
        return self.splitter.split_documents(documents)
    
class LegalDocumentSplitter:
    def __init__(self,
                 chunk_size: int = 1024,
                 chunk_overlap: int = 128,
                 section_markers: Optional[List[str]] = None,
                 ):
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.section_markers = section_markers or [
            "NỘI DUNG VỤ ÁN",
            "NHẬN ĐỊNH CỦA TÒA ÁN",
            "QUYẾT ĐỊNH",
        ]
        
        self.section_pattern = re.compile(r'(' + '|'.join(map(re.escape, self.section_markers)) + r')')
    
        self.text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", " ", ""],
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        final_chunks = []
        
        for doc in documents:
            sections = self._split_by_sections(doc.page_content)
            document_url = doc.metadata.get("source", "unknown_source")
            
            for section_idx, (section_name, section_text) in enumerate(sections):
                metadata = {
                    "source": document_url,
                    "section": section_name
                }
                
                if len(section_text) > self.chunk_size:
                    chunks = self.text_splitter.create_documents(
                        [section_text], 
                        metadatas=[metadata.copy() for _ in range(len(section_text)//self.chunk_size + 1)]
                    )
                    
                    for chunk_idx, chunk in enumerate(chunks):
                        chunk.metadata["chunk_index"] = f"{section_idx}.{chunk_idx}"
                    
                    final_chunks.extend(chunks)
                else:
                    chunk = Document(
                        page_content=section_text,
                        metadata=metadata
                    )
                    chunk.metadata["chunk_index"] = f"{section_idx}.0"
                    final_chunks.append(chunk)
        
        return final_chunks
    
    def _split_by_sections(self, text: str) -> List[tuple]:
        matches = list(self.section_pattern.finditer(text))
        
        if not matches:
            return [("DOCUMENT", text)]
        
        sections = []
        for i in range(len(matches)):
            start = matches[i].start()
            section_name = matches[i].group()
            
            end = matches[i+1].start() if i < len(matches) - 1 else len(text)
            
            section_text = text[start:end].strip()
            sections.append((section_name, section_text))
        
        if matches[0].start() > 0:
            sections.insert(0, ("THÔNG TIN VỤ ÁN", text[:matches[0].start()].strip()))
            
        return sections

    def __call__(self, documents):
        return self.split_documents(documents)

class LawDocumentSplitter:
    def __init__(self,
                 chunk_size: int = 1024,
                 chunk_overlap: int = 128,
                 section_markers: Optional[List[str]] = None,
                 ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.section_markers = section_markers or ["Điều"]
        self.article_pattern = re.compile(r'(Điều\s+\d+\.?)')
        self.text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", " ", ""],
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        final_chunks = []
        for doc in documents:
            matches = list(self.article_pattern.finditer(doc.page_content))
            sections = self._split_by_pattern(doc.page_content, matches) if matches else []

            if not sections:
                continue

            # Extract source filename without path for cleaner IDs
            source = doc.metadata.get("source", "unknown")
            source_file = os.path.basename(source) if source != "unknown" else source

            for section_idx, (section_name, section_text) in enumerate(sections):
                article_num = section_name.split()[1].strip('.')
                
                # Generate a stable unique identifier for this article section
                section_unique_id = f"{source_file}_article_{article_num}"
                
                metadata = {
                    "source": doc.metadata.get("source", "unknown_source"),
                    "section": section_name,
                    # Use a UUID hash but always produce same ID for same input
                    "doc_id": str(uuid.uuid5(uuid.NAMESPACE_DNS, section_unique_id))
                }
                
                if len(section_text) > self.chunk_size:
                    chunks = self.text_splitter.create_documents(
                        [section_text],
                        metadatas=[metadata.copy() for _ in range(len(section_text)//self.chunk_size + 1)]
                    )
                    
                    for chunk_idx, chunk in enumerate(chunks, 1):
                        chunk_idx_str = f"L.{article_num}.{chunk_idx}"
                        # Create a deterministic ID that will be the same on every run
                        chunk_unique_id = f"{section_unique_id}_{chunk_idx}"
                        chunk.metadata["doc_id"] = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_unique_id))
                        chunk.metadata["chunk_index"] = chunk_idx_str
                        self._clean_metadata(chunk.metadata)
                    
                    final_chunks.extend(chunks)
                else:
                    chunk = Document(
                        page_content=section_text,
                        metadata=metadata
                    )
                    chunk_idx_str = f"L.{article_num}.1"
                    # For single chunks, still make a deterministic ID
                    chunk_unique_id = f"{section_unique_id}_1"
                    chunk.metadata["doc_id"] = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_unique_id))
                    chunk.metadata["chunk_index"] = chunk_idx_str
                    self._clean_metadata(chunk.metadata)
                    final_chunks.append(chunk)
        
        return final_chunks
    
    def _clean_metadata(self, metadata):
        keys_to_keep = ["source", "section", "chunk_index", "doc_id"]
        keys_to_remove = [key for key in list(metadata.keys()) if key not in keys_to_keep]
        for key in keys_to_remove:
            metadata.pop(key, None)

    def _split_by_pattern(self, text: str, matches) -> List[tuple]:
        sections = []
        for i in range(len(matches)):
            start = matches[i].start()
            section_name = matches[i].group().strip()
            end = matches[i+1].start() if i < len(matches) - 1 else len(text)
            section_text = text[start:end].strip()
            sections.append((section_name, section_text))
        
        return sections

    def __call__(self, documents):
        return self.split_documents(documents)
from typing import List, Optional
import re
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
        """Split text by legal section headers."""
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
from typing import Union, List, Dict
import glob
from tqdm import tqdm
import multiprocessing
import json
import time
import bs4
import os
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.rag.utils import LegalDocumentSplitter, TextSplitter, LawDocumentSplitter

def extract_urls_from_json(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    urls = []
    for item in data:
        if 'url' in item:
            urls.append(item['url'])
    return urls

URL_DOCUMENT_CACHE = {}

def fetch_content_from_url(url, retry_count=2, backoff_factor=1.5):
    if url in URL_DOCUMENT_CACHE:
        return URL_DOCUMENT_CACHE[url]
    
    for attempt in range(retry_count + 1):
        try:
            loader = WebBaseLoader(
                web_paths=[url],
                bs_kwargs=dict(parse_only=bs4.SoupStrainer(id="vanban_content"))
            )
            documents = loader.load()
            
            for doc in documents:
                doc.metadata["source"] = url
                doc.metadata["doc_id"] = hash(url)
            
            URL_DOCUMENT_CACHE[url] = documents
            time.sleep(0.5)
            return documents
        except Exception as e:
            if attempt < retry_count:
                wait_time = backoff_factor * (2 ** attempt)
                print(f"Error fetching {url}: {e}. Retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
            else:
                print(f"Failed to fetch {url} after {retry_count+1} attempts: {e}")
                return []

def get_optimal_workers():
    cores = multiprocessing.cpu_count()
    return min(max(8, cores * 2), 32)

class BaseLoader:
    def __init__(self) -> None:
        self.num_processes = get_optimal_workers()

    def __call__(self, files: List[str], **kwargs):
        pass

class WebLoader(BaseLoader):
    def __init__(self) -> None:
        super().__init__()
        self.num_workers = get_optimal_workers()

    def __call__(self, json_files: List[str], **kwargs):
        workers = kwargs.get('workers', self.num_workers)
        all_documents = []
        all_urls = []
        
        print("Extracting URLs from JSON files...")
        for json_file in json_files:
            all_urls.extend(extract_urls_from_json(json_file))
        
        total_urls = len(all_urls)
        print(f"Found {total_urls} URLs to process")
        
        batch_size = min(max(10, workers * 2), 50)
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for i in range(0, len(all_urls), batch_size):
                batch_urls = all_urls[i:i+batch_size]
                
                future_to_url = {executor.submit(fetch_content_from_url, url): url for url in batch_urls}
                
                completed = 0
                with tqdm(total=len(batch_urls), desc=f"Batch {i//batch_size+1}/{(total_urls-1)//batch_size+1}", leave=False) as pbar:
                    for future in as_completed(future_to_url):
                        documents = future.result()
                        all_documents.extend(documents)
                        completed += 1
                        pbar.update(1)
                
                if i + batch_size < len(all_urls) and completed > 0:
                    time.sleep(min(1.0, 3.0 / completed))
        
        print(f"Total documents loaded from URLs: {len(all_documents)}")
        return all_documents

class PDFLoader(BaseLoader):
    def __init__(self) -> None:
        super().__init__()
        self.num_workers = get_optimal_workers()
        
    def __call__(self, pdf_files: List[str], **kwargs):
        workers = kwargs.get('workers', self.num_workers)
        all_documents = []
        
        print(f"Processing {len(pdf_files)} PDF files with paths:")
        for pdf_file in pdf_files:
            print(f"  - {pdf_file} (exists: {os.path.exists(pdf_file)})")
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_pdf = {executor.submit(self.process_pdf, pdf_file): pdf_file 
                             for pdf_file in pdf_files}
            
            with tqdm(total=len(pdf_files), desc="Processing PDFs") as pbar:
                for future in as_completed(future_to_pdf):
                    pdf_file = future_to_pdf[future]
                    try:
                        documents = future.result()
                        print(f"Extracted {len(documents)} pages from {pdf_file}")
                        all_documents.extend(documents)
                    except Exception as e:
                        print(f"Error processing {pdf_file}: {e}")
                    pbar.update(1)
        
        print(f"Total documents extracted from PDFs: {len(all_documents)}")
        return all_documents
    
    def process_pdf(self, pdf_file):
        try:
            if not os.path.isfile(pdf_file):
                print(f"PDF file does not exist or is not a file: {pdf_file}")
                return []
                
            print(f"Loading PDF: {pdf_file}")
            documents = []
            
            methods = [
                self._try_pypdf,
                self._try_pdfminer,
                self._try_unstructured
            ]
            
            for method in methods:
                try:
                    documents = method(pdf_file)
                    if documents:
                        break
                except Exception as e:
                    print(f"Failed with {method.__name__}: {str(e)}")
            
            if not documents:
                print(f"All PDF loading methods failed for {pdf_file}")
                return []
            
            for doc in documents:
                doc.metadata["source"] = pdf_file
                doc.metadata["doc_id"] = hash(f"{pdf_file}_{doc.metadata.get('page', 0)}")
                
                if isinstance(doc.metadata["doc_id"], int) and doc.metadata["doc_id"] < 0:
                    doc.metadata["doc_id"] = abs(doc.metadata["doc_id"])
                
                if "page" not in doc.metadata:
                    doc.metadata["page"] = 0
            
            return documents
        except Exception as e:
            print(f"Error loading PDF {pdf_file}: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
            
    def _try_pypdf(self, pdf_file):
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(pdf_file)
        return loader.load()
        
    def _try_pdfminer(self, pdf_file):
        from langchain_core.documents import Document
        
        try:
            from pdfminer.high_level import extract_text, extract_pages
            from pdfminer.layout import LTTextContainer
            
            full_text = extract_text(pdf_file)
            
            documents = []
            for i, page_layout in enumerate(extract_pages(pdf_file)):
                page_text = ""
                for element in page_layout:
                    if isinstance(element, LTTextContainer):
                        page_text += element.get_text()
                
                if not page_text.strip() and full_text:
                    parts = full_text.split('\n\n')
                    page_fraction = max(1, len(parts) // 20)
                    start_idx = i * page_fraction
                    end_idx = (i + 1) * page_fraction
                    page_text = '\n\n'.join(parts[start_idx:end_idx])
                
                if page_text.strip():
                    documents.append(Document(
                        page_content=page_text.strip(),
                        metadata={"page": i, "source": pdf_file}
                    ))
            
            if not documents and full_text:
                documents = [Document(
                    page_content=full_text,
                    metadata={"page": 0, "source": pdf_file}
                )]
                
            return documents
        except ImportError:
            from langchain_community.document_loaders import PDFMinerLoader
            loader = PDFMinerLoader(pdf_file)
            return loader.load()
        
    def _try_unstructured(self, pdf_file):
        from langchain_community.document_loaders import UnstructuredPDFLoader
        loader = UnstructuredPDFLoader(pdf_file)
        return loader.load()

class Loader:
    def __init__(self,
                 split_kwargs: dict = {
                     "chunk_size": 1000,
                     "chunk_overlap": 200
                 },
                 use_legal_splitter: bool = True) -> None:
        self.loaders = {
            "json": WebLoader(),
            "pdf": PDFLoader()
        }
        self.doc_splitters = {
            "json": LegalDocumentSplitter(**split_kwargs) if use_legal_splitter else TextSplitter(**split_kwargs),
            "pdf": LawDocumentSplitter(**split_kwargs)
        }
        self.default_splitter = TextSplitter(**split_kwargs)

    def load(self, files: Union[str, List[str]], workers: int = None):
        if isinstance(files, str):
            files = [files]
        workers = workers or get_optimal_workers()
        print(f"Loading documents using {workers} workers...")
        file_groups = self._group_files_by_extension(files)
        all_documents = []
        for ext, file_list in file_groups.items():
            if not file_list:
                continue
            if ext in self.loaders:
                print(f"Processing {len(file_list)} {ext.upper()} files...")
                docs = self.loaders[ext](file_list, workers=workers)
                for doc in docs:
                    doc.metadata["file_type"] = ext
                all_documents.extend(docs)
        json_docs = [doc for doc in all_documents if doc.metadata.get("file_type") == "json"]
        pdf_docs = [doc for doc in all_documents if doc.metadata.get("file_type") == "pdf"]
        split_documents = []
        if json_docs:
            print(f"Splitting {len(json_docs)} judgment documents...")
            split_json_docs = self.doc_splitters["json"](json_docs)
            # Ensure file_type is preserved after splitting
            for doc in split_json_docs:
                doc.metadata["file_type"] = "json"
            split_documents.extend(split_json_docs)
        if pdf_docs:
            print(f"Splitting {len(pdf_docs)} law documents...")
            split_pdf_docs = self.doc_splitters["pdf"](pdf_docs)
            # Ensure file_type is preserved after splitting
            for doc in split_pdf_docs:
                doc.metadata["file_type"] = "pdf"
            split_documents.extend(split_pdf_docs)
        print(f"Total document chunks after splitting: {len(split_documents)}")
        return split_documents

    def _group_files_by_extension(self, files: List[str]) -> Dict[str, List[str]]:
        groups = {
            "json": [],
            "pdf": []
        }
        for file_path in files:
            ext = os.path.splitext(file_path)[1].lower().strip(".")
            if ext in groups:
                groups[ext].append(file_path)
            else:
                print(f"Warning: Unsupported file type: {file_path}")
        return groups

    def load_dir(self, dir_path: str, workers: int = None):
        json_files = glob.glob(f"{dir_path}/*.json")
        pdf_files = glob.glob(f"{dir_path}/*.pdf")
        all_files = json_files + pdf_files
        
        if not all_files:
            raise ValueError(f"No supported files (JSON or PDF) found in {dir_path}")
        
        print(f"Found {len(json_files)} JSON files and {len(pdf_files)} PDF files")
        return self.load(all_files, workers=workers)
from typing import Union, List
import glob
from tqdm import tqdm
import multiprocessing
import json
import time
import bs4
import os
from langchain_community.document_loaders import WebBaseLoader
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.rag.utils import LegalDocumentSplitter, TextSplitter

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
    return min(max(8, cores * 2), 32)  # Between 8 and 32 based on cores

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
        
        return all_documents
    
class Loader:
    def __init__(self,
                 file_type: str = "json",
                 split_kwargs: dict = {
                     "chunk_size": 1000,
                     "chunk_overlap": 200
                 },
                 use_legal_splitter: bool = True) -> None:
        
        assert file_type == "json", "file_type must be json"
        self.file_type = file_type
        self.doc_loader = WebLoader()
        
        if use_legal_splitter:
            self.doc_splitter = LegalDocumentSplitter(**split_kwargs)
        else:
            self.doc_splitter = TextSplitter(**split_kwargs)

    def load(self, files: Union[str, List[str]], workers: int = None):
        if isinstance(files, str):
            files = [files]
            
        workers = workers or get_optimal_workers()
        print(f"Loading documents using {workers} workers...")
        
        doc_loaded = self.doc_loader(files, workers=workers)
        
        print(f"Splitting {len(doc_loaded)} documents...")
        doc_split = self.doc_splitter(doc_loaded)
        
        return doc_split

    def load_dir(self, dir_path: str, workers: int = None):
        files = glob.glob(f"{dir_path}/*.json")
        assert len(files) > 0, f"No JSON files found in {dir_path}"
        return self.load(files, workers=workers)
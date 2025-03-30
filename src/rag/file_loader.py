from typing import Union, List
import glob
from tqdm import tqdm
import multiprocessing
import json
import time
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def extract_urls_from_json(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    urls = []
    for item in data:
        if 'url' in item:
            urls.append(item['url'])
    return urls

def fetch_content_from_url(url):
    try:
        loader = WebBaseLoader(
            web_paths=[url],
            bs_kwargs=dict(parse_only=bs4.SoupStrainer(id="vanban_content"))
        )
        documents = loader.load()
        time.sleep(1)
        return documents
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return []

def get_num_cpu():
    return multiprocessing.cpu_count()

class BaseLoader:
    def __init__(self) -> None:
        self.num_processes = get_num_cpu()

    def __call__(self, files: List[str], **kwargs):
        pass

class WebLoader(BaseLoader):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, json_files: List[str], **kwargs):
        all_documents = []
        
        for json_file in tqdm(json_files, desc="Processing JSON files"):
            urls = extract_urls_from_json(json_file)
            
            batch_size = 10
            
            for i in range(0, len(urls), batch_size):
                batch_urls = urls[i:i+batch_size]
                
                with tqdm(total=len(batch_urls), desc=f"Fetching batch {i//batch_size+1}", leave=False) as pbar:
                    for url in batch_urls:
                        documents = fetch_content_from_url(url)
                        all_documents.extend(documents)
                        pbar.update(1)
                
                if i + batch_size < len(urls):
                    time.sleep(3)
        
        return all_documents
    
class TextSplitter:
    def __init__(self,
                 separators: List[str] = ["\n\n", "\n", " ", ""],
                 chunk_size: int = 1000,
                 chunk_overlap: int = 100
                 ) -> None:
        
        self.splitter = RecursiveCharacterTextSplitter(
            separators=separators,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def __call__(self, documents):
        return self.splitter.split_documents(documents)
    
class Loader:
    def __init__(self,
                 file_type: str = "json",
                 split_kwargs: dict = {
                     "chunk_size": 500,
                     "chunk_overlap": 50
                 }) -> None:
        
        assert file_type == "json", "file_type must be json"
        self.file_type = file_type
        self.doc_loader = WebLoader()
        self.doc_splitter = TextSplitter(**split_kwargs)

    def load(self, files: Union[str, List[str]], workers: int = 2):
        if isinstance(files, str):
            files = [files]
        doc_loaded = self.doc_loader(files, workers=workers)
        doc_split = self.doc_splitter(doc_loaded)
        return doc_split

    def load_dir(self, dir_path: str, workers: int = 2):
        files = glob.glob(f"{dir_path}/*.json")
        assert len(files) > 0, f"No JSON files found in {dir_path}"
        return self.load(files, workers=workers)
import argparse
import time
import os
import glob
from src.rag.file_loader import Loader, get_optimal_workers
from src.rag.vectorstore import VectorDB

def main():
    parser = argparse.ArgumentParser(description='Load and index legal documents')
    parser.add_argument('--data_dir', default='data_source/judgment', help='Directory containing JSON and/or PDF files')
    parser.add_argument('--reset', action='store_true', help='WARNING: Delete and recreate collection(s)')
    parser.add_argument('--upsert', action='store_true', help='Update existing documents instead of duplicating')
    parser.add_argument('--workers', type=int, default=0, help='Number of workers (0=auto)')
    parser.add_argument('--chunk_size', type=int, default=1000, help='Document chunk size')
    parser.add_argument('--chunk_overlap', type=int, default=200, help='Document chunk overlap')
    args = parser.parse_args()
    
    workers = args.workers if args.workers > 0 else get_optimal_workers()
    
    start_time = time.time()
    print(f"Loading documents from {args.data_dir} with {workers} workers...")
    
    pdf_files = glob.glob(f"{args.data_dir}/*.pdf")
    json_files = glob.glob(f"{args.data_dir}/*.json")
    
    if pdf_files:
        print(f"Found {len(pdf_files)} PDF files to process")
    
    if json_files:
        print(f"Found {len(json_files)} JSON files to process")
    
    loader = Loader(
        split_kwargs={
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap
        }
    )
    
    doc_loaded = loader.load_dir(args.data_dir, workers=workers)
    
    load_time = time.time()
    print(f"Loaded {len(doc_loaded)} document chunks in {load_time - start_time:.2f} seconds")
    
    # Separate documents by type
    judgment_docs = [doc for doc in doc_loaded if doc.metadata.get("file_type") == "json"]
    law_docs = [doc for doc in doc_loaded if doc.metadata.get("file_type") == "pdf"]
    
    print(f"Judgment documents: {len(judgment_docs)}")
    print(f"Law documents: {len(law_docs)}")
    
    # Create separate collections
    if judgment_docs:
        print(f"Indexing {len(judgment_docs)} judgment documents into 'judgment_collection'...")
        judgment_vector_db = VectorDB(
            documents=judgment_docs,
            collection_name="judgment_collection",
            reset_collection=args.reset,
            upsert=args.upsert
        )
        
    if law_docs:
        print(f"Indexing {len(law_docs)} law documents into 'law_collection'...")
        law_vector_db = VectorDB(
            documents=law_docs,
            collection_name="law_collection",
            reset_collection=args.reset,
            upsert=args.upsert
        )
    
    index_time = time.time()
    print(f"Successfully indexed documents in {index_time - load_time:.2f} seconds")
    print(f"Total processing time: {index_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
import argparse
import time
import multiprocessing
from src.rag.file_loader import Loader, get_optimal_workers
from src.rag.vectorstore import VectorDB

def main():
    parser = argparse.ArgumentParser(description='Load and index legal documents')
    parser.add_argument('--data_dir', default='data_source/judgment', help='Directory containing JSON files')
    parser.add_argument('--collection', default='judgment_collection', help='Vector DB collection name')
    parser.add_argument('--reset', action='store_true', help='WARNING: Delete and recreate collection')
    parser.add_argument('--upsert', action='store_true', help='Update existing documents instead of duplicating')
    parser.add_argument('--workers', type=int, default=0, help='Number of workers (0=auto)')
    parser.add_argument('--chunk_size', type=int, default=1000, help='Document chunk size')
    parser.add_argument('--chunk_overlap', type=int, default=200, help='Document chunk overlap')
    args = parser.parse_args()
    
    workers = args.workers if args.workers > 0 else get_optimal_workers()
    
    start_time = time.time()
    print(f"Loading documents from {args.data_dir} with {workers} workers...")
    
    loader = Loader(
        file_type="json", 
        split_kwargs={
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap
        }
    )
    
    doc_loaded = loader.load_dir(args.data_dir, workers=workers)
    
    load_time = time.time()
    print(f"Loaded {len(doc_loaded)} document chunks in {load_time - start_time:.2f} seconds")
    
    print(f"Indexing documents into collection '{args.collection}'...")
    print(f"Mode: {'RESET & CREATE NEW' if args.reset else ('UPSERT' if args.upsert else 'ADD NEW ONLY')}")
    
    vector_db = VectorDB(
        documents=doc_loaded, 
        collection_name=args.collection,
        reset_collection=args.reset,
        upsert=args.upsert
    )
    
    index_time = time.time()
    print(f"Successfully indexed {len(doc_loaded)} chunks in {index_time - load_time:.2f} seconds")
    print(f"Total processing time: {index_time - start_time:.2f} seconds")
    print("To avoid duplicates in the future, always use the --upsert flag")

if __name__ == "__main__":
    main()
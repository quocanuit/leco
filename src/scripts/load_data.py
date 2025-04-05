import argparse
from src.rag.file_loader import Loader
from src.rag.vectorstore import VectorDB

def main():
    parser = argparse.ArgumentParser(description='Load and index legal documents')
    parser.add_argument('--data_dir', required=True, help='Directory containing JSON files')
    parser.add_argument('--collection', default='judgment_collection', help='Vector DB collection name')
    parser.add_argument('--reset', action='store_true', help='Delete and recreate collection')
    args = parser.parse_args()

    print(f"Loading documents from {args.data_dir}...")
    loader = Loader(file_type="json")
    doc_loaded = loader.load_dir(args.data_dir, workers=2)
    print(f"Loaded {len(doc_loaded)} document chunks")
    
    print(f"Indexing documents into collection '{args.collection}'...")
    vector_db = VectorDB(
        documents=doc_loaded, 
        collection_name=args.collection,
        reset_collection=args.reset
    )
    
    print(f"Successfully indexed {len(doc_loaded)} chunks")

if __name__ == "__main__":
    main()
from langchain_qdrant import Qdrant, QdrantVectorStore
from qdrant_client import QdrantClient, models
from langchain_huggingface import HuggingFaceEmbeddings
import os
import uuid
import hashlib
from dotenv import load_dotenv

load_dotenv()

class VectorDB:
    def __init__(self,
                documents=None,
                vector_db=QdrantVectorStore,
                embedding=HuggingFaceEmbeddings(model_name="BAAI/bge-m3"),
                collection_name="judgment_collection",
                location=os.getenv("VECTOR_DB_URL"),
                client=None,
                reset_collection=False,
                upsert=True
            ) -> None:

        self.vector_db = vector_db
        self.embedding = embedding
        self.collection_name = collection_name
        self.location = location
        self.client = client if client else QdrantClient(url=location)
        self.upsert = upsert
        
        if reset_collection:
            try:
                self.client.delete_collection(collection_name)
                print(f"Collection {collection_name} deleted")
            except Exception as e:
                print(f"Error deleting collection: {e}")
        
        self.db = self._build_db(documents)

    def get_document_id(self, doc):
        if "doc_id" in doc.metadata and doc.metadata["doc_id"]:
            return doc.metadata["doc_id"]
            
        if "source" in doc.metadata and doc.metadata["source"]:
            source = doc.metadata["source"]
            content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()[:8]
            unique_str = f"{source}_{content_hash}"
        else:
            unique_str = doc.page_content
            
        doc_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, unique_str))
        
        doc.metadata["doc_id"] = doc_id
        return doc_id

    def _build_db(self, documents):
        if documents is None:
            db = self.vector_db(
                client=self.client,
                collection_name=self.collection_name,
                embedding=self.embedding 
            )
            return db
            
        print("Assigning document IDs...")
        for doc in documents:
            self.get_document_id(doc)
            
        collections = self.client.get_collections()
        collection_exists = any(col.name == self.collection_name for col in collections.collections)
        
        count = 0
        if collection_exists:
            count = self.client.count(collection_name=self.collection_name).count
            print(f"Collection '{self.collection_name}' exists with {count} points")
        
        if not collection_exists:
            print(f"Creating new collection '{self.collection_name}'")
            sample_embedding = self.embedding.embed_query("Sample text")
            vector_size = len(sample_embedding)
            
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE
                )
            )
            
        if self.upsert and collection_exists:
            print(f"Upserting documents to existing collection '{self.collection_name}'")
            
            batch_size = 50
            total_processed = 0
            skip_count = 0
            
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i+batch_size]
                print(f"Processing batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1} with {len(batch)} documents")
                
                ids = [doc.metadata["doc_id"] for doc in batch]
                
                existing_ids = []
                for doc_id in ids:
                    try:
                        point = self.client.retrieve(
                            collection_name=self.collection_name,
                            ids=[doc_id]
                        )
                        if point and len(point) > 0:
                            existing_ids.append(doc_id)
                    except Exception:
                        pass
                
                new_batch = []
                for doc in batch:
                    if doc.metadata["doc_id"] in existing_ids:
                        skip_count += 1
                    else:
                        new_batch.append(doc)
                
                if not new_batch:
                    print(f"Skipping batch - all {len(batch)} documents already exist")
                    continue
                    
                print(f"Found {len(new_batch)} new documents to insert")
                
                texts = [doc.page_content for doc in new_batch]
                embeddings = self.embedding.embed_documents(texts)
                
                points = []
                for doc, embedding in zip(new_batch, embeddings):
                    doc_id = doc.metadata["doc_id"]
                    points.append({
                        "id": doc_id,
                        "vector": embedding,
                        "payload": {
                            "page_content": doc.page_content,
                            "metadata": doc.metadata
                        }
                    })
                
                if points:
                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=points
                    )
                    total_processed += len(points)
            
            print(f"Skipped {skip_count} existing documents")
            print(f"Inserted {total_processed} new documents")
            
            new_count = self.client.count(collection_name=self.collection_name).count
            print(f"Collection now has {new_count} points (was {count})")
            
            db = self.vector_db(
                client=self.client,
                collection_name=self.collection_name,
                embedding=self.embedding
            )
            return db
        
        print(f"Adding documents to collection '{self.collection_name}' with IDs")
        
        texts = [doc.page_content for doc in documents]
        embeddings = self.embedding.embed_documents(texts)
        
        points = []
        for doc, embedding in zip(documents, embeddings):
            doc_id = doc.metadata["doc_id"]
            points.append({
                "id": doc_id,
                "vector": embedding,
                "payload": {
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                }
            })
            
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        db = self.vector_db(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embedding
        )
        return db
    
    def search(self, query, k=5):
        return self.db.similarity_search(query, k=k)
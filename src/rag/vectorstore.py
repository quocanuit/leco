from langchain_qdrant import QdrantVectorStore
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

    def get_document_ids(self, docs):
        for doc in docs:
            if "doc_id" in doc.metadata and doc.metadata["doc_id"]:
                continue
                
            if "source" in doc.metadata and doc.metadata["source"]:
                source = doc.metadata["source"]
                content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()[:8]
                unique_str = f"{source}_{content_hash}"
            else:
                unique_str = doc.page_content
                
            doc_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, unique_str))
            doc.metadata["doc_id"] = doc_id
        
        return [doc.metadata["doc_id"] for doc in docs]

    def _build_db(self, documents):
        if documents is None:
            db = self.vector_db(
                client=self.client,
                collection_name=self.collection_name,
                embedding=self.embedding 
            )
            return db
            
        print("Assigning document IDs...")
        doc_ids = self.get_document_ids(documents)
            
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
            
            batch_size = min(max(20, len(documents) // 20), 200)
            total_processed = 0
            skip_count = 0
            
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i+batch_size]
                batch_ids = [doc.metadata["doc_id"] for doc in batch]
                
                print(f"Processing batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1} with {len(batch)} documents")
                
                try:
                    existing_points = self.client.retrieve(
                        collection_name=self.collection_name,
                        ids=batch_ids,
                        with_payload=False,
                        with_vectors=False
                    )
                    existing_ids = [point.id for point in existing_points]
                except Exception as e:
                    print(f"Error retrieving existing points: {e}")
                    existing_ids = []
                
                new_batch = [doc for doc in batch if doc.metadata["doc_id"] not in existing_ids]
                skip_count += len(batch) - len(new_batch)
                
                if not new_batch:
                    print(f"Skipping batch - all {len(batch)} documents already exist")
                    continue
                    
                print(f"Found {len(new_batch)} new documents to insert")
                
                embed_batch_size = min(50, len(new_batch))
                all_points = []
                
                for j in range(0, len(new_batch), embed_batch_size):
                    sub_batch = new_batch[j:j+embed_batch_size]
                    texts = [doc.page_content for doc in sub_batch]
                    
                    print(f"Generating embeddings for {len(texts)} documents")
                    embeddings = self.embedding.embed_documents(texts)
                    
                    for doc, embedding in zip(sub_batch, embeddings):
                        doc_id = doc.metadata["doc_id"]
                        all_points.append({
                            "id": doc_id,
                            "vector": embedding,
                            "payload": {
                                "page_content": doc.page_content,
                                "metadata": doc.metadata
                            }
                        })
                
                if all_points:
                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=all_points
                    )
                    total_processed += len(all_points)
            
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
        
        optimal_batch = min(max(50, len(documents) // 10), 200)
        
        for i in range(0, len(documents), optimal_batch):
            batch = documents[i:i+optimal_batch]
            print(f"Processing batch {i//optimal_batch + 1}/{(len(documents)-1)//optimal_batch + 1}")
            
            texts = [doc.page_content for doc in batch]
            print(f"Generating embeddings for {len(texts)} documents")
            embeddings = self.embedding.embed_documents(texts)
            
            points = []
            for doc, embedding in zip(batch, embeddings):
                doc_id = doc.metadata["doc_id"]
                points.append({
                    "id": doc_id,
                    "vector": embedding,
                    "payload": {
                        "page_content": doc.page_content,
                        "metadata": doc.metadata
                    }
                })
                
            print(f"Upserting {len(points)} points to collection")
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
        
    def get_retriever(self, search_kwargs=None):
        if search_kwargs is None:
            search_kwargs = {"k": 5}
            
        return self.db.as_retriever(search_kwargs=search_kwargs)
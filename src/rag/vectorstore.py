from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

class VectorDB:
    def __init__(self,
                documents=None,
                vector_db=Qdrant,
                embedding=HuggingFaceEmbeddings(model_name="BAAI/bge-m3"),
                collection_name="default_collection",
                location=os.getenv("VECTOR_DB_URL")
            ) -> None:

        self.vector_db = vector_db
        self.embedding = embedding
        self.collection_name = collection_name
        self.location = location
        self.db = self._build_db(documents)

    def _build_db(self, documents):
        db = self.vector_db.from_documents(
                documents=documents,
                embedding=self.embedding,
                collection_name=self.collection_name,
                location=self.location
            )
        
        return db

    def get_retriever(self,
                      search_type: str = "similarity",
                      search_kwargs: dict = {'k': 3}
                      ):
        retriever = self.db.as_retriever(search_type=search_type,
                                         search_kwargs=search_kwargs)
        return retriever
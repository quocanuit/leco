from langchain_community.vectorstores import Qdrant
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv

load_dotenv()

class VectorDB:
    def __init__(self,
                documents=None,
                vector_db=Qdrant,
                embedding=None,
                collection_name="default_collection",
                location=os.getenv("VECTOR_DB_URL")
            ) -> None:

        self.embedding = embedding or GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GEMINI_API_KEY")
        )
        self.vector_db = vector_db
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

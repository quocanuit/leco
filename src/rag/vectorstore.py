# from typing import Union
# from langchain_community.vectorstores import Chroma, FAISS
# from langchain_huggingface import HuggingFaceEmbeddings

# class VectorDB:
#     def __init__(self,
#                 documents=None,
#                 vector_db: Union[Chroma, FAISS] = Chroma,
#                 embedding=HuggingFaceEmbeddings(),
#             ) -> None:

#         self.vector_db = vector_db
#         self.embedding = embedding
#         self.db = self._build_db(documents)

#     def _build_db(self, documents):
#         db = self.vector_db.from_documents(documents=documents,
#                                            embedding=self.embedding)
#         return db

#     def get_retriever(self,
#                       search_type: str = "similarity",
#                       search_kwargs: dict = {'k': 10}
#                       ):
#         retriever = self.db.as_retriever(search_type=search_type,
#                                          search_kwargs=search_kwargs)
#         return retriever

from langchain_community.vectorstores import Qdrant
# from qdrant_client import QdrantClient
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

class VectorDB:
    def __init__(self,
                documents=None,
                vector_db=Qdrant,
                embedding=HuggingFaceBgeEmbeddings(model_name="BAAI/bge-m3"),
                collection_name="default_collection",
                location=":memory:"
            ) -> None:

        self.vector_db = vector_db
        self.embedding = embedding
        self.collection_name = collection_name
        self.location = location
        self.db = self._build_db(documents)

    def _build_db(self, documents):
        if self.vector_db == Qdrant:
            db = self.vector_db.from_documents(
                documents=documents,
                embedding=self.embedding,
                collection_name=self.collection_name,
                location=self.location
            )
        else:
            db = self.vector_db.from_documents(
                documents=documents,
                embedding=self.embedding
            )
        return db

    def get_retriever(self,
                      search_type: str = "similarity",
                      search_kwargs: dict = {'k': 10}
                      ):
        retriever = self.db.as_retriever(search_type=search_type,
                                         search_kwargs=search_kwargs)
        return retriever
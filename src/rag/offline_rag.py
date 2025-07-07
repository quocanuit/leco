import re
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os

class Str_OutputParser(StrOutputParser):
    def __init__(self) -> None:
        super().__init__()

    def parse(self, text: str) -> str:
        return self.extract_answer(text)

    def extract_answer(self,
                       text_response: str,
                       pattern: str = r"Trả lời:\s*(.*)"
                       ) -> str:

        match = re.search(pattern, text_response, re.DOTALL)
        if match:
            answer_text = match.group(1).strip()
            return answer_text
        else:
            return text_response


class Offline_RAG:
    def __init__(self, llm) -> None:
        self.llm = llm
        self.prompt = PromptTemplate(
            input_variables=["context", "question", "chat_history"],
            template=self.load_prompt_template("prompt.txt")
        )
        self.str_parser = Str_OutputParser()

    def get_chain(self, retriever):
        input_data = {
            "context": lambda x: self.format_docs(retriever.invoke(x["question"])),
            "question": lambda x: x["question"],
            "chat_history": lambda x: x.get("chat_history", "")
        }

        rag_chain = (
            input_data
            | self.prompt
            | self.llm
            | self.str_parser
        )
        return rag_chain

    def get_chain(self):
        from src.rag.vectorstore import VectorDB
        
        def dynamic_retrieval_chain(inputs):
            source_type = inputs.get("source_type", "judgment")
            question = inputs["question"]
            chat_history = inputs.get("chat_history", "")
            
            if source_type == "law":
                collection_name = "law_collection"
            else:
                collection_name = "judgment_collection"
            
            retriever = VectorDB(collection_name=collection_name).get_retriever()
            context = self.format_docs(retriever.invoke(question), source_type=source_type)
            
            formatted_inputs = {
                "context": context,
                "question": question,
                "chat_history": chat_history
            }
            
            response = self.prompt.format(**formatted_inputs)
            llm_response = self.llm.invoke(response)
            
            return self.str_parser.parse(llm_response.content if hasattr(llm_response, 'content') else str(llm_response))
        
        return dynamic_retrieval_chain

    def format_docs(self, docs, source_type=None):
        sorted_docs = sorted(docs, key=self._get_sort_key)
        formatted_docs = []

        for doc in sorted_docs:
            chunk_index = doc.metadata.get("chunk_index", "")
            section = doc.metadata.get("section", "")
            source = doc.metadata.get("source", "")

            if source_type == "judgment" and source:
                header = f"[BẢN ÁN: {source}]"
            elif section and chunk_index:
                header = f"[{section} - {chunk_index}]"
            else:
                header = ""
            
            formatted_content = f"{header}\n{doc.page_content}" if header else doc.page_content
            formatted_docs.append(formatted_content)

        return "\n\n".join(formatted_docs)
        
    def _get_sort_key(self, doc):
        chunk_index = doc.metadata.get("chunk_index", "")
        if not chunk_index:
            return (float('inf'), float('inf'))
        
        try:
            parts = chunk_index.split('.')
            if len(parts) >= 3:
                if parts[0] in ['L', 'J']:
                    return (int(parts[1]), int(parts[2]))
            elif len(parts) >= 2:
                return (int(parts[0]), int(parts[1]))
            return (float('inf'), float('inf'))
        except (ValueError, IndexError):
            return (float('inf'), float('inf'))

    def load_prompt_template(self, filename):
        path = os.path.join(os.path.dirname(__file__), filename)
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
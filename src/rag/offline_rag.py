# import re
# from langchain.prompts import PromptTemplate
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser
# import os

# class Str_OutputParser(StrOutputParser):
#     def __init__(self) -> None:
#         super().__init__()

#     def parse(self, text: str) -> str:
#         return self.extract_answer(text)
    
#     def extract_answer(self,
#                        text_response: str,
#                        pattern: str = r"Trả lời:\s*(.*)"
#                        ) -> str:
        
#         match = re.search(pattern, text_response, re.DOTALL)
#         if match:
#             answer_text = match.group(1).strip()
#             return answer_text
#         else:
#             return text_response
        
# class Offline_RAG:
#     def __init__(self, llm) -> None:
#         self.llm = llm
#         self.prompt = PromptTemplate(
#             input_variables=["context", "question"],
#             template=self.load_prompt_template("prompt.txt")
#         )
#         self.str_parser = Str_OutputParser()

#     def get_chain(self, retriever):
#         input_data = {
#             "context": retriever | self.format_docs,
#             "question": RunnablePassthrough()
#         }
#         rag_chain = (
#             input_data
#             | self.prompt
#             | self.llm  
#             | self.str_parser
#         )
#         return rag_chain
    
#     def format_docs(self, docs):
#         return "\n\n".join(doc.page_content for doc in docs)
    
#     def load_prompt_template(self, filename):
#         path = os.path.join(os.path.dirname(__file__), filename)
#         with open(path, "r", encoding="utf-8") as f:
#             return f.read()










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

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def load_prompt_template(self, filename):
        path = os.path.join(os.path.dirname(__file__), filename)
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
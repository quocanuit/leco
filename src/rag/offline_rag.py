import re
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

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
        # self.prompt = hub.pull("rlm/rag-prompt")
        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
Bạn là một trợ lý tư vấn pháp lý, tên của bạn là LECO. Sử dụng ngữ cảnh được cung cấp dưới đây để trả lời câu hỏi. Nếu bạn không biết câu trả lời, hãy nói rằng bạn không biết. Trả lời bằng tiếng Việt và giữ câu trả lời ngắn gọn, phải trả lời thật chính xác với mục đích câu hỏi.
Lưu ý: Nếu ngữ cảnh không liên quan đến câu hỏi, hãy bỏ qua ngữ cảnh và trả lời dựa trên tình huống đó.
Câu hỏi: {question}
Ngữ cảnh: {context}
Trả lời:
"""
        )
        self.str_parser = Str_OutputParser()

    def get_chain(self, retriever):
        input_data = {
            "context": retriever | self.format_docs,
            "question": RunnablePassthrough()
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
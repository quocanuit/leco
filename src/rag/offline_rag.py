import re
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os

class Str_OutputParser(StrOutputParser):
    def __init__(self) -> None:
        super().__init__()
        self._debug_info = {}

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
            
    def get_debug_info(self):
        return self._debug_info
        
    def set_debug_info(self, key, value):
        self._debug_info[key] = value
        
class Offline_RAG:
    def __init__(self, llm) -> None:
        self.llm = llm
        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=self.load_prompt_template("prompt.txt")
        )
        self.str_parser = Str_OutputParser()

    def get_chain(self, retriever):
        retriever_chain = retriever | self.format_docs
        
        def combine_inputs(input_data):
            if isinstance(input_data, dict) and "question" in input_data:
                question = input_data["question"]
                debug_mode = input_data.get("debug", False)
            else:
                question = input_data
                debug_mode = False
            
            doc_content = retriever_chain.invoke(question)
            return {"context": doc_content, "question": question, "debug": debug_mode}
        
        def process_prompt(inputs):
            formatted_prompt = self.prompt.format(
                context=inputs["context"], 
                question=inputs["question"]
            )
            
            if inputs.get("debug", False):
                self.str_parser.set_debug_info("prompt", formatted_prompt)
                self.str_parser.set_debug_info("context", inputs["context"])
                
            return formatted_prompt
        
        rag_chain = (
            RunnablePassthrough()
            | combine_inputs
            | process_prompt
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
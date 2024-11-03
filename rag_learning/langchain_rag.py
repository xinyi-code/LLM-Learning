"""
# File       : VectorBase.py
# Time       ：2024/10/20 23:25
# Author     ：xinyi
# Description：
"""

import os
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage, ChatMessage
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredWordDocumentLoader, CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import Dict, List, Optional, Tuple, Union

import warnings

warnings.filterwarnings("ignore")

print("------------------ 1.文档分割-----------------")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
documents = CSVLoader(file_path="../data/test_data.csv", encoding="utf-8").load()
docs = text_splitter.split_documents(documents)

print("------------------ 2.向量库加载-----------------")
EmbeddingModel = HuggingFaceEmbeddings(model_name="D:/MyProject/TorchProject/NLP/model_hub/bge-small-zh-v1.5",
                                       model_kwargs={'device': 'cpu'},
                                       encode_kwargs={'normalize_embeddings': True})
vector_save_path = 'VectorStores/test_storage'
if not os.path.exists(vector_save_path):
    vector = FAISS.from_documents(docs, EmbeddingModel)
    vector.save_local(vector_save_path)
else:
    vector = FAISS.load_local(folder_path=vector_save_path, embeddings=EmbeddingModel,
                              allow_dangerous_deserialization=True)

print("------------------ 3.LLM Chat Model定义-----------------")
PROMPT_TEMPLATE = dict(
    RAG_PROMPT_TEMPALTE="""结合以上下文来回答用户的问题。
        问题: {question}
        可参考的上下文：
        ···
        {context}
        ···
        如果给定的上下文无法让你做出回答，请回答数据库中没有这个内容，不要臆想推测，请使用中文回答。
        回答:""",
)


class MyChatOpenAI():
    def __init__(self, model: str = "Qwen2.5-7B-Instruct",
                 api_key: str = "OPENAI_API_KEY",
                 base_url: str = "http://localhost:6006/v1",
                 temperature: float = 0.1) -> None:
        super().__init__()
        self.client = ChatOpenAI(
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            model_name=model
        )

    def llm_chat(self, prompt: str, history: List[dict], content: str) -> str:
        history.append(
            HumanMessage(content=PROMPT_TEMPLATE['RAG_PROMPT_TEMPALTE'].format(question=prompt, context=content))
        )
        response = self.client(messages=history)
        return response.content


print("------------------ 4.RAG问答-----------------")
query = '为什么人在剧烈活动后不能马上停下来'
contents = vector.similarity_search_with_score(query, k=3)
print(f"检索结果： {contents}")
context = [c[0].page_content for c in contents]
print(f"检索结果文本： {context}")

ChatModel = MyChatOpenAI()
llm_answer = ChatModel.llm_chat(query, [], context[0])
print(f"llm_answer: {llm_answer}")

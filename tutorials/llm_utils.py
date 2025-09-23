#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/9/23 17:17
# @Author  : xinyi
# @File    : llm_utils.py
# @Description    :


from typing import List

import requests
from typing import List, Dict, Optional
from langchain.embeddings.base import Embeddings

class SiliconFlowEmbedding(Embeddings):
    """硅基流动Qwen嵌入模型的LangChain封装"""

    def __init__(self, api_key: str, emb_model: str, emb_api_url:str):
        self.api_key = api_key
        self.emb_model = emb_model
        self.emb_api_url = emb_api_url

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """为文档列表生成嵌入向量"""

        payload = {
            "model": self.emb_model,
            "input": texts,
            "encoding_format": "float"
        }

        response = requests.post(self.emb_api_url, headers=self.headers, json=payload)
        response.raise_for_status()

        result = response.json()
        return [item['embedding'] for item in result['data']]

    def embed_query(self, text: str) -> List[float]:
        """为查询文本生成嵌入向量"""
        return self.embed_documents([text])[0]

class SiliconFlowChat():
    """硅基流动LLM模型的LangChain封装"""

    def __init__(self, api_key: str, chat_model: str, chat_api_url: str):
        self.api_key = api_key
        self.chat_model = chat_model
        self.chat_api_url = chat_api_url

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    def chat_completion(self,
                        messages: List[Dict],
                        temperature: float = 0.0,
                        max_tokens: int = 2048,
                        stream: bool = False) -> Optional[str]:
        """
        与Qwen Instruct模型对话:cite[6]
        """
        payload = {
            "model": self.chat_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream
        }

        try:
            response = requests.post(f"{self.chat_api_url}/chat/completions",
                                     headers=self.headers,
                                     json=payload)
            response.raise_for_status()

            result = response.json()
            return result['choices'][0]['message']['content']

        except requests.exceptions.RequestException as e:
            print(f"聊天API请求错误: {e}")
            return None

# 使用示例
if __name__ == "__main__":
    from config import config
    # 测试嵌入模型
    emb_client = SiliconFlowEmbedding(api_key=config.llm_api_key,
                                               emb_model= config.emb_model,
                                               emb_api_url=config.emb_api_base)

    test_texts = ["这是一个测试文档", "这是另一个测试文档"]
    embeddings = emb_client.embed_documents(test_texts)
    print(f"生成 {len(embeddings)} 个嵌入向量，每个维度为 {len(embeddings[0])}")

    # 测试对话功能
    chat_client = SiliconFlowChat(api_key=config.llm_api_key,
                                               chat_model= config.llm_model,
                                               chat_api_url= config.llm_api_base
                                  )
    messages = [
        {"role": "system", "content": "你是一个专业的AI助手。"},
        {"role": "user", "content": "讲一个冷效果"}
    ]

    response = chat_client.chat_completion(messages)
    if response:
        print("AI回复:", response)
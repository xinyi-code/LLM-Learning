#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/9/23 19:50
# @Author  : xinyi
# @File    : test.py
# @Description    :

from config import config
from llm_utils import SiliconFlowEmbedding, SiliconFlowChat

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
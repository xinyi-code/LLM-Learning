"""
#!/usr/bin/env python
# -*- coding:utf-8 -*-
@Author: xinyi
@file:vllm_langchain.py
@date: 2024/11/27 16:39
@description: langchain下的ChatOpenAI， 实现批量调用
"""

from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = ChatOpenAI(
    model_name = 'Qwen2.5-3B-Instruct',
    openai_api_key=openai_api_key,
    openai_api_base=openai_api_base,
)

chat = client.generate(
    [
        [SystemMessage(content='you are a helpful assistant'),
         HumanMessage(content='hello, 你是谁？')
        ],
        [SystemMessage(content='you are a helpful assistant'),
         HumanMessage(content='你能做什么？')]
    ]
)
print(chat)
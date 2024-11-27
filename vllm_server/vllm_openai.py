"""
#!/usr/bin/env python
# -*- coding:utf-8 -*-
@Author: xinyi
@file:vllm_server.py
@date: 2024/11/20 19:24
@description:  OpenAI接口单次请求
@refer: https://qwen.readthedocs.io/zh-cn/latest/deployment/vllm.html#multi-gpu-distributed-serving
"""

from openai import OpenAI
# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

chat_response = client.chat.completions.create(
    model="Qwen2.5-3B-Instruct",
    messages=[
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": "Tell me something about what's you name."},
    ],
    temperature=0.7,
    top_p=0.8,
    max_tokens=512,
    extra_body={
        "repetition_penalty": 1.05,
    },
)
print("Chat response:", chat_response)
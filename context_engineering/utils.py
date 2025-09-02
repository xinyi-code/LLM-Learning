#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/9/1 20:02
# @Author  : xinyi
# @File    : utils.py
# @Description    :

import base64
import urllib.parse
from langchain.schema import HumanMessage
from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings

SILICONFLOW_API_KEY = "sk-mjlmettgpyvcjzadwauiesmzukmkfkrgxtlpbgcotzbtgicj"

llm = init_chat_model(
    model="Qwen/Qwen2.5-14B-Instruct",
    temperature=0,
    model_provider="openai",
    model_kwargs={
        "openai_api_key": SILICONFLOW_API_KEY,
        "openai_api_base": "https://api.siliconflow.cn/v1"
    }
)
# ------------------- test
# messages = [HumanMessage(content="您好， 简单介绍下自己？")]
# response = llm.invoke(messages)
# print(f"Response: {response.content}")

# embeddings = init_embeddings("openai:text-embedding-3-small")
# 直接使用 OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="BAAI/bge-m3",
                        openai_api_key = SILICONFLOW_API_KEY,
                        openai_api_base = "https://api.siliconflow.cn/v1")
# 批量生成嵌入
# texts = [
#     "硅基流动嵌入服务",
#     "高质量的中文嵌入模型",
#     "快速且价格合理的API服务"
# ]
# batch_embeddings = embeddings.embed_documents(texts)
# print(f"批量生成 {len(batch_embeddings)} 个嵌入")
# print(f"每个嵌入维度: {len(batch_embeddings[0])}")

def save_png(data, path):
    """ save image
    :param data:  data
    :param path: save path
    """
    encoded_data = base64.b64encode(data).decode('utf-8')
    with open(path, "wb") as f:
        f.write(base64.b64decode(encoded_data))


from typing import TypedDict


# Define a graph state with two fields
class State(TypedDict):
    """State schema for the joke generator workflow.

    Attributes:
        topic: The topic for joke generation
        joke: The generated joke content
    """
    topic: str
    joke: str


from rich.console import Console
from rich.panel import Panel
import json

console = Console()


def format_message_content(message):
    """Convert message content to displayable string"""
    if isinstance(message.content, str):
        return message.content
    elif isinstance(message.content, list):
        # Handle complex content like tool calls
        parts = []
        for item in message.content:
            if item.get('type') == 'text':
                parts.append(item['text'])
            elif item.get('type') == 'tool_use':
                parts.append(f"\n🔧 Tool Call: {item['name']}")
                parts.append(f"   Args: {json.dumps(item['input'], indent=2)}")
        return "\n".join(parts)
    else:
        return str(message.content)


def format_messages(messages):
    """Format and display a list of messages with Rich formatting"""
    for m in messages:
        msg_type = m.__class__.__name__.replace('Message', '')
        content = format_message_content(m)

        if msg_type == 'Human':
            console.print(Panel(content, title="🧑 Human", border_style="blue"))
        elif msg_type == 'Ai':
            console.print(Panel(content, title="🤖 Assistant", border_style="green"))
        elif msg_type == 'Tool':
            console.print(Panel(content, title="🔧 Tool Output", border_style="yellow"))
        else:
            console.print(Panel(content, title=f"📝 {msg_type}", border_style="white"))


def format_message(messages):
    """Alias for format_messages for backward compatibility"""
    return format_messages(messages)

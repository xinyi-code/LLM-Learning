"""
# File       : LLM.py
# Time       ：2024/10/15 23:25
# Author     ：xinyi
# Description：大模型的调用接口
"""

import os
from openai import OpenAI
from zhipuai import ZhipuAI

from typing import Dict, List, Optional, Tuple, Union

PROMPT_TEMPLATE = dict(
    RAG_PROMPT_TEMPALTE="""结合以上下文来回答用户的问题。
        输入问题: {question}
        可参考的上下文：
        ```
        {context}
        ```
        如果给定的上下文无法让你做出回答，请回答数据库中没有这个内容，不要臆想推测，请使用中文回答。
        输出回答:"""
)


class BaseModel:
    def __init__(self) -> None:
        pass

    def chat(self, prompt: str, history: List[dict], content: str) -> str:
        pass

    def load_model(self):
        pass


class OpenChatModel(BaseModel):
    """ OpenAI Model and GLM Model API
    """

    def __init__(self, model_name: str = "gpt-3.5-turbo-1106") -> None:
        super().__init__()
        self.model_name = model_name
        if 'gpt' in self.model_name:
            self.chat_model = OpenAI()
            self.chat_model.api_key = os.getenv("OPENAI_API_KEY")
            self.chat_model.base_url = os.getenv("OPENAI_BASE_URL")
        elif 'glm' in self.model_name:
            self.chat_model = ZhipuAI(api_key=os.getenv("ZHIPUAI_API_KEY"))
        else:
            print("check model name please.")

    def chat(self, prompt: str, history: List[dict], content: str) -> str:
        history.append({'role': 'user',
                        'content': PROMPT_TEMPLATE['RAG_PROMPT_TEMPALTE'].format(question=prompt, context=content)})
        response = self.chat_model.chat.completions.create(
            model=self.model_name,
            messages=history,
            max_tokens=150,
            temperature=0.1
        )
        return response.choices[0].message.content


class DashscopeChat(BaseModel):
    """
    Qwen Model API
    """

    def __init__(self, path: str = '', model: str = "qwen-turbo") -> None:
        super().__init__(path)
        self.model = model

    def chat(self, prompt: str, history: List[Dict], content: str) -> str:
        import dashscope
        dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
        history.append({'role': 'user',
                        'content': PROMPT_TEMPLATE['RAG_PROMPT_TEMPALTE'].format(question=prompt, context=content)})
        response = dashscope.Generation.call(
            model=self.model,
            messages=history,
            result_format='message',
            max_tokens=150,
            temperature=0.1
        )
        return response.output.choices[0].message.content

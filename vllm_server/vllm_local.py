"""
#!/usr/bin/env python
# -*- coding:utf-8 -*-
@Author: xyshu
@file:vllm_local.py
@date: 2024/11/20 19:34
@description: 本地内存加载与调用
"""

from vllm import LLM, SamplingParams
from transformers import AutoProcessor
import os
import json

# 自动下载模型时，指定使用modelscope; 否则，会从HuggingFace下载
os.environ['VLLM_USE_MODELSCOPE'] = 'True'


def get_completion(prompts, model, tokenizer=None, max_tokens=512, temperature=0.8, top_p=0.95, max_model_len=2048):
    stop_token_ids = [151329, 151336, 151338]
    # 创建采样参数。temperature 控制生成文本的多样性，top_p 控制核心采样的概率
    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens,
                                     stop_token_ids=stop_token_ids)
    # 初始化 vLLM 推理引擎
    llm = LLM(model=model, tokenizer=tokenizer, max_model_len=max_model_len, trust_remote_code=True)
    outputs = llm.generate(prompts, sampling_params)
    return outputs


if __name__ == "__main__":
    # 初始化 vLLM 推理引擎
    model = '/root/autodl-tmp/Qwen/Qwen2-VL-2B-Instruct'  # 指定模型路径
    tokenizer = None
    # 加载分词器后传入vLLM 模型，但不是必要的。
    processor = AutoProcessor.from_pretrained(model)

    messages = [
        {"role": "system", "content": "你是一个有用的助手。"},
        {"role": "user", "content": [
            {"type": "image_url",
             "image_url": {
                 "url": "https://modelscope.oss-cn-beijing.aliyuncs.com/resource/qwen.png"}
             },
            {"type": "text", "text": "插图中的文本是什么？"}
        ]
         }
    ]

    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    outputs = get_completion(prompt, model, tokenizer=tokenizer, max_tokens=512, temperature=1, top_p=1,
                             max_model_len=2048)

    # 输出是一个包含 prompt、生成文本和其他信息的 RequestOutput 对象列表。
    # 打印输出。
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
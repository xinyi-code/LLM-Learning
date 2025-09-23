#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/9/23 17:21
# @Author  : xinyi
# @File    : config.py
# @Description    :

import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field



class AppConfig(BaseModel):
    # Generic LLM (e.g., SiliconFlow OpenAI-compatible)
    llm_api_key: str = Field(default_factory=lambda: os.getenv("LLM_API_KEY", ""))
    llm_api_base: str = Field(default_factory=lambda: os.getenv("LLM_API_BASE", ""))
    llm_model: str = Field(default_factory=lambda: os.getenv("LLM_MODEL", ""))

    emb_api_base: str = Field(default_factory=lambda: os.getenv("EMB_API_BASE", ""))
    emb_model: str = Field(default_factory=lambda: os.getenv("EMB_MODEL", ""))


# 加载环境变量env
load_dotenv('../.env')
config = AppConfig()
# LLM Learning
LLM Learning手册，其中包含LLM API使用与RAG实现等实例

## 项目结构
root  
├─data   
├─rag_learning    RAG实现  
├─tutorials       Langchain接口Demo
│  

## 项目说明
### 1.tutorials (tutorials)
1. ChatAPI相关组件的demo示例
2. Chain的示例
3. RAG示例


### 2.RAG Learning (rag_learning)
![rag流程](./imgs/rag_process.jpg)
项目中包含两个实现方式
1. 基于langchain框架实现的rag系统
>  python langchain_rag.py
2. 不采用框架0-1实现的rag系统
> test_rag.ipynb

### 3.LLM服务部署 (vllm_server)
包含vllm部署、fastAPI部署、性能比较报告等

### 4.embedding模型 (embedding_modelzhen)
主要为bge-m3模型的混合检索以及fine-tune例子

### To Update
1. embedding模型finetune
2. agent实现




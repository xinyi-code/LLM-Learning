# vLLM 部署
基于vLLM 部署实例

## 项目结构


## 项目说明
### 1. vllm框架
#### 1.1 本地调用
> python  vllm_local.py

#### 1.2 部署示例
step1. 启动模型服务 
> vllm serve Qwen/Qwen2.5-7B-Instruct  

step2. 调用模型服务 
> python  vllm_server.py


### 2. FastAPI框架部署示例
采用
> python  fastapi_server.py



## 性能对比
- 测试一下vLLm生成速度
```shell script
python benchmark_throughput.py \
	--model /root/autodl-tmp/qwen/Qwen2.5-7B-Instruct \
	--backend vllm \
	--input-len 64 \
	--output-len 128 \
	--num-prompts 25 \
	--seed 2024 \
    --dtype float16 \
    --max-model-len 512
```
生成速率：
> Throughput: 9.14 requests/s, 1754.43 tokens/s

- huggingface的Transformers 库调用方式
```shell script
python benchmark_throughput.py \
	--model /root/autodl-tmp/qwen/Qwen2.5-7B-Instruct \
	--backend hf \
	--input-len 64 \
	--output-len 128 \
	--num-prompts 25 \
	--seed 2024 \
	--dtype float16 \
    --hf-max-batch-size 25
```

生成速率：
> Throughput: 6.99 requests/s, 1342.53 tokens/s

总体上， 由于vLLM框架有K-V Cache、Page Attention优化， vLLM快不少

## To Update
1.FastAPI 多实例


## refer
1. https://github.com/vllm-project/vllm
2. https://github.com/datawhalechina/self-llm



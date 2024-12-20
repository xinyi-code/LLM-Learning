# Finetune

In this example, we show how to finetune the embedder with your data.

## 1. Installation

```shell
pip install -U FlagEmbedding[finetune]
```

## 2. Data format
模型微调的数据集格式是json line格式文件，json格式如下：

```jsonl
{"query": str, "pos": List[str], "neg":List[str],  "prompt": "Retrieve semantically similar text.", "type": "symmetric_sts"}
```
query是查询，pos是正文本列表，neg是负文本列表, prompt主要作用在query上标明模型任务, type 是设置模型类型（主要是bge-en-icl，包含 normal, symmetric_class, symmetric_clustering三种模式）


### hard负例数据集构造
添加了L2、余弦相似度计算方式， 挖掘与query相似，但不是正样本的数据，构建hard_negative数据集

```shell
cd scripts
sh run_neg_mine.sh
```


### 模型蒸馏
调整模型在自己的数据集上自适应， 通过混合检索模式得到分数，去蒸馏原模型，提高单检索模式性能

```shell
cd scripts
sh run_distillation.sh
```


### bge-m3 fine-tune

```shell
torchrun --nproc_per_node 2 \
	-m FlagEmbedding.finetune.embedder.encoder_only.m3 \
	--model_name_or_path BAAI/bge-m3 \
    --cache_dir ./cache/model \
    --train_data ./example_data/retrieval \
    			 ./example_data/sts/sts.jsonl \
    			 ./example_data/classification-no_in_batch_neg \
    			 ./example_data/clustering-no_in_batch_neg \
    --cache_path ./cache/data \
    --train_group_size 8 \
    --query_max_len 512 \
    --passage_max_len 512 \
    --pad_to_multiple_of 8 \
    --knowledge_distillation True \
    --same_dataset_within_batch True \
    --small_threshold 0 \
    --drop_threshold 0 \
    --output_dir ./test_encoder_only_m3_bge-m3_sd \
    --overwrite_output_dir \
    --learning_rate 1e-5 \
    --fp16 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --dataloader_drop_last True \
    --warmup_ratio 0.1 \
    --gradient_checkpointing \
    --deepspeed ../ds_stage0.json \
    --logging_steps 1 \
    --save_steps 1000 \
    --negatives_cross_device \
    --temperature 0.02 \
    --sentence_pooling_method cls \
    --normalize_embeddings True \
    --kd_loss_type m3_kd_loss \
    --unified_finetuning True \
    --use_self_distill True \
    --fix_encoder False \
    --self_distill_start_step 0
```

Here are some new arguments:

- **`colbert_dim`**: Dim of colbert linear
- **`unified_finetuning`**: Use unify fine-tuning
- **`use_self_distill`**: Use self-distill when using unify fine-tuning
- **`fix_encoder`**: Freeze the parameters of encoder
- **`self_distill_start_step`**: Num of step when using self-distill


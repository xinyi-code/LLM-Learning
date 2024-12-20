set -x

cd ..
python add_reranker_score.py \
--input_file ./example_data/sts.jsonl \
--output_file ./example_data/out_neg_sts.jsonl \
--reranker_name_or_path BAAI/bge-reranker-v2-m3 \
--devices cuda:0 cuda:1 \
--cache_dir ./cache/model \
--reranker_query_max_length 512 \
--reranker_max_length 1024
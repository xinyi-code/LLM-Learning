set -x

cd ..
python hard_neg_mine.py \
--model_name_or_path BAAI/bge-m3 \
--input_file ./example_data/sts.jsonl \
--output_file ./example_data/out_neg_sts.jsonl \
--range_for_sampling 2-200 \
--negative_number 15 \
--use_gpu_for_searching
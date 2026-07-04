python infer.py \
  --base_path ./models/FLUX.1-Kontext-dev \
  --lora_path ./checkpoints/lora \
  --input_dir ./assets \
  --output_dir ./outputs \
  --json_path ./assets/metadata.json \
  --device 7 \
  --dtype bf16
#!/bin/bash
#SBATCH --job-name=qwen_asr_ft
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --output=logs/finetune_%j.out
#SBATCH --error=logs/finetune_%j.err

echo "============================================"
echo "Qwen3-ASR Finetuning Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "============================================"

# Activate conda
source /home/apps/MLDL/DL-CondaPy3/etc/profile.d/conda.sh
conda activate qwen_env   # <-- your env

# Move to project directory
cd /scratch/supreetm_iitp/devansh/Qwen3-ASR

echo ""
echo "Starting Finetuning..."

python finetuning/qwen3_asr_sft.py \
  --model_path Qwen/Qwen3-ASR-1.7B \
  --train_file ./train.jsonl \
  --eval_file ./eval.jsonl \
  --output_dir ./qwen3-asr-finetuning-out \
  --batch_size 32 \
  --grad_acc 4 \
  --lr 2e-5 \
  --epochs 1 \
  --save_steps 200 \
  --save_total_limit 5

EXIT_CODE=$?

echo ""
echo "============================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Finetuning COMPLETED SUCCESSFULLY!"
else
    echo "Finetuning FAILED with exit code $EXIT_CODE"
fi
echo "Date: $(date)"
echo "============================================"

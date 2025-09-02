#!/bin/bash

#PBS -q normal
#PBS -j oe
#PBS -l select=1:ncpus=16:mem=64g
#PBS -l walltime=05:00:00
#PBS -N correct-l8r1
#PBS -P personal-btey003

# MODEL='Llama70B'
# TYPE='rag3'
# FILENAME="Llama-70B-Instruct_sample=199_dp=5_${TYPE}"

# MODEL='Llama8B'
# TYPE='rag1'
# FILENAME="Llama-8B-Instruct_sample=199_dp=5_${TYPE}"

# MODEL='Mistral7B'
# TYPE='rag2'
# FILENAME="Mistral-7B-Instruct_sample=199_dp=5_${TYPE}"

# MODEL='Qwen'
# TYPE='rag3'
# FILENAME="Qwen_sample=199_dp=5_${TYPE}"

MODEL='Llamacode'
TYPE='tts3'
FILENAME="Llamacode_sample=199_dp=5_${TYPE}"

# MODEL='Llama1B'
# TYPE='rag3'
# FILENAME="Llama-1B-Instruct_sample=199_dp=5_${TYPE}"

module load miniforge3
conda init
conda activate creativity
module load cuda/11.6.2

BASE_PATH="/home/users/ntu/btey003/scratch/coding-creativity"
cd $BASE_PATH
export PYTHONPATH=${BASE_PATH}

export HF_HUB_CACHE=/home/users/ntu/btey003/scratch
export HF_HOME=/home/users/ntu/btey003/scratch

# pip install --upgrade --quiet transformers pandas numpy torch accelerate
huggingface-cli login --token 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
echo "Starting correctness evaluation now."

# correctness_evaluation

python -u steps/creativity_evaluation.py \
    --task correctness \
    --inference-result-path datasets/CodeForce/inference-mohor/GreedyNonGreedy/Neocoder/non-greedy/${MODEL}/${FILENAME}.json \
    --test-case-path datasets/CodeForce/NeoCoder/test_cases_annotated.json \
    --save-folder datasets/CodeForce/evaluation/non-greedy/${MODEL} \
    --type ${TYPE}

# python -u steps/creativity_evaluation.py \
#     --task correctness \
#     --inference-result-path datasets/CodeForce/old_evaluation/${MODEL}/${FILENAME}_creativity.json \
#     --test-case-path datasets/CodeForce/NeoCoder/test_cases_annotated.json \
#     --save-folder datasets/CodeForce/evaluation/${MODEL} \
#     --type ${TYPE}

echo "Correctness evaluation done."
echo "Starting detecting techniques now."

# detect_techniques
# python steps/creativity_evaluation.py \
#     --task detection \
#     --inference-result-path datasets/CodeForce/evaluation/non-greedy/${MODEL}/${FILENAME}_creativity.json \
#     --human-solution-path datasets/CodeForce/NeoCoder/human_solutions.json \

# echo "Detect techniques done."
# echo "Calculating scores now."

# sleep 30s

# # neogauge
# python steps/creativity_evaluation.py \
#     --task creativity \
#     --inference-result-path datasets/CodeForce/evaluation/non-greedy/${MODEL}/${FILENAME}_creativity.json \
#     --human-solution-path datasets/CodeForce/NeoCoder/human_solution_techniques.json \
#     --save-folder datasets/CodeForce/neoresults/non-greedy/${MODEL} \
#     --type ${TYPE}

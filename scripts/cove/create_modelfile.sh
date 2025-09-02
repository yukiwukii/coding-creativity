#!/bin/bash
#PBS -q normal
#PBS -j oe
#PBS -l select=1:ncpus=16:ngpus=2
#PBS -l walltime=16:00:00
#PBS -N create
#PBS -P personal-btey003

module load miniforge3
conda activate cov
cd scratch/coding-creativity/scripts/cove

export OLLAMA_MODELS=/home/users/ntu/btey003/scratch

/home/users/ntu/btey003/scratch/coding-creativity/bin/ollama serve &
sleep 10

while ! /home/users/ntu/btey003/scratch/coding-creativity/bin/ollama list >/dev/null 2>&1; do
    echo "Waiting for Ollama API..."
    sleep 3
done

/home/users/ntu/btey003/scratch/coding-creativity/bin/ollama create llama70-13 -f ./modelfile13
/home/users/ntu/btey003/scratch/coding-creativity/bin/ollama create llama70-2 -f ./modelfile2

/home/users/ntu/btey003/scratch/coding-creativity/bin/ollama list
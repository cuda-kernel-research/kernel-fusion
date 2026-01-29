#!/bin/bash
#SBATCH --job-name=test1 
#SBATCH -N 1 
# old syntax was --gpus=2
#SBATCH --gpus=1


pwd; hostname ; date 

/usr/bin/nvidia-smi

cd ~/cudafusion/kernel-fusion/experiments || exit 1

~/cudafusion/kernel-fusion/venv/bin/python main.py

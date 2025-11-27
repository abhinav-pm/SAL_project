#!/bin/bash
#SBATCH -A param.krishna
#SBATCH --partition=u22
#SBATCH -n 1
#SBATCH -n 14
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2048
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=END
#SBATCH --nodelist=node02

cd
source trojen/bin/activate

python3 -V
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.total,memory.used --format=csv,noheader,nounits -l 300 > gpu_usage.log &


cd /home/abhinav.pm/ABHI/SAL/v2

echo "Job started on $(hostname) at $(date)"

python3 g_train_s2p.py

echo "Job finished at $(date)"
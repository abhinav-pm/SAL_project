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

cd /home/abhinav.pm/ABHI/SAL/v2/p2g_de
python3 -V
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.total,memory.used --format=csv,noheader,nounits -l 300 > gpu_usage.log &



echo "Job started on $(hostname) at $(date)"

python3 c_train_p2g_german.py

echo "Job finished at $(date)"
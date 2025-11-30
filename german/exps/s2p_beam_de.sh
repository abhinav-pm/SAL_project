#!/bin/bash
#SBATCH -A research
#SBATCH -p u22
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=2-00:00:00
#SBATCH --nodelist=gnode012
#SBATCH --job-name=beam

cd
source trojen/bin/activate

cd /home/abhinav.pm/ABHI/SAL/v2/130_hours_exps
python3 -V
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.total,memory.used --format=csv,noheader,nounits -l 300 > gpu_usage.log &



echo "Job started on $(hostname) at $(date)"

python3 generate_all_p2g_data_beam.py \
    --s2p_model_path /home/abhinav.pm/ABHI/SAL/v2/s2p_model_german_phoneme_split \
    --dataset_path /scratch/ABHI/common_voice_de_190h \
    --train_tsv /home/abhinav.pm/ABHI/SAL/v2/130_hours_exps/phonemized/train_130_phoneme.tsv \
    --output_dir ./p2g_training_20_data \
    --generate_danp_beam \
    --beam_size 32

echo "Job finished at $(date)"
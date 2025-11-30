#!/bin/bash
#SBATCH -A research
#SBATCH -p u22
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=2-00:00:00
#SBATCH --nodelist=gnode046
#SBATCH --job-name=beam

cd
cd /home2/dasari.priyanka/ABHI
source trojen/bin/activate

cd /home2/dasari.priyanka/ABHI/SAL/polish/exps
python3 -V
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.total,memory.used --format=csv,noheader,nounits -l 300 > gpu_usage.log &



echo "Job started on $(hostname) at $(date)"

python3 generate_all_p2g_data_polish.py \
    --s2p_model_path /home2/dasari.priyanka/ABHI/SAL/s2p_model_polish_phoneme_v1 \
    --clips_dir /scratch/priyanka/common_voice_pl_23/pl/clips \
    --train_tsv /home2/dasari.priyanka/ABHI/SAL/polish/exps/phonimized/train_130h_phoneme.tsv \
    --output_dir ./p2g_training_polish_130h \
    --generate_danp_beam \
    --beam_size 32

echo "Job finished at $(date)"
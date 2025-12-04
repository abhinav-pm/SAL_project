1. DANP Sampling

python3 generate_all_p2g_polish_resume.py \
    --s2p_model_path ./s2p_model_polish \
    --clips_dir /scratch/priyanka/common_voice_polish23/cv-corpus-23.0-2025-09-05/pl/clips \
    --train_tsv /home/abhinav.pm/ABHI/SAL/v4/phonemized/train_130h_phoneme.tsv \
    --output_dir ./p2g_training_polish_130h \
    --generate_danp_sampling \
    --num_samples 500

2. DANP Beam

python3 generate_all_p2g_polish_resume.py \
    --s2p_model_path ./s2p_model_polish \
    --clips_dir /scratch/priyanka/common_voice_polish23/cv-corpus-23.0-2025-09-05/pl/clips \
    --train_tsv /home/abhinav.pm/ABHI/SAL/v4/phonemized/train_130h_phoneme.tsv \
    --output_dir ./p2g_training_polish_130h \
    --generate_danp_beam \
    --beam_size 32
    
3. TKM

python3 generate_all_p2g_polish_resume.py \
    --s2p_model_path ./s2p_model_polish \
    --clips_dir /scratch/priyanka/common_voice_polish23/cv-corpus-23.0-2025-09-05/pl/clips \
    --train_tsv /home/abhinav.pm/ABHI/SAL/v4/phonemized/train_130h_phoneme.tsv \
    --output_dir ./p2g_training_polish_130h \
    --generate_tkm \
    --tkm_k 32
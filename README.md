## Project Workflow

The project is structured into three main phases for each language (Polish and German):

1.  **S2P Model Training:** A `wav2vec2` model is fine-tuned to convert speech into phoneme sequences.
2.  **Baseline P2G Model Training:** An `mT5` model is fine-tuned to translate perfect phoneme sequences into text.
3.  **Advanced P2G Experiments:** The P2G model is retrained using **DANP** (Data Augmentation with Noisy Phonemes) and **TKM** (Top-K Marginalized) strategies to improve robustness and accuracy.

## Directory Structure

The repository is organized by language, with each containing a similar workflow.

```
SAL_project/
├── german/
│   ├── s2p/                # Phase 1: S2P Model (data prep, train, eval)
│   ├── p2g/                # Phase 2: Baseline P2G Model (data prep, train, eval)
│   └── exps/               # Phase 3: Advanced DANP & TKM Experiments
└── polish/
    └── ...                 # (Same structure as German)
```

## How to Run

Follow these steps sequentially for each language. Scripts are located in the respective directories.

#### Phase 1: Train the S2P Model (`s2p/`)

1.  **Prepare Data:** Run `d_..._split.py` to create data splits from the raw Common Voice dataset.
2.  **Phonemize:** Use the `e_phonemizer.ipynb` notebook to convert text to phonemes.
3.  **Train:** Execute `g_train_s2p.py` to fine-tune the S2P model.
4.  **Evaluate:** Run `j_evaluate_...py` to get the final Phoneme Error Rate (PER).

#### Phase 2: Train the Baseline P2G Model (`p2g/`)

1.  **Prepare Data:** Run `a_prepare_p2g_data...py` and `b_clean_p2g_data...py` on the phonemized files.
2.  **Train:** Execute `c_train_p2g...py` to fine-tune the baseline P2G model.
3.  **Evaluate:** Run `d_evaluate_p2g.py` to get the baseline Word Error Rate (WER).

#### Phase 3: Run Advanced Experiments (`exps/`)

1.  **Generate Noisy Data:** Use the scripts in the root of `exps/` (like `e_generate_all_p2g_data.py`) to create the augmented datasets for DANP and TKM using the trained S2P model.
2.  **Train Advanced Models:** In the `p2g_training_exps/` subdirectory, run the training scripts for the DANP and TKM models.
3.  **Evaluate:** Use the evaluation scripts in the same directory to get the final WER results for all advanced strategies.



All files can be accessed here : https://drive.google.com/drive/folders/1kG_DWS4NfONZcBNWiZxrUJYKZhSXOrnz?usp=drive_link
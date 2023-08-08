# Text-MiningGP12

These are the files and data used for generation.

Do note the order to run the files and some of their dependencies.
Depending on your current environment, there may be more packages required.
Running of notebooks with the relevant folders are to be done on your machine locally.

## Dependencies
- seaborn
- sentencepiece
- textacy
- en_core_web_lg
- en_core_web_md
- openai (and an api key for the code to work)

## Required folders
- data
- models
- results

## Notebook Sequence

### EDA
- T5_00_EDA_LO.ipynb (used for EDA of WebNLG Triplet Format)

### For Text Preprocessing
- covar_00_bbcBatch_LO.ipynb (used to batch preprocess documents, replace text with covariance resolution)
- triplets_01_clean_LO.ipynb (triple generation, greedy)
- triplets_02_tripleFormatting_LO.ipynb (triple generation, strict)

### For LLM Fine Tuning
All these are Kaggle notebooks
- GPT_00_fineTuning.ipynb
- T5_01_train8epoch.ipynb
- T5_01_train16epoch.ipynb

### For Text Generation
All export to a folder called “results”
- GPT_01_textGenDistil_LO.ipynb
- GPT_01_textGenDistilFiltered_LO.ipynb
- GPT_01_textGenGPT2med_LO.ipynb
- ChatGPT_01_promptedTextGen.ipynb
- T5_02_textGen8epoch.ipynb
- T5_02_textGen16Epoch_LO.ipynb

### For Evaluation
- scoring_BatchSimplicityIndex_LO.ipynb
- util_simplicityIndex.py (this is imported into other notebooks)

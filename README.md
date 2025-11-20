# Text Summarization Using LongT5 on PubMed Dataset

This project demonstrates abstractive text summarization on the PubMed dataset using the [LongT5](https://arxiv.org/abs/2112.07916) transformer model with global attention, leveraging the Hugging Face Transformers and Datasets libraries.
This is an experimentation project attempted to explore the text summarization task and reproduce the results obtained in the original work by Guo, M and the team from Google that developed the Long T5 base model with the Tglobal attention mechanism. Considering the limitations of the computation resources and time, the evaluation was constrained to only 1000 test records of the PubMed Dataset, which took about 3 hours to obtain the results using T4 GPU.

## Overview

- **Model:** `google/long-t5-tglobal-base` (pre-trained)
- **Dataset:** [ccdv/pubmed-summarization](https://huggingface.co/datasets/ccdv/pubmed-summarization)
- **Task:** Generate concise summaries (abstracts) from long biomedical articles.
- **Evaluation:** ROUGE metrics on 1000 test records.

## Features

- End-to-end pipeline: data loading, preprocessing, tokenization, inference, and evaluation.
- Handles long input sequences (up to 4096 tokens).
- Efficient inference with GPU support and memory optimization.
- Includes sample code for generating and evaluating summaries.

## Usage

1. **Install dependencies:**
    ```sh
    pip install torch transformers datasets evaluate rouge_score
    ```

2. **Run the notebook:**
    - Open `NLP_TextSummarization_LongT5.ipynb` in Jupyter or VS Code.
    - Execute cells sequentially to:
        - Load the model and dataset
        - Tokenize and preprocess data
        - Evaluate model performance on test data
        - Generate summaries for sample articles

3. **Sample Inference:**
    ```python
    test_article = pubmed_dataset['test'][0]['article']
    summary = generate_summary(test_article)
    print(summary)
    ```

## Results

- **ROUGE-1:** ~29.16
- **ROUGE-2:** ~8.54
- **ROUGE-L:** ~16.74

These scores indicate the model captures key unigrams but has room for improvement in phrase and sequence coherence.

## References
Guo, M., Ainslie, J., Uthus, D. C., Ontañón, S., Ni, J., Sung, Y.-H., & Yang, Y. (2022). LongT5: Efficient Text-To-Text Transformer for Long Sequences. https://doi.org/10.18653/v1/2022.findings-naacl.55 

Summarization. (n.d.). Huggingface.co. https://huggingface.co/docs/transformers/en/tasks/summarization 

Synced. (2021, December 21). Google’s Transformer-Based LongT5 Achieves Performance Gains by Scaling Both Input Length and Model Size. Medium; SyncedReview. https://medium.com/syncedreview/googles-transformer-based-longt5-achieves-performance-gains-by-scaling-both-input-length-and-model-687afb8a3274

## Author

- Kamatchi Gnanavel
- AI/ML Engineer

---

*This project is part of my AI/ML engineering portfolio. For questions or collaboration, feel free to reach out!*




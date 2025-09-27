# Evaluation of Synthetic Data

This code implements the evaluation of synthetic data using multiple metrics, including: 1) MAUVE; 2) feature divergence; 3) domain classification.

## Setup

```
pip install -r requirements.txt
```

## Example Scripts

### MAUVE

Step 1: Obtain sequence embedding for bioRxiv dataset, using the embedding model [Specter2](https://huggingface.co/allenai/specter2). 
The model is chosen as it is trained on scientific papers and supports sequence length up to 512, which is well-aligned with our dataset.

Step 2: Compute MAUVE on the obtained embeddings of real data vs. synthetic data. Embeddings of real data is supplied at [embeddings/biorxiv_valid_test__specter2_len-512_embeddings.npy](embeddings/biorxiv_valid_test__specter2_len-512_embeddings.npy).

```
cd mauve
# Step 1
bash scripts/run_embed_biorxiv.sh [file_path] [gpu_id]
# Step 2
bash scripts/run_compute_mauve_biorxiv.sh [cpu_id_begin] [cpu_id_end] [embedding_path]
```

### Feature distribution divergence

Step 1: Prompt LLM (Gemini) for feature (schemas) extraction. See [annotation](../annotation/).

Step 2: Convert the inferred schema (in the string format of a JSON object) to pandas dataframe. 

Step 3: Compute distribution divergence between the schemas of synthetic data and that of real data.


```
# Steps 2+3
cd feature_divergence
bash scripts/run_biorxiv_schema_convert_compute_divergence.sh [file_path_of_llm_extraction] [version_id]
```

### Domain classification

Step 1: Prompt LLM (Gemini) to annotate the domains for the synthetic data. See [annotation](../annotation).

Step 2: Fine-tune a SciBERT classifier on the annotated synthetic data for the task of 8-class classification. Test the fine-tuned model on the private test data (located under [classification/data](./classification/data)). 

```
# Step 2
cd classification
bash scripts/run_train_classifier.sh [file_path_of_synthetic_data]
```
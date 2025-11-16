# Schema Design & Extraction for bioRxiv Dataset

This folder implements calling Gemini APIs for:

(1) designing a schema;

(2) inferring the features from bioRxiv abstracts, based on a given schema.

The prompt used for (1) is in [schema_design_prompt.txt](prompts/schema_design_prompt.txt).
The prompt used for (2) is in [biorxiv_schema_extraction_prompt.txt](prompts/biorxiv_schema_extraction_prompt.txt), which included the full schema.

## Setup
```
pip install -q -U google-genai
```

Note that a ``GOOGLE_API_KEY`` needs to be obtained from [Google AI Studio](https://ai.google.dev/gemini-api/docs/api-key) and set as an environment variable.

## Example Script

(1) Calls Gemini APIs for designing a schema, based on provided `data_description`, `workload_description`, and `num_features`. We encourage readers to play with these configurations to refine their schema design tailored to their own datasets.

```bash
python design_schema.py
```

(2)
Calls Gemini APIs to infer bioRxiv features, and saves the extracted results to a local file. 

```bash
bash scripts/run_infer_biorxiv_schema.sh [directory] [file_ext] [file_name] [column_name]
```
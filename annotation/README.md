# Schema Extraction for bioRxiv Dataset

This code implements calling Gemini APIs for inferring the features from bioRxiv abstracts, based on a given schema.

The prompt use for prompting is presented at [biorxiv_schema_extraction_prompt.txt](biorxiv_schema_extraction_prompt.txt), which included the full schema.

## Setup
```
pip install -q -U google-genai
```

Note that a ``GOOGLE_API_KEY`` needs to be obtained from [Google AI Studio](https://ai.google.dev/gemini-api/docs/api-key) and set as an environment variable.

## Example Script

Calls Gemini APIs to infer bioRxiv features, and saves the extracted results to a local file. 

```
bash scripts/run_infer_biorxiv_schema.sh [directory] [file_ext] [file_name] [column_name]
```
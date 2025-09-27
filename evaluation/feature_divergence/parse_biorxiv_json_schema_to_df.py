"""Parses a JSON-like string from a specific column in a CSV file,

cleans the resulting data, and saves it to a new CSV file.
"""

# --- Standard Library Imports ---
import argparse
import json
import logging
import os
import sys

# --- Third-Party Library Imports ---
import pandas as pd

# --- Setup Standard Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def parse_and_clean_schema(schema_string: str) -> dict:
  """Attempts to parse a schema string with multiple fallback strategies.

  This function exactly replicates the original script's parsing logic:
  1. Try parsing the string with the first 7 and last 3 characters stripped.
  2. If that fails, try parsing the entire string.
  3. If both fail, return a dictionary with all values as None.
  """
  parsed_json = None

  # Strategy 1: Try parsing a slice of the string
  try:
    # Handles the case "```json{}```"
    parsed_json = json.loads(schema_string[7:-3])
  except (json.JSONDecodeError, TypeError):
    # Strategy 2: If slicing fails, try parsing the whole string
    try:
      parsed_json = json.loads(schema_string)
    except (json.JSONDecodeError, TypeError):
      # Strategy 3: If all parsing fails, return None
      return None, 0

  # --- Clean up the successfully parsed data ---
  if parsed_json:
    # Original script asserted a length of 8. We can log if it's different.
    if len(parsed_json) != 8:
      logging.warning(
          f"Parsed JSON has an unexpected length of {len(parsed_json)}."
      )

  return parsed_json, 1


def main():
  """Main function to run the data parsing and cleaning pipeline."""
  parser = argparse.ArgumentParser(
      description="Parse and clean schema data from a CSV file."
  )
  parser.add_argument(
      "--input_path",
      "-i",
      type=str,
      required=True,
      help="Path to the input CSV file.",
  )
  parser.add_argument(
      "--schema_column",
      type=str,
      default="inferred_schema",
      help="Name of the column containing the schema strings.",
  )
  parser.add_argument(
      "--output_path",
      "-o",
      type=str,
      default="biorxiv_parsed_schema.csv",
      help="Path to save the cleaned output CSV file.",
  )
  args = parser.parse_args()

  # --- Load Data ---
  try:
    logging.info(f"Reading data from {args.input_path}...")
    df = pd.read_csv(args.input_path)
  except FileNotFoundError:
    logging.error(f"Input file not found: {args.input_path}")
    sys.exit(1)
  except Exception as e:
    logging.error(f"Error reading CSV file: {e}")
    sys.exit(1)

  # --- Parse and Clean Data ---
  logging.info(
      f"Parsing and cleaning data from the '{args.schema_column}' column..."
  )

  # --- Create the Default/Fallback Dictionary ---
  default_schema = {
      "primary_research_area": None,
      "model_organism": None,
      "experimental_approach": None,
      "dominant_data_type": None,
      "research_focus_scale": None,
      "disease_mention": None,
      "sample_size": None,
      "research_goal": None,
  }

  parsed_data = []
  cnt_success = 0
  for text in df[args.schema_column]:
    parsed_json, success = parse_and_clean_schema(text)
    if success:
      cnt_success += 1
    else:
      # If parsing failed, append the default schema
      logging.warning(f"Failed to parse schema for text: {text}")
      # Append a dictionary with all None values
      parsed_json = {key: None for key in default_schema.keys()}
    parsed_data.append(parsed_json)

  df_parsed = pd.DataFrame(parsed_data)

  logging.info(
      f"Successfully parsed {cnt_success} out of {len(df)} total entries."
  )

  # --- Save Results ---
  output_dir = os.path.dirname(args.output_path)
  os.makedirs(output_dir, exist_ok=True)
  logging.info(f"Saving cleaned data to {args.output_path}...")
  df_parsed.to_csv(args.output_path, index=False, encoding="utf-8")
  logging.info("Processing complete.")


if __name__ == "__main__":
  main()

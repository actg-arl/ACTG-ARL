"""Main script for differentially private synthesis of bioRxiv schema data via AIM."""

import argparse
import logging
import os
import domain
import generate
import pandas as pd

# --- Setup Standard Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def main():
  """Main function to load Biorxiv schema data, define the data domain,

  and generate synthetic data based on it.
  """

  parser = argparse.ArgumentParser(
      description="Generate synthetic Biorxiv schema data."
  )
  parser.add_argument("--eps", type=float, default=4.0)
  parser.add_argument(
      "--rho",
      type=float,
      default=0,
      help="Rho value for the synthetic data generation.",
  )
  parser.add_argument(
      "--pgm_iters",
      type=int,
      default=2000,
      help=(
          "Number of optimization iterations for the synthetic data generation."
      ),
  )
  parser.add_argument(
      "--num_gen",
      type=int,
      default=5000,
      help="Number of synthetic samples to generate.",
  )
  parser.add_argument(
      "--data_path",
      type=str,
      default="biorxiv/biorxiv_schema_v2_train_gemini-2.5-flash_parsed.csv",
  )
  args = parser.parse_args()

  # --- 1. Load the real dataset ---
  data_path = os.path.join("../data", args.data_path)
  logging.info(f"Loading data from {data_path}...")
  try:
    df = pd.read_csv(data_path)
  except FileNotFoundError:
    raise FileNotFoundError(f"Error: Data file not found at {data_path}")
  logging.info(f"Successfully loaded {len(df)} records.")

  # --- 2. Define the data schema (attribute_domains) ---
  logging.info("Defining data schema...")

  attribute_domains = {
      "map_primary_research_area": domain.CategoricalAttribute([
          "Biochemistry",
          "Bioinformatics",
          "Biophysics",
          "Cancer Biology",
          "Cell Biology",
          "Clinical Trials",
          "Developmental Biology",
          "Ecology",
          "Epidemiology",
          "Evolutionary Biology",
          "Genetics",
          "Genomics",
          "Immunology",
          "Microbiology",
          "Molecular Biology",
          "Neuroscience",
          "Paleontology",
          "Pathology",
          "Pharmacology and Toxicology",
          "Physiology",
          "Plant Biology",
          "Public Health",
          "Scientific Communication and Education",
          "Structural Biology",
          "Synthetic Biology",
          "Systems Biology",
          "Zoology",
          "Other",
      ]),
      "map_model_organism": domain.CategoricalAttribute([
          "Human",
          "Mouse/Rat",
          "Zebrafish",
          "Drosophila melanogaster",
          "Caenorhabditis elegans",
          "Saccharomyces cerevisiae",
          "Escherichia coli",
          "Arabidopsis thaliana",
          "Plant",
          "Cell Culture",
          "In Silico / Computational",
          "Other Mammal",
          "Other Vertebrate",
          "Other Invertebrate",
          "Other Microbe",
          "Not Applicable / Review",
          "Other",
      ]),
      "map_experimental_approach": domain.CategoricalAttribute([
          "Wet Lab Experimentation",
          "Computational / In Silico Analysis",
          "Clinical Study",
          "Field Study / Observation",
          "Case Study / Case Review",
          "Review / Meta-analysis",
          "New Method Development",
          "Theoretical Modeling",
          "Other",
      ]),
      "map_dominant_data_type": domain.CategoricalAttribute([
          "Genomic",
          "Transcriptomic",
          "Proteomic",
          "Metabolomic",
          "Imaging",
          "Structural",
          "Phenotypic / Behavioral",
          "Ecological / Environmental",
          "Clinical / Patient Data",
          "Simulation / Model Output",
          "Multi-omics",
          "Other",
      ]),
      "map_research_focus_scale": domain.CategoricalAttribute([
          "Molecular",
          "Cellular",
          "Circuit / Network",
          "Tissue / Organ",
          "Organismal",
          "Population",
          "Ecosystem",
          "Multi-scale",
          "Other",
      ]),
      "map_disease_mention": domain.CategoricalAttribute([
          "Cancer",
          "Neurodegenerative Disease",
          "Infectious Disease",
          "Metabolic Disease",
          "Cardiovascular Disease",
          "Autoimmune / Inflammatory Disease",
          "Psychiatric / Neurological Disorder",
          "Genetic Disorder",
          "No Specific Disease Mentioned",
          "Other",
      ]),
      "map_sample_size": domain.CategoricalAttribute([
          "Single Subject / Case Study",
          "Small Cohort (<50 subjects)",
          "Medium Cohort (50-1000 subjects)",
          "Large Cohort / Population-scale (>1000 subjects)",
          "Relies on Cell/Animal Replicates",
          "Not Specified / Not Applicable",
      ]),
      "map_research_goal": domain.CategoricalAttribute([
          "Investigating a mechanism",
          "Characterizing a system/molecule",
          "Developing a method/tool",
          "Identifying novel elements",
          "Testing a hypothesis",
          "Quantifying a parameter",
          "Evaluating/Comparing approaches",
          "Other",
      ]),
  }

  # --- 3. Run the synthetic data generation ---
  logging.info("Starting synthetic data generation...")
  t_start = pd.Timestamp.now()

  synth = generate.run(
      attribute_domains,
      df,
      epsilon=args.eps,
      delta=3.38e-6,
      rho=args.rho,
      num_records=args.num_gen,
      pgm_iters=args.pgm_iters,
  )
  logging.info(
      "Synthetic data generation completed in: %s", pd.Timestamp.now() - t_start
  )

  # --- 4. Save and inspect the output ---
  if args.rho:
    output_path = f"synthetic_biorxiv_et_{args.num_gen//1000}k_rho-{args.rho}_iter-{args.pgm_iters}.csv"
  else:
    output_path = f"synthetic_biorxiv_et_{args.num_gen//1000}k_eps-{args.eps}_iter-{args.pgm_iters}.csv"
  logging.info(f"Saving {len(synth)} generated records to {output_path}...")
  synth.to_csv(output_path, index=False)

  logging.info(
      "Generation complete. Here are the first 5 rows of the synthetic data:"
  )
  logging.info("\n%s", synth.head())


if __name__ == "__main__":
  main()

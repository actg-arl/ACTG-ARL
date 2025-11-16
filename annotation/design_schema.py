import os

from google import genai

if __name__ == "__main__":
  GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

  if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set.")

  client = genai.Client(api_key=GOOGLE_API_KEY)

  data_description = (
      "This is a dataset of biological research paper abstracts hosted on the"
      " bioRxiv preprint server."
  )
  workload_description = (
      "Help feature the category and main idea of the paper, for the purpose of"
      " e.g. assigning a reviewer."
  )
  num_features = 8

  prompt_template = open("prompts/schema_design_prompt.txt").read()
  input_prompt = prompt_template.format(
      data_description=data_description,
      workload_description=workload_description,
      num_features=num_features,
  )

  MODEL_NAME = "gemini-2.5-flash-lite-preview-06-17"

  response = client.models.generate_content(
      model=MODEL_NAME,
      contents=input_prompt,
  )

  print(response.text.strip())

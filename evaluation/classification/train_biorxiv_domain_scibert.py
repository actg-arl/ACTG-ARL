"""Fine-tunes a SciBERT model for Biorxiv domain classification."""

import argparse
import logging
import sys

import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('training_run.log'),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


class BiorxivDataset(torch.utils.data.Dataset):
  """A PyTorch Dataset for handling tokenized text and labels."""

  def __init__(self, encodings, labels):
    self.encodings = encodings
    self.labels = labels

  def __getitem__(self, idx):
    item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    item['labels'] = torch.tensor(self.labels[idx])
    return item

  def __len__(self):
    return len(self.labels)


def compute_metrics(pred):
  """Metric calculation function."""
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  precision, recall, f1_macro, _ = precision_recall_fscore_support(
      labels, preds, average='macro'
  )
  acc = accuracy_score(labels, preds)
  return {
      'accuracy': acc,
      'macro_f1': f1_macro,
      'precision': precision,
      'recall': recall,
  }


def main():
  # --- 3. Argument Parsing and Data Loading (Modified) ---
  logger.info('--- Loading and Preparing Data ---')
  parser = argparse.ArgumentParser(
      description='Fine-tune SciBERT for Biorxiv Domain Classification'
  )
  parser.add_argument('--train_df_path', type=str, default='biorxiv_train.csv')
  parser.add_argument(
      '--test_df_path',
      type=str,
      default='biorxiv_domain_valid_test_gemini-2.5-flash.csv',
  )
  parser.add_argument('--field_train', type=str, default='generated_text')
  parser.add_argument('--field_test', type=str, default='abstract')
  parser.add_argument('--n_epochs', type=float, default=2)
  args = parser.parse_args()

  # Load dataframes from CSVs
  df_train_full = pd.read_csv(
      f'data/{args.train_df_path}'
  )
  df_test = pd.read_csv(
      f'data/{args.test_df_path}'
  )

  df_train_full = df_train_full.dropna(
      subset=[args.field_train, 'inferred_domain']
  )
  logger.info(f'Training data shape after dropping NA: {df_train_full.shape}')

  # For robust training, we'll split the training data into a train and validation set.
  df_train, df_val = train_test_split(
      df_train_full,
      test_size=0.1,
      random_state=42,
      stratify=df_train_full['inferred_domain'],
  )

  # Extract text and labels
  X_train = df_train[args.field_train].tolist()
  X_val = df_val[args.field_train].tolist()
  X_test = df_test[args.field_test].tolist()
  y_train_labels = df_train['inferred_domain'].tolist()
  y_val_labels = df_val['inferred_domain'].tolist()
  y_test_labels = df_test['inferred_domain'].tolist()

  # --- 4. Label Encoding ---
  label_encoder = LabelEncoder()
  y_train = label_encoder.fit_transform(y_train_labels)
  y_val = label_encoder.transform(y_val_labels)
  y_test = label_encoder.transform(y_test_labels)

  num_classes = len(label_encoder.classes_)
  logger.info(f'Number of classes: {num_classes}')

  # --- 5. Tokenization ---
  logger.info('\n--- Tokenizing Text ---')
  model_name = 'allenai/scibert_scivocab_uncased'
  # model_name = "bert-base-uncased"
  tokenizer = AutoTokenizer.from_pretrained(model_name)

  # Tokenize all datasets. We truncate to the max length the model can handle.
  train_encodings = tokenizer(
      X_train, truncation=True, padding=True, max_length=512
  )
  val_encodings = tokenizer(
      X_val, truncation=True, padding=True, max_length=512
  )
  test_encodings = tokenizer(
      X_test, truncation=True, padding=True, max_length=512
  )

  # Create PyTorch datasets
  train_dataset = BiorxivDataset(train_encodings, y_train)
  val_dataset = BiorxivDataset(val_encodings, y_val)
  test_dataset = BiorxivDataset(test_encodings, y_test)

  # --- 6. Model Fine-Tuning ---
  logger.info('\n--- Setting up Fine-Tuning ---')

  model = AutoModelForSequenceClassification.from_pretrained(
      model_name, num_labels=num_classes
  )

  training_args = TrainingArguments(
      output_dir='./results',
      num_train_epochs=args.n_epochs,
      per_device_train_batch_size=16,
      per_device_eval_batch_size=64,
      warmup_steps=500,
      weight_decay=0.01,
      logging_dir='./logs',
      logging_steps=100,
      eval_strategy='epoch',
  )

  # Initialize the Trainer
  trainer = Trainer(
      model=model,
      args=training_args,
      train_dataset=train_dataset,
      eval_dataset=val_dataset,
      compute_metrics=compute_metrics,
  )

  logger.info('\n--- Starting Training ---')
  trainer.train()

  # --- 7. Final Evaluation ---
  logger.info('\n--- Evaluating on Test Set ---')
  test_results = trainer.evaluate(eval_dataset=test_dataset)

  logger.info('\n--- Final Test Results ---')
  for key, value in test_results.items():
    logger.info(f'{key}: {value:.4f}')


if __name__ == '__main__':
  main()

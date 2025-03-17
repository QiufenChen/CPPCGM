#!/usr/bin/env python
# Author  : KerryChen
# File    : predict.py
# Time    : 2025/1/20 22:09

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load sequences from CSV file
file = '/home/qfchen/CPPCGM/CPPGenerator/results/generated_peptides.csv'

df = pd.read_csv(file)
true_sequences = df['Ture Peptides'].values.tolist()
sequences = df['Generated Peptides'].values.tolist()
sequences = [' '.join(seq) for seq in sequences]
# Define the models and their corresponding tokenizers
models = [
    {
        "name": "ProtBert",
        "model_path": "../CPPCGM/model/ProtBert_CPPSet1",
        "tokenizer_path": '../Rostlab/prot_bert'
    },
    {
        "name": "ProtBert_BFD",
        "model_path": "../CPPCGM/model/ProtBert_BFD_CPPSet1/",
        "tokenizer_path": '../Rostlab/prot_bert_bfd'
    },
    {
        "name": "ProtElectra",
        "model_path": "../model/ProtElectra_CPPSet1/",
        "tokenizer_path": '../Rostlab/prot_electra_discriminator_bfd'
    }
]

# Function to get predictions from a model
def get_predictions(model_path, tokenizer_path, sequences):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, do_lower_case=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    predictions = []
    with torch.no_grad():
        for seq in sequences:
            inputs = tokenizer(seq, return_tensors="pt", padding=True, truncation=True, max_length=70)
            outputs = model(**inputs)
            logits = outputs.logits
            predictions.append(torch.argmax(logits, dim=1).item())
    return predictions

# Collect predictions from all models
results = {"Sequence": sequences}

for model_info in models:
    model_name = model_info["name"]
    predictions = get_predictions(model_info["model_path"], model_info["tokenizer_path"], sequences)
    results[model_name] = predictions

# Convert results to DataFrame
result_df = pd.DataFrame(results)

# Remove spaces from the sequences in the 'Sequence' column
result_df['Sequence'] = result_df['Sequence'].str.replace(" ", "")
result_df['True Sequence'] = true_sequences

# Save the merged results to a CSV file
output_file = './results/evaluate_peptides.csv'
result_df.to_csv(output_file, index=False)

print(f"Results saved to {output_file}")

count = result_df[(result_df["ProtBert"] + result_df["ProtBert_BFD"] + result_df["ProtElectra"] >= 2)].shape[0]

print(f"Number of sequences with at least 2 '1's in the last three columns: {count}")



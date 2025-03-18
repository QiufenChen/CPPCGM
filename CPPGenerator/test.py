#!/usr/bin/env python
# Author  : KerryChen
# File    : test.py
# Time    : 2025/1/16 23:56

import re
import os
import torch
import torch.nn as nn
import random
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertLMHeadModel, BertForSequenceClassification, AdamW
from sklearn.metrics import accuracy_score, roc_auc_score
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# Initialize model and tokenizer
model_name = "../Rostlab/prot_bert"
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained(model_name)

class Generator(nn.Module):
    """ Define Generator """

    def __init__(self):
        super().__init__()
        self.model = BertLMHeadModel.from_pretrained(model_name, return_dict=True).to(device)

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return torch.argmax(output[0], -1), output.logits


def perturb_sequence(sequence, perturb_rate=0.5):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    perturbed = []
    for aa in sequence:
        if random.random() < perturb_rate:
            perturbed.append(random.choice(amino_acids))
        else:
            perturbed.append(aa)
    return ' '.join(perturbed)

class SequenceDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        real_seq = self.sequences[index]
        fake_seq = perturb_sequence(real_seq)

        real_seq = re.sub(r"[UZOB]", "X", " ".join(real_seq))
        fake_seq = re.sub(r"[UZOB]", "X", fake_seq)

        real_enc = tokenizer(real_seq, truncation=True, padding='max_length', max_length=70, return_tensors="pt")
        fake_enc = tokenizer(fake_seq, truncation=True, padding='max_length', max_length=70, return_tensors="pt")

        return (real_enc['input_ids'].squeeze(0),
                fake_enc['input_ids'].squeeze(0),
                fake_enc['attention_mask'].squeeze(0),
                real_seq, fake_seq)


# Load dataset and create Dataloader
file = '../dataset/independent_set.csv'
seqs = [item for item in pd.read_csv(file)['Sequence'].values]
batch_size = 1
dataset = SequenceDataset(seqs)
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

model = Generator()
model.load_state_dict(torch.load('/home/qfchen/CPPCGM/CPPGenerator/model/generator_5.pt'))

model.eval()
generated_sequences = []
with torch.no_grad():
    for i, data in enumerate(dataloader):
        gene_data, gene_logits = model(data[1].to(device), data[2].to(device))
        decoded_seqs = tokenizer.batch_decode(gene_data, skip_special_tokens=True)
        generated_sequences.extend(decoded_seqs)

for i, seq in enumerate(generated_sequences):
    print(f"Generated peptide {i+1}: {seq.replace(' ', '')}")


df = pd.DataFrame(generated_sequences, columns=["Generated Peptides"])
df['Ture Peptides'] = seqs
df['Generated Peptides'] = df['Generated Peptides'].str.replace(" ", "")
df.to_csv('results/generated_peptides.csv', index=False)

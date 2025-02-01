#!/usr/bin/env python
# Author  : KerryChen
# File    : CPPGenerator.py
# Time    : 2025/1/16 15:19

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
import torch
import copy
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# Fix seed for reproducibility
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(42)

# Initialize model and tokenizer
model_name = "../Rostlab/prot_bert_bfd"
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained(model_name)

# Generator Model
class Generator(nn.Module):
    """ Define Generator """

    def __init__(self):
        super().__init__()
        self.model = BertLMHeadModel.from_pretrained(model_name, return_dict=True).to(device)

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return output.logits  # Only return logits

# Discriminator Model
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=1).to(device)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        return outputs.logits

# Sequence perturbation function for more realistic variations
def perturb_sequence(sequence, perturb_rate=0.5):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    perturbed = []
    for aa in sequence:
        if random.random() < perturb_rate:
            perturbed.append(random.choice(amino_acids))
        else:
            perturbed.append(aa)
    return ' '.join(perturbed)

# Dataset Class
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

# Calculate Overlap Rate
def get_overlap_rate(output, target):
    correct = sum(o == t for o, t in zip(output, target))
    return correct / len(target)

# Calculate BLEU and ROUGE scores
def calculate_text_metrics(real_seqs, fake_seqs):
    bleu_scores = [sentence_bleu([real.split()], fake.split()) for real, fake in zip(real_seqs, fake_seqs)]
    rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = [rouge_scorer_obj.score(real, fake) for real, fake in zip(real_seqs, fake_seqs)]
    return np.mean(bleu_scores), rouge_scores

# Dynamic learning rate scheduler
def adjust_learning_rate(optimizer, epoch, base_lr, decay_rate=0.9):
    new_lr = base_lr * (decay_rate ** epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

# Training Loop
def train(generator, discriminator, dataloader, num_epochs, gen_optimizer, disc_optimizer, save_name):
    gen_criterion = nn.CrossEntropyLoss()
    disc_criterion = nn.BCEWithLogitsLoss()

    results = []

    best_bleu_score = 0.0
    for epoch in range(num_epochs):
        gen_loss_total, disc_loss_total, overlap_total, auc_total, bleu_total = 0, 0, 0, 0, 0
        rouge_total = []

        for real_data, fake_data, fake_attention_mask, real_seq, fake_seq in dataloader:
            real_data = real_data.to(device)
            fake_data = fake_data.to(device)
            fake_attention_mask = fake_attention_mask.to(device)

            # Train Discriminator
            with torch.no_grad():
                fake_generated = generator(fake_data, fake_attention_mask).argmax(dim=-1)

            real_labels = torch.ones(len(real_data), 1).to(device)
            fake_labels = torch.zeros(len(fake_data), 1).to(device)

            real_output = discriminator(real_data, attention_mask=torch.ones_like(real_data).to(device))
            fake_output = discriminator(fake_data, attention_mask=fake_attention_mask)

            real_loss = disc_criterion(real_output, real_labels)
            fake_loss = disc_criterion(fake_output, fake_labels)
            disc_loss = real_loss + fake_loss

            disc_optimizer.zero_grad()
            disc_loss.backward()
            disc_optimizer.step()
            disc_loss_total += disc_loss.item()

            # Train Generator with adversarial loss
            fake_logits = generator(fake_data, fake_attention_mask)
            fake_probs = torch.sigmoid(discriminator(fake_data, attention_mask=fake_attention_mask))
            adv_loss = -torch.log(fake_probs).mean()

            gen_loss = gen_criterion(fake_logits.permute(0, 2, 1), real_data) + adv_loss

            gen_optimizer.zero_grad()
            gen_loss.backward()
            gen_optimizer.step()
            gen_loss_total += gen_loss.item()

            overlap_total += get_overlap_rate(fake_generated.cpu().numpy().flatten(), real_data.cpu().numpy().flatten())

            # Calculate AUC
            labels = torch.cat([real_labels, fake_labels], dim=0).cpu().numpy()
            outputs = torch.cat([real_output, fake_output], dim=0).detach().cpu().numpy()
            auc_total += roc_auc_score(labels, outputs)

            # Calculate BLEU and ROUGE
            bleu, rouge = calculate_text_metrics(real_seq, fake_seq)
            bleu_total += bleu
            rouge_total.extend(rouge)

        # Adjust learning rates dynamically
        adjust_learning_rate(gen_optimizer, epoch, base_lr=1e-5)
        adjust_learning_rate(disc_optimizer, epoch, base_lr=1e-5)

        avg_rouge = {key: np.mean([r[key].fmeasure for r in rouge_total]) for key in rouge_total[0]} if rouge_total else {}
        results.append([epoch+1,
                        f"{gen_loss_total/len(dataloader):.4f}",
                        f"{disc_loss_total/len(dataloader):.4f}",
                        f"{overlap_total/len(dataloader):.4f}",
                        f"{bleu_total/len(dataloader):.4f}",
                        round(avg_rouge['rouge1'], 4), round(avg_rouge['rouge2'], 4), round(avg_rouge['rougeL'], 4)])

        bleu_score = bleu_total/len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Gen Loss: {gen_loss_total/len(dataloader):.4f}, Disc Loss: {disc_loss_total/len(dataloader):.4f}, Overlap: {overlap_total/len(dataloader):.4f}, AUC: {auc_total/len(dataloader):.4f}, BLEU: {bleu_total/len(dataloader):.4f}, ROUGE: {avg_rouge}")

        if bleu_score > best_bleu_score:
            # Save best Models
            # os.makedirs('./model', exist_ok=True)
            best_generator = copy.deepcopy(generator)
            best_discriminator = copy.deepcopy(discriminator)

    os.makedirs('./model', exist_ok=True)
    torch.save(best_generator.state_dict(), './model/generator_' + save_name + '.pt')
    torch.save(best_discriminator.state_dict(), './model/discriminator_' + save_name + '.pt')

    columns = ['Epoch', 'Gen Loss', 'Disc Loss', 'Overlap', 'BLEU', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L']
    df = pd.DataFrame(results, columns=columns)

    os.makedirs('./results', exist_ok=True)
    df.to_csv('./results/training_results_' + save_name + '.csv', index=False)
    return df


# Hyperparameters
batch_size = 16
num_epochs = 10

# Load Data
file = '../dataset/CPPSetAll.csv'
sequences = pd.read_csv(file)['Sequence'].tolist()
dataset = SequenceDataset(sequences)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize Models and Optimizers
generator = Generator()
discriminator = Discriminator()
gen_optimizer = AdamW(generator.parameters(), lr=1e-5)
disc_optimizer = AdamW(discriminator.parameters(), lr=1e-5)

# Train the Models
my_df = train(generator, discriminator, dataloader, num_epochs, gen_optimizer, disc_optimizer, '5')









# import re
# import os
# import torch
# import torch.nn as nn
# import random
# import pandas as pd
# import numpy as np
# from torch.utils.data import DataLoader, Dataset
# from transformers import BertTokenizer, BertLMHeadModel, BertForSequenceClassification, AdamW
# from sklearn.metrics import accuracy_score, roc_auc_score
# import warnings
#
# warnings.filterwarnings("ignore")
#
# # Fix seed for reproducibility
# def setup_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.backends.cudnn.deterministic = True
#
# setup_seed(42)
#
# # Initialize model and tokenizer
# model_name = "../Rostlab/prot_bert_bfd"
# device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
# tokenizer = BertTokenizer.from_pretrained(model_name)
#
# # Generator Model
# class Generator(nn.Module):
#     """ Define Generator """
#
#     def __init__(self):
#         super().__init__()
#         self.model = BertLMHeadModel.from_pretrained(model_name, return_dict=True).to(device)
#
#     def forward(self, input_ids, attention_mask):
#         output = self.model(input_ids=input_ids, attention_mask=attention_mask)
#         return output.logits  # Only return logits
#
# # Discriminator Model
# class Discriminator(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=1).to(device)
#
#     def forward(self, input_ids, attention_mask):
#         outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
#         return outputs.logits
#
# # Sequence perturbation function for more realistic variations
# def perturb_sequence(sequence, perturb_rate=0.1):
#     amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
#     perturbed = []
#     for aa in sequence:
#         if random.random() < perturb_rate:
#             perturbed.append(random.choice(amino_acids))
#         else:
#             perturbed.append(aa)
#     return ' '.join(perturbed)
#
# # Dataset Class
# class SequenceDataset(Dataset):
#     def __init__(self, sequences):
#         self.sequences = sequences
#
#     def __len__(self):
#         return len(self.sequences)
#
#     def __getitem__(self, index):
#         real_seq = self.sequences[index]
#         fake_seq = perturb_sequence(real_seq)
#
#         real_seq = re.sub(r"[UZOB]", "X", " ".join(real_seq))
#         fake_seq = re.sub(r"[UZOB]", "X", fake_seq)
#
#         real_enc = tokenizer(real_seq, truncation=True, padding='max_length', max_length=70, return_tensors="pt")
#         fake_enc = tokenizer(fake_seq, truncation=True, padding='max_length', max_length=70, return_tensors="pt")
#
#         return (real_enc['input_ids'].squeeze(0),
#                 fake_enc['input_ids'].squeeze(0),
#                 fake_enc['attention_mask'].squeeze(0))
#
# # Calculate Overlap Rate
# def get_overlap_rate(output, target):
#     correct = sum(o == t for o, t in zip(output, target))
#     return correct / len(target)
#
# # Dynamic learning rate scheduler
# def adjust_learning_rate(optimizer, epoch, base_lr, decay_rate=0.9):
#     new_lr = base_lr * (decay_rate ** epoch)
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = new_lr
#
# # Training Loop
# def train(generator, discriminator, dataloader, num_epochs, gen_optimizer, disc_optimizer):
#     gen_criterion = nn.CrossEntropyLoss()
#     disc_criterion = nn.BCEWithLogitsLoss()
#
#     for epoch in range(num_epochs):
#         gen_loss_total, disc_loss_total, overlap_total, auc_total = 0, 0, 0, 0
#
#         for real_data, fake_data, fake_attention_mask in dataloader:
#             real_data = real_data.to(device)
#             fake_data = fake_data.to(device)
#             fake_attention_mask = fake_attention_mask.to(device)
#
#             # Train Discriminator
#             with torch.no_grad():
#                 fake_generated = generator(fake_data, fake_attention_mask).argmax(dim=-1)
#
#             real_labels = torch.ones(len(real_data), 1).to(device)
#             fake_labels = torch.zeros(len(fake_data), 1).to(device)
#
#             real_output = discriminator(real_data, attention_mask=torch.ones_like(real_data).to(device))
#             fake_output = discriminator(fake_data, attention_mask=fake_attention_mask)
#
#             real_loss = disc_criterion(real_output, real_labels)
#             fake_loss = disc_criterion(fake_output, fake_labels)
#             disc_loss = real_loss + fake_loss
#
#             disc_optimizer.zero_grad()
#             disc_loss.backward()
#             disc_optimizer.step()
#             disc_loss_total += disc_loss.item()
#
#             # Train Generator with adversarial loss
#             fake_logits = generator(fake_data, fake_attention_mask)
#             fake_probs = torch.sigmoid(discriminator(fake_data, attention_mask=fake_attention_mask))
#             adv_loss = -torch.log(fake_probs).mean()
#
#             gen_loss = gen_criterion(fake_logits.permute(0, 2, 1), real_data) + adv_loss
#
#             gen_optimizer.zero_grad()
#             gen_loss.backward()
#             gen_optimizer.step()
#             gen_loss_total += gen_loss.item()
#
#             overlap_total += get_overlap_rate(fake_generated.cpu().numpy().flatten(), real_data.cpu().numpy().flatten())
#
#             # Calculate AUC
#             labels = torch.cat([real_labels, fake_labels], dim=0).cpu().numpy()
#             outputs = torch.cat([real_output, fake_output], dim=0).detach().cpu().numpy()
#             auc_total += roc_auc_score(labels, outputs)
#
#         # Adjust learning rates dynamically
#         adjust_learning_rate(gen_optimizer, epoch, base_lr=1e-5)
#         adjust_learning_rate(disc_optimizer, epoch, base_lr=1e-5)
#
#         print(f"Epoch {epoch+1}/{num_epochs}, Gen Loss: {gen_loss_total/len(dataloader):.4f}, Disc Loss: {disc_loss_total/len(dataloader):.4f}, Overlap: {overlap_total/len(dataloader):.4f}, AUC: {auc_total/len(dataloader):.4f}")
#
# # Hyperparameters
# batch_size = 16
# num_epochs = 50
#
# # Load Data
# file = '../dataset/CPPSetAll.csv'
# sequences = pd.read_csv(file)['Sequence'].tolist()
# dataset = SequenceDataset(sequences)
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#
# # Initialize Models and Optimizers
# generator = Generator()
# discriminator = Discriminator()
# gen_optimizer = AdamW(generator.parameters(), lr=1e-5)
# disc_optimizer = AdamW(discriminator.parameters(), lr=1e-5)
#
# # Train the Models
# train(generator, discriminator, dataloader, num_epochs, gen_optimizer, disc_optimizer)
#
# # Save Models
# os.makedirs('./model', exist_ok=True)
# torch.save(generator.state_dict(), './model/generator.pt')
# torch.save(discriminator.state_dict(), './model/discriminator.pt')
#

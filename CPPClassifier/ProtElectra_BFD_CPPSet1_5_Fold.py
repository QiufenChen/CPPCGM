import random

import torch
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForSequenceClassification
from torch.utils.data import Dataset
import os
import json
import pandas as pd
import re
import warnings
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, matthews_corrcoef
from imblearn.metrics import specificity_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

os.environ["WANDB_MODE"] = "disabled"
warnings.filterwarnings("ignore")

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

setup_seed(42)


# Load dataset
all_df = pd.read_csv('../dataset/CPPSet1.csv')
model_name = '../Rostlab/prot_electra_discriminator_bfd'

class DeepLocDataset(Dataset):
    def __init__(self, seqs, labels, tokenizer_name=model_name, max_length=70):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, do_lower_case=False)
        self.seqs = seqs
        self.labels = labels
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        seq = " ".join("".join(self.seqs[idx].split()))
        seq = re.sub(r"[UZOB]", "X", seq)
        seq_ids = self.tokenizer(seq, truncation=True, padding='max_length', max_length=self.max_length)
        sample = {key: torch.tensor(val) for key, val in seq_ids.items()}
        sample['labels'] = torch.tensor(self.labels[idx])
        return sample

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    spec = specificity_score(labels, preds)
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'specificity': spec,
        'f1': f1,
        'mcc': matthews_corrcoef(labels, preds)
    }

def model_init():
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return model

# K-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
seqs = all_df['Sequence'].apply(lambda x: ' '.join(x)).tolist()
labels = all_df['Label'].tolist()

fold_results = []

for fold, (train_idx, valid_idx) in enumerate(kf.split(seqs, labels)):
    print(f"Fold {fold + 1}")
    train_seqs = [seqs[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    valid_seqs = [seqs[i] for i in valid_idx]
    valid_labels = [labels[i] for i in valid_idx]

    train_dataset = DeepLocDataset(train_seqs, train_labels, tokenizer_name=model_name, max_length=70)
    valid_dataset = DeepLocDataset(valid_seqs, valid_labels, tokenizer_name=model_name, max_length=70)

    training_args = TrainingArguments(
        learning_rate=1e-4,
        output_dir=f'results/CPPSet1_ProtElecttra_fold_42_{fold + 1}',
        num_train_epochs=20,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=10,
        weight_decay=0.01,
        logging_dir=f'logs/CPPSet1_ProtElecttra_fold_42_{fold + 1}',
        logging_steps=1,
        do_train=True,
        do_eval=True,
        evaluation_strategy="epoch",
        gradient_accumulation_steps=8,
        fp16=True,
        fp16_opt_level="02",
        run_name=f"ProtBert-Fold-42-{fold + 1}",
        seed=42
    )

    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()

    print(f'<<<<<<<<<<<<<<< The result of Fold {fold + 1} >>>>>>>>>>>>>>>>>')
    metrics = trainer.evaluate(valid_dataset)
    fold_results.append(metrics)

    print('<<<<<<<<<<<<<<< Save the model >>>>>>>>>>>>>>>>>')
    file_name = f'model/CPPSet1_ProtElecttra_fold_42_{fold + 1}'
    trainer.save_model(file_name + '.pt')

    pred_scores = trainer.predict(valid_dataset).predictions
    true_labels = trainer.predict(valid_dataset).label_ids
    pred_labels = torch.argmax(torch.tensor(pred_scores), -1)

    torch.save(pred_labels, f'results/CPPSet1_ProtElectra_pred_babels_fold_42_{fold + 1}.pt')
    torch.save(true_labels, f'results/CPPSet1_ProtElectra_true_babels_fold_42_{fold + 1}.pt')

# Aggregate results
avg_results = {key: np.mean([fold[key] for fold in fold_results]) for key in fold_results[0].keys()}

file_path = 'results/CPPSet1_ProtElecttra_42.json'

# 将avg_results保存成json文件
with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(avg_results, f, ensure_ascii=False, indent=4)
print("<<<<<<<<<<<<<<< Cross-Validation Results >>>>>>>>>>>>>>>>>")
print(avg_results)


# import torch
# from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForSequenceClassification
# from torch.utils.data import Dataset
# import os
# import pandas as pd
# import requests
# from tqdm.auto import tqdm
# import numpy as np
# from sklearn.metrics import accuracy_score, precision_recall_fscore_support, matthews_corrcoef
# from imblearn.metrics import specificity_score
# import pandas as pd
# from sklearn.model_selection import train_test_split
# import re
# import warnings
# import os
# os.environ["WANDB_MODE"] = "disabled"
#
# warnings.filterwarnings("ignore")
#
# df = pd.read_csv('../dataset/CPPSet1.csv')
# train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Label'])
# valid_df, test_df = train_test_split(test_df, test_size=0.5, random_state=42, stratify=test_df['Label'])
#
# train_df.to_csv('../dataset/CPPSet1-train.csv', index=False)
# valid_df.to_csv('../dataset/CPPSet1-valid.csv', index=False)
# test_df.to_csv('../dataset/CPPSet1-test.csv', index=False)
#
# print("Done!")
#
# model_name = '../Rostlab/prot_electra_discriminator_bfd'
# # model_name = '../Rostlab/prot_xlnet'
#
#
# class DeepLocDataset(Dataset):
#     def __init__(self, split="train", tokenizer_name=model_name, max_length=80):
#         self.datasetFolderPath = '../dataset/'
#         self.trainFilePath = os.path.join(self.datasetFolderPath, 'CPPSet1-train.csv')
#         self.validFilePath = os.path.join(self.datasetFolderPath, 'CPPSet1-valid.csv')
#         self.testFilePath = os.path.join(self.datasetFolderPath, 'CPPSet1-test.csv')
#         self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, do_lower_case=False)
#         if split=="train":
#             self.seqs, self.labels = self.load_dataset(self.trainFilePath)
#         elif split=="valid":
#             self.seqs, self.labels = self.load_dataset(self.validFilePath)
#         else:
#             self.seqs, self.labels = self.load_dataset(self.testFilePath)
#         self.max_length = max_length
#
#     def load_dataset(self,path):
#         df = pd.read_csv(path,names=['input','labels'],skiprows=1)
#         seq = list(df['input'])
#         seq = [' '.join(i) for i in seq]
#         label = list(df['labels'])
#         assert len(seq) == len(label)
#         return seq, label
#
#     def __len__(self):
#         return len(self.labels)
#
#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
#         seq = " ".join("".join(self.seqs[idx].split()))
#         seq = re.sub(r"[UZOB]", "X", seq)
#         seq_ids = self.tokenizer(seq, truncation=True, padding='max_length', max_length=self.max_length)
#         sample = {key: torch.tensor(val) for key, val in seq_ids.items()}
#         sample['labels'] = torch.tensor(self.labels[idx])
#         return sample
#
# def compute_metrics(pred):
#     labels = pred.label_ids
#     preds = pred.predictions.argmax(-1)
#     precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
#     acc = accuracy_score(labels, preds)
#     spec = specificity_score(labels, preds)
#     return {
#         'accuracy': acc,
#         'precision': precision,
#         'recall': recall,
#         'specificity': spec,
#         'f1': f1,
#         'mcc' : matthews_corrcoef(labels, preds)
#     }
#
# def model_init():
#     model = AutoModelForSequenceClassification.from_pretrained(model_name)#.cuda()
#     #for param in model.parameters(): param.data = param.data.contiguous()
#     return model
#
# train_dataset = DeepLocDataset(split="train", tokenizer_name=model_name, max_length=70)
# valid_dataset = DeepLocDataset(split="valid", tokenizer_name=model_name, max_length=70)
# test_dataset = DeepLocDataset(split="test", tokenizer_name=model_name, max_length=70)
#
#
# training_args = TrainingArguments(
#     learning_rate=1e-4,
#     output_dir='../results',
#     num_train_epochs=20,
#     per_device_train_batch_size=8,
#     per_device_eval_batch_size=8,
#     warmup_steps=10,
#     weight_decay=0.01,
#     logging_dir='../logs',
#     logging_steps=1,
#     do_train=True,
#     do_eval=True,
#     evaluation_strategy="epoch",
#     gradient_accumulation_steps=30,
#     fp16=True,
#     fp16_opt_level="02",
#     run_name="ProBert-BFD-MS",
#     seed=3407)
#
# trainer = Trainer(
#     model_init=model_init,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=valid_dataset,
#     compute_metrics = compute_metrics)
#
# trainer.train()
#
# print('<<<<<<<<<<<<<<< The result of valid dataset >>>>>>>>>>>>>>>>>')
# trainer.evaluate(valid_dataset)
#
# print('<<<<<<<<<<<<<<< The result of test dataset >>>>>>>>>>>>>>>>>')
# trainer.evaluate(test_dataset)
#
# print('<<<<<<<<<<<<<<< Save the model >>>>>>>>>>>>>>>>>')
# trainer.save_model('../model/ProtElectra_CPPSet1.pt')
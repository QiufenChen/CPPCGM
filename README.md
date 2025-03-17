# CPPCGM
A novel framework called CPPCGM (Cell Penetrating Peptide Categorical and Generative Model) is specialized for recognizing and generating CPPs among various peptides just using primary sequence information. 

## Overall framework
The construction of **CPPCGM** involved two components, **CPPClassifier** and **CPPGenerator**. (A) The CPPs classification network incorporates three pre-trained models ([ProtBert](https://huggingface.co/Rostlab/prot_bert), [ProtBert-BFD](https://huggingface.co/Rostlab/prot_bert_bfd), and [ProtElectra-BFD](https://huggingface.co/Rostlab/prot_electra_discriminator_bfd)) for final decision result through voting. (B) The CPP generation network consists of two neural networks: the generator (ProtBert) and the discriminator (ProtBert). These two networks engage in a mutual adversarial process, aiming to train and generate realistic CPPs.

![Figure2_Framework](https://github.com/user-attachments/assets/3f5979e3-beff-41d1-bffd-8f5de3a3590c)

## Quick Start
### Requirements
```
python==3.8.19
numpy==1.26.1
pandas==1.4.2
scikit-learn==1.3.2
scipy==1.10.1
tokenizers==0.19.1
torch==2.4.1+cu121  
torchaudio==2.4.1+cu121  
torchvision==0.19.1+cu121
transformers==4.42.0
```

### Download CPPCGM
```
git clone https://github.com/QiufenChen/CPPCGM.git
```

### Creating the Environment and Installing Libraries 
```
conda create --name <env> --file condalist.txt
```

### Dataset Preparation
We used three benchmark datasets to evaluate the performance of CPPClassifier, namely CPPSet1 ([CPPSet1-train](https://github.com/QiufenChen/CPPCGM/blob/main/dataset/CPPSet1-train.csv) and [CPPSet1-test](https://github.com/QiufenChen/CPPCGM/blob/main/dataset/CPPSet1-test.csv)), CPPSet2 ([CPPSet2-train](https://github.com/QiufenChen/CPPCGM/blob/main/dataset/CPPSet2-train.csv) and [CPPSet2-test](https://github.com/QiufenChen/CPPCGM/blob/main/dataset/CPPSet2-test.csv)), CPPSet3 ([CPPSet3-train](https://github.com/QiufenChen/CPPCGM/blob/main/dataset/CPPSet3-train.csv) and [CPPSet3-test](https://github.com/QiufenChen/CPPCGM/blob/main/dataset/CPPSet3-test.csv)). Their detailed information is shown in the table:
|  Dataset | Positive Samples | Negative Samples | Train Set | Test Set |
| :----: | :----: | :----: | :----: | :----: |
| **CPPSet1** | 1150 | 1150 |  1840    | 460 |
| **CPPSet2** | 454 | 462 |  733 | 183 |
| **CPPSet3** | 730 | 2758 | 1147 | 2341 |
| **CPPSetAll** | 1700 | - | - | - | - |
| **CPPSet4** | 99 | - | - | - | - |

Training CPPGenerator model with [CPPSetAll](https://github.com/QiufenChen/CPPCGM/blob/main/dataset/CPPSetAll.csv), which was positive samples integrated from CPPSet1, CPPSet2, and CPPSet3.

**CPPSet1** is initially sourced CPPs from [CPPsite 2.0](https://webs.iiitd.edu.in/raghava/cppsite/downloads.php), which provides two distinct datasets: \textit{Peptides having natural residues} and \textit{Peptides having non-natural residues}, collectively comprising 1855 entries. For this study, we specifically downloaded the \textit{Peptides having natural residues} dataset, containing 1564 CPPs as positive samples. To ensure data quality, duplicate entries and sequences containing non-standard characters (e.g., X, B, -, Z, O, and U) were systematically removed. After this processing step, 1150 CPPs remained for subsequent analysis. Randomly select a number of negative samples equal to the positive samples from UniProt, with sequence lengths less than 62.

CPPSet2 and CPPSet3 were downloaded from [SiameseCPP](https://github.com/ZXRush001/SiameseCPP/tree/main/dataset).

Testing CPPGenerator model with [CPPSet4](https://github.com/QiufenChen/CPPCGM/blob/main/dataset/CPPSet4.txt). This dataset sourced from the [CellPPD website](https://webs.iiitd.edu.in/raghava/cellppd/dataset.php), contains 99 experimentally validated CPPs sourced from the literature, which were excluded from the training set, as well as 99 randomly generated non-CPPs.
### Run model
- (1) Run CPPClassifier reference [run_CPPClassifier](https://github.com/QiufenChen/CPPCGM/blob/main/CPPClassifier/run_CPPClassifier.md)

- (2) Run CPPGenerator reference [run_CPPGenerator](https://github.com/QiufenChen/CPPCGM/blob/main/CPPGenerator/run_CPPGenerator.md)
### Test model
The models are saved [Here](https://drive.google.com/drive/folders/19NOMd5v2z8atrNn5D0zq29Rw2hSVUe5W?usp=drive_link).
## Contributing to the project
If you encounter problems using CPPCGM, don't hesitate to get in touch with us (chenqf829@foxmail.com)!

## Progress
README for running CPPCGM.

# CPPCGM
A novel framework called CPPCGM (Cell Penetrating Peptide Categorical and Generative Model) is specialized for recognizing and generating CPPs among various peptides just using primary sequence information. 

## Overall framework
The construction of **CPPCGM** involved two components, **CPPClassifier** and **CPPGenerator**. (A) The CPPs classification network incorporates three pre-trained models ([ProtBert](https://huggingface.co/Rostlab/prot_bert), [ProtBert-BFD](https://huggingface.co/Rostlab/prot_bert_bfd), and [ProtElectra-BFD](https://huggingface.co/Rostlab/prot_electra_discriminator_bfd)) for final decision result through voting. (B) The CPP generation network consists of two neural networks: the generator (ProtBert-BFD) and the discriminator (ProtBert-BFD). These two networks engage in a mutual adversarial process, aiming to train and generate realistic CPPs.

![Figure2_Framework](https://github.com/user-attachments/assets/bacfcb04-0d71-44bf-9e17-81f4147b5cbf)

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

### Dataset Preparation
We used three benchmark datasets to evaluate the performance of CPPClassifier, namely CPPSet1 ([CPPSet1-train](https://github.com/QiufenChen/CPPCGM/blob/main/dataset/CPPSet1-train.csv) and [CPPSet1-test](https://github.com/QiufenChen/CPPCGM/blob/main/dataset/CPPSet1-test.csv)), CPPSet2 ([CPPSet2-train](https://github.com/QiufenChen/CPPCGM/blob/main/dataset/CPPSet2-train.csv) and [CPPSet2-test](https://github.com/QiufenChen/CPPCGM/blob/main/dataset/CPPSet2-test.csv)), CPPSet3 ([CPPSet3-train](https://github.com/QiufenChen/CPPCGM/blob/main/dataset/CPPSet3-train.csv) and [CPPSet3-test](https://github.com/QiufenChen/CPPCGM/blob/main/dataset/CPPSet3-test.csv)). Their detailed information is shown in the table:
|  Dataset | Positive Samples | Negative Samples | Train Set | Test Set |
| :----: | :----: | :----: | :----: | :----: |
| **CPPSet1** | 1150 | 1150 |  1840    | 460 |
| **CPPSet2** | 708 | 708 |  1133 | 283 |
| **CPPSet3** | 730 | 2758 | 1147 | 2341 |
| **CPPSetAll** | 1700 | - | - | - | - |

Training CPPGenerator model with [CPPSetAll](https://github.com/QiufenChen/CPPCGM/blob/main/dataset/CPPSetAll.csv)

### Run model
- (1) Run CPPClassifier reference [run_CPPClassifier](https://github.com/QiufenChen/CPPCGM/blob/main/CPPClassifier/run_CPPClassifier.md)

- (2) Run CPPGenerator reference [run_CPPGenerator](https://github.com/QiufenChen/CPPCGM/blob/main/CPPGenerator/run_CPPGenerator.md)

## Contributing to the project
If you encounter problems using CPPCGM, feel free to contact us (chenqf829@foxmail.com)!

## Progress
README for running CPPCGM.

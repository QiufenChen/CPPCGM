# CPPCGM
A novel framework called CPPCGM (Cell Penetrating Peptide Categorical and Generative Model) is specialized for recognizing CPPs among various peptides just using primary sequence information and generating new CPP sequences. 

### Overall framework
The construction of CPPCGM involved two components, CPPClassifier and CPPGenerator. (A) The CPPs classification network incorporates three pre-trained models (ProtBert, ProtBert-BFD, and ProtElectra-Discriminator-BFD) for integrated learning. (B) The CPP generation network consists of two neural networks: the generator (ProtBert-BFD) and the discriminator (ProtBert-BFD). These two networks engage in a mutual adversarial process, aiming to train and generate realistic CPPs.
![Figure2_Framework](https://github.com/QiufenChen/CPPCGM/assets/52032167/b4a2d053-9c2d-44e4-a948-673de8cb43a3)

### Quick Start
#### Requirements
```
python==3.6
numpy==1.26.1
pandas==2.1.2
scikit-learn==1.3.2
scipy==1.11.3
seaborn==0.13.0
tokenizers==0.15.2
torch==1.13.1+cu116
torchaudio==0.13.1+cu116
torchvision==0.14.1+cu116
transformers==4.39.3
```

#### Download CPPCGM
```
git clone https://github.com/QiufenChen/CPPCGM.git
```

#### Dataset Preparation
We used three benchmark datasets to evaluate the performance of CPPClassifier, namely CPPSet1 ([CPPSet1-train](https://github.com/QiufenChen/CPPCGM/blob/main/dataset/CPPSet1-train.csv) and [CPPSet1-test](https://github.com/QiufenChen/CPPCGM/blob/main/dataset/CPPSet1-test.csv)), CPPSet2 ([CPPSet2-train](https://github.com/QiufenChen/CPPCGM/blob/main/dataset/CPPSet2-train.csv) and [CPPSet2-test](https://github.com/QiufenChen/CPPCGM/blob/main/dataset/CPPSet2-test.csv)), CPPSet3 ([MLCPP-train](https://github.com/QiufenChen/CPPCGM/blob/main/dataset/MLCPP-train.csv) and [MLCPP-test](https://github.com/QiufenChen/CPPCGM/blob/main/dataset/MLCPP-test.csv)). Their detailed information is shown in the table:
|        | Positive Samples | Negative Samples |
| :----: | :----: | :----: |
| **CPPSet1** | 1150 | 1150 |
| **CPPSet2** | 708 | 708 |
| **CPPSet3** | 730 | 2758 |

Training CPPGenerator model with [CPPSetAll](https://github.com/QiufenChen/CPPCGM/blob/main/dataset/CPPSetAll.csv)


#### Run model
(1) Run CPPGenerator reference [run_CPPGenerator](https://github.com/QiufenChen/CPPCGM/blob/main/CPPGenerator/run_CPPGenerator.md)
(2) Run CPPClassifier reference [CPPClassifier](https://github.com/QiufenChen/CPPCGM/blob/main/CPPClassifier/run_CPPClassifier.md)

### Contributing to the project
Any pull requests or issues are welcome.

### Progress
README for running CPPCGM.

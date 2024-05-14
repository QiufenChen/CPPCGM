# CPPCGM
A novel framework called CPPCGM (Cell Penetrating Peptide Categorical and Generative Model) is specialized for recognizing CPPs among various peptides just using primary sequence information and generating new CPP sequences. 

### Overall framework
The construction of CPPCGM involved two components, CPPClassifier and CPPGenerator. (A) The CPPs classification network incorporates three pre-trained models (ProtBert, ProtBert-BFD, and ProtElectra-Discriminator-BFD) for integrated learning. (B) The CPP generation network consists of two neural networks: the generator (ProtBert-BFD) and the discriminator (ProtBert-BFD). These two networks engage in a mutual adversarial process, aiming to train and generate realistic CPPs.
![Figure2_Framework](https://github.com/QiufenChen/CPPCGM/assets/52032167/b4a2d053-9c2d-44e4-a948-673de8cb43a3)

### Quick Start
#### Requirements
```
Python 3.6
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

# CPPCGM
A novel framework called CPPCGM (Cell Penetrating Peptide Categorical and Generative Model) is specialized for recognizing CPPs among various peptides just using primary sequence information and generating new CPP sequences. 

### Overall framework
 The construction of CPPCGM involved two components, CPPClassifier and CPPGenerator. (A) The CPPs classification network incorporates three pre-trained models (ProtBert, ProtBert-BFD, and ProtElectra-Discriminator-BFD) for integrated learning. (B) The CPP generation network consists of two neural networks: the generator (ProtBert-BFD) and the discriminator (ProtBert-BFD). These two networks engage in a mutual adversarial process, aiming to train and generate realistic CPPs.
![Figure2_Framework](https://github.com/QiufenChen/CPPCGM/assets/52032167/b4a2d053-9c2d-44e4-a948-673de8cb43a3)


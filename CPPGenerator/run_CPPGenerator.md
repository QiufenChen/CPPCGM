## Run CPPGenerator
Referring to the generative adversarial network (GAN), we designed a generative framework named CPPGenerator to generate de novo CPPs. (**See Fig.(B)**). The CPPGenerator consists of two sub-modules, namely discriminator $D$ and generator $G$, where $G$ and $D$ are defined by the ProtBert-BFD model.

![Figure2_Framework](https://github.com/user-attachments/assets/bc0ddc7b-2a37-48ab-b169-e0547d60d934)

The input of the generator is a peptide sequence to increase random noise, which can be constructed in three strategies:
- (1) mask 50\% tokens
- (2) randomly replace 50\% tokens
- (3) randomly generate a sequence

At this point, you should be in a Linux environment and have the required packages configured.

(1) Train model CPPGenerator with a strategy that masks 50% of the sequence residues.
```
python mask_CPPGenerator.py > log/mask.log
```
(2) Train model CPPGenerator with a strategy that generates sequences randomly.
```
python random_CPPGenerator.py > log/random.log
```
(3) Train model CPPGenerator with a strategy that replaces 50% of the sequence residues.
```
python replace_CPPGenerator.py > log/replace.log
```
**Note：** 
The well-trained models will be saved in the directory `./model/`.

## Generating new peptides
Generate class CPPs using the well-trained model, and refer to the testing process in `mask_predict.ipynb`, `random_predict.ipynb`, and `replace_predict.ipynb`.



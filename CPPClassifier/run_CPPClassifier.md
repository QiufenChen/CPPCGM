# CPPClassifier
In this study, we performed CPPClassifier by fine-tuning the three pretrained models from [Rostlab](https://huggingface.co/Rostlab), namely ProtBert, ProtBert-BFD, and ProtElectra-Discriminator-BFD. The final decision is made based on the vote (**See Fig.(A)**).

![Figure2_Framework](https://github.com/user-attachments/assets/ab8cbdf5-6f4d-42de-bdcd-4913b120448e)

- **ProtBERT** is inspired by the [BERT](https://arxiv.org/pdf/1810.04805) from natural language processing but has been adapted to handle the unique properties of proteins. ProtBERT is trained on 217 million protein sequences sourced from [UniRef100](https://www.uniprot.org/help/downloads) and consists of 30 layers and 16 attention heads, resulting in a total parameter count of 4.2 million. It is efficient and can process long sequences, making it suitable for a wide range of protein-related tasks.
  
- **ProtBert-BFD** is an extended version of ProtBERT, pretrained on the [BFD](https://bfd.mmseqs.com/) dataset, which contains 2.1 billion protein sequences—over an order of magnitude larger than [UniProt](https://www.uniprot.org/). ProtBERT and ProtBERT-BFD both use the same architecture and are pretrained using masked language modeling (MLM) for protein sequences. The primary difference lies in the size of the datasets they are pretrained on, with ProtBERT-BFD offering broader sequence coverage due to the massive scale of BFD compared to UniRef100 for ProtBERT.
  
- **ProtElectra-Discriminator-BFD** is derived from the [ELECTRA](https://arxiv.org/pdf/2003.10555). ELECTRA trains two transformer models: the generator and the discriminator. The generator's function is to replace tokens within a sequence, and it is trained similarly to a masked language model. The discriminator, which is the focus of our interest, is responsible for identifying the tokens that were replaced by the generator in the sequence. Thus, ProtELECTRA-Discriminator-BFD is also pretrained on the BFD dataset using the same discriminator as ELECTRA.

## Download pretrained models
-  **ProtBert** is downloaded from [Here](https://huggingface.co/Rostlab/prot_bert).
-  **ProtBert-BFD** is downloaded from [Here](https://huggingface.co/Rostlab/prot_bert_bfd).
-  **ProtElectra-Discriminator-BFD** is downloaded from [Here](https://huggingface.co/Rostlab/prot_electra_discriminator_bfd).

## Fine-tuning
Herein, we fine-tuned the three aforementioned PLMs to enhance their ability to identify CPPs. Run scripts [ProtBert-CPPSet1.ipynb](https://github.com/QiufenChen/CPPCGM/blob/main/CPPClassifier/ProtBert-CPPSet1.ipynb), [ProtBert_BFD_CPPSet1.ipynb](https://github.com/QiufenChen/CPPCGM/blob/main/CPPClassifier/ProtBert_BFD_CPPSet1.ipynb), and [ProtElectra_BFD_CPPSet1.ipynb](https://github.com/QiufenChen/CPPCGM/blob/main/CPPClassifier/ProtElectra_BFD_CPPSet1.ipynb) in sequence, and then run script [Ensembling-CPPSet1.ipynb](https://github.com/QiufenChen/CPPCGM/blob/main/CPPClassifier/Ensembling-CPPSet1.ipynb) to output the results of the vote among these three models.

## Prediction
The fine-tuned model is used for predicting unknown data. Running script [predict_generate_peptides_mask.ipynb](https://github.com/QiufenChen/CPPCGM/blob/main/CPPClassifier/predict_generate_peptides_mask.ipynb) or [predict_generate_peptides_random.ipynb](https://github.com/QiufenChen/CPPCGM/blob/main/CPPClassifier/predict_generate_peptides_random.ipynb) or [predict_generate_peptides_replace.ipynb](https://github.com/QiufenChen/CPPCGM/blob/main/CPPClassifier/predict_generate_peptides_replace.ipynb) will do the job; users should replace the paths in the script with their own.

**Note that the three scripts differ only in the input data.**

## Contributing to the project
Any pull requests or issues are welcome.






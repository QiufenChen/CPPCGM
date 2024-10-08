We cannot provide the complete pre-trained models ([ProtBert](https://huggingface.co/Rostlab/prot_bert/tree/main), [ProtBert-BFD](https://huggingface.co/Rostlab/prot_bert_bfd/tree/main), [ProtElectra-Discriminator-BFD](https://huggingface.co/Rostlab/prot_electra_discriminator_bfd/tree/main)) used in the manuscript because these models had a very large volume. If you want to successfully run our model, please go to the [Rostlab](https://huggingface.co/Rostlab) to download them.

[ProtBert](https://huggingface.co/Rostlab/prot_bert/tree/main) is based on Bert model which pretrained on a large corpus of protein sequences in a self-supervised fashion. This means it was pretrained on the raw protein sequences only, with no humans labelling them in any way (which is why it can use lots of publicly available data) with an automatic process to generate inputs and labels from those protein sequences.

![image](https://github.com/QiufenChen/CPPCGM/assets/52032167/cdc400c2-a24d-4d7c-9904-af847a4d7267)

[ProtBert-BFD](https://huggingface.co/Rostlab/prot_bert_bfd/tree/main) is based on Bert model which pretrained on a large corpus of protein sequences in a self-supervised fashion. This means it was pretrained on the raw protein sequences only, with no humans labelling them in any way (which is why it can use lots of publicly available data) with an automatic process to generate inputs and labels from those protein sequences.

![image](https://github.com/QiufenChen/CPPCGM/assets/52032167/e2bcae6d-52d4-4ff2-bebf-ddadc90f69a3)

[ProtElectra-Discriminator-BFD](https://huggingface.co/Rostlab/prot_electra_discriminator_bfd/tree/main) employed adversarial training to improve pre-training effectiveness. The ProtElectra-Discriminator-BFD model were pre-trained on [BFD](https://bfd.mmseqs.com/), a dataset consisting of 2.1 billion protein sequences. 

![image](https://github.com/QiufenChen/CPPCGM/assets/52032167/aa54c8dd-5875-4d0b-a931-dbc46e74533b)

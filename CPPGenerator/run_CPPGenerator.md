# Run CPPGenerator
Referring to the generative adversarial network (GAN), we designed a generative framework named CPPGenerator to generate de novo CPPs. (**See Fig.(B)**). The CPPGenerator consists of two sub-modules, namely discriminator $D$ and generator $G$, where $G$ and $D$ are defined by the ProtBert-BFD model.

![Figure2_Framework](https://github.com/user-attachments/assets/4ccd5b7f-c5a8-41b3-8bff-a944fba0cc5f)

The input of the generator is a peptide sequence to increase random noise. Herein, a perturbation function was utilized to generate pseudo-sequences by introducing controlled random mutations into input peptide sequences. Specifically, each amino acid in the sequence was replaced with a randomly selected alternative amino acid based on a predefined probability, mimicking the mutation process of protein sequences. 
```
def perturb_sequence(sequence, perturb_rate=0.5):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    perturbed = []
    for aa in sequence:
        if random.random() < perturb_rate:
            perturbed.append(random.choice(amino_acids))
        else:
            perturbed.append(aa)
    return ' '.join(perturbed)
```

## Training
```
python CPPGenerator.py
```

**Note：** 
The well-trained models will be saved in the directory `./model/`.

## Generating new peptides
Generate class CPPs using the well-trained model. In our study, we chose a perturbation rate of 0.5. This value ensures sufficient diversity in the generated sequences while minimizing the deviation from the structure and function of real sequences.
```
python test.py
```

## Contributing to the project
Any pull requests or issues are welcome.


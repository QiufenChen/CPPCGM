### Run CPPGenerator
Referring to the generative adversarial network (GAN), we designed a generative framework named CPPGenerator for the rational design of CPPs.

The input of the generator is a peptide sequence to increase random noise, which can be constructed in three strategies, (1) mask 50\% tokens, (2) randomly replace 50\% tokens, and (3) randomly generate a sequence.
```
python mask_CPPGenerator.py > log/mask.log
```

```
python random_CPPGenerator.py > log/random.log
```

```
python replace_CPPGenerator.py > log/replace.log
```

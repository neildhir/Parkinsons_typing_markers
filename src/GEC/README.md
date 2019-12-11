# Grammatical error correction

Approach to finding an embedded representation of the correction-functionality, utilised in neural GEC methods.

## Concept

The approach here used neural-machine translation (NMT) methods, but which have been modified to support GEC functionality. Consequently, we are using training corpuses which are of the same language, but where training pairs (x,y) consists of a grammatically incorrect sentence (x) and its target, which is the grammatically correct version of the same sentence (y).

## Classification

The Encoder-Decoder network, constructs a latent representation of each sentence-pair, and this represention is what we use in our classification model, to numerically and conceptually encode the grammatically incorrect sentence, and the corrective action required to transform it into the target sentence.

## Reference papers

- "A Multilayer Convolutional Encoder-Decoder Neural Network for Grammatical Error Correction"
- "Language Model Based Grammatical Error Correction without Annotated Training Data"

## Implemented models

- `model1.py` is based on [this](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html) implementation.

All the above models have been appropriated for GEC purposes.

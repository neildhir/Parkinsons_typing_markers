"""
We use flair as the control classing for all embeddings, as they have already
done all the work of combining typical embeddings in one convenient place.

We consider two types of embeddings:

1) Static
-Classic word embeddings are static and word-level, meaning that each distinct word gets exactly one pre-computed embedding.

2) Dynamic
"""
from flair.embeddings import (ELMoEmbeddings,
                              DocumentPoolEmbeddings,
                              CharacterEmbeddings,
                              DocumentRNNEmbeddings,
                              FlairEmbeddings,
                              StackedEmbeddings,
                              WordEmbeddings,
                              BertEmbeddings)
from flair.data import Sentence
import torch
from pandas import read_csv, DataFrame
from pathlib import Path
import warnings
from typing import List
from numpy import array


def get_static_sentence_embeddings(df: DataFrame,
                                   word_embedding_type: str,
                                   sentence_ID: int = None) -> List[array]:
    """
    Method for getting static (i.e. do not require training)
    sentence embeddings from the MJFF data.

    We only use document pooling here to enforce the static nature of this method,
    if we use the RNN document embedding method, we would have to train it in a downstream
    task (see the dynamic embedding function for this).

    Parameters
    ----------
    word_embedding_type : str
        Which type of embeddings we are using
    sentence_ID : int
        The sentence ID used in the MJFF dataset
    df : DataFrame
        The dataframe containing the typed sentences in the MJFF dataset.

    Returns
    -------
    List[np.array]
        List of embeddings, where the len(Returns) == df.shape[0] (i.e. one embedding per sentence)
    """
    assert 'Preprocessed_typed_sentence' in df.columns
    assert word_embedding_type in ['glove', 'extvec', 'crawl', 'twitter', 'turian', 'news']
    if sentence_ID is not None:
        assert sentence_ID in set(df.sentence_ID)

    warnings.warn("This method is currently only designed for English.")

    # Init standard character embeddings
    character_embeddings = CharacterEmbeddings()
    # Init standard GloVe embedding
    word_embedding = WordEmbeddings(word_embedding_type)
    # Instantitate word embedding method
    # TODO: we can expand this embedding operation substantially, by adding more types of word-embeddings
    sentence_embeddings = DocumentPoolEmbeddings([word_embedding,
                                                  character_embeddings])
    # Either we get embeddings for the whole dataset or just specific sentences
    if sentence_ID is not None:
        # All sentences with sentence_ID
        df_sentences = df.loc[df.sentence_ID == sentence_ID].Preprocessed_typed_sentence
    else:
        # All sentences
        df_sentences = df.Preprocessed_typed_sentence

    X = []  # Store embeddings here
    # TODO: check these two rows, they may not make any sense
    with torch.no_grad():
        sentence_embeddings.eval()
        for which_sent, typed_sentence in enumerate(df_sentences):
            if which_sent % 100 == 0:
                print("Embedding sentence nr: %i" % which_sent)
            sentence = Sentence(typed_sentence)
            # Embed
            sentence_embeddings.embed(sentence)
            # Make one tensor of all word embeddings of a sentence
            X.append(sentence.get_embedding().cpu().numpy())

    return X


def get_dynamic_sentence_embeddings():
    pass

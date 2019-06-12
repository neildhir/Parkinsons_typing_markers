
"""
Three models under consideration for sentence-classification (i.e. PD/HC classification)

1. BERT + basic classifier, this look _only_ at the text and nothing else at the first instance
2. Use ELMO/BERT embeddings and feed these into a classifier along with other information
3. Contextual string embeddings (most recent, has excellent performance)
"""

# Inspired by this example: https://github.com/kamujun/elmo_experiments/blob/master/elmo_experiment/notebooks/elmo_text_classification_on_imdb.ipynb

from allennlp.data.fields import TextField, MetadataField, LabelField, Field
from allennlp.data.dataset_readers import DatasetReader
from pathlib import Path
from typing import Callable, List, Dict, Optional, Iterator
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from functools import partial
from overrides import overrides

from allennlp.data import Instance
from allennlp.data.token_indexers.single_id_token_indexer import SingleIdTokenIndexer
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.nn import util as nn_util


class MJFFdatasetReader(DatasetReader):
    """
    Takes the preprocessed MJFF dataset (English) and turns it into proper tensors, for usage
    by classification model.

    Parameters
    ----------
    DatasetReader : AllenNLP class
        Base class for reading the data
    """

    def __init__(self,
                 tokenizer: Callable[[str], List[str]] = lambda x: x.split(),
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_seq_len: Optional[int] = 100,
                 testing: Optional[bool] = False) -> None:
        super().__init__(lazy=False)

        self.tokenizer = tokenizer
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.max_seq_len = max_seq_len
        self.testing = testing

    @overrides
    def text_to_instance(self,
                         tokens: List[Token],
                         id: str = None,
                         label: int = None) -> Instance:

        # Tokens from each typed sentence get stored here
        sentence_field = TextField(tokens, self.token_indexers)
        fields = {"tokens": sentence_field}
        # Carries the patient identifies
        id_field = MetadataField(id)
        fields["Patient_ID"] = id_field  # This is _NOT_ a tensor
        # Here we store the diagnosis as a categorical (PD/HC)
        label_field = LabelField(label, skip_indexing=True)
        fields["Diagnosis"] = label_field

        return Instance(fields)

    @overrides
    def _read(self, file_path: str) -> Iterator[Instance]:
        df = pd.read_csv(file_path)
        # Check that all is well by confirming that all fields are present in the dataset
        assert set(['Patient_ID', 'Diagnosis', 'Preprocessed_typed_sentence']
                   ) <= set(df.columns), print(df.columns)
        if self.testing:
            df = df.head(1000)
        for i, row in df.iterrows():
            yield self.text_to_instance(
                [Token(x) for x in self.tokenizer(row["Preprocessed_typed_sentence"])],  # Typed sentence
                row["Patient_ID"],  # Patient ID
                row["Diagnosis"],  # Diagnosis
            )

import os
from pathlib import Path

import torch
from flair.data import Sentence
from flair.datasets import ClassificationCorpus
from flair.embeddings import (BertEmbeddings, CharacterEmbeddings,
                              DocumentRNNEmbeddings, FlairEmbeddings,
                              WordEmbeddings)
from flair.hyperparameter.param_selection import (OptimizationValue, Parameter,
                                                  SearchSpace,
                                                  TextClassifierParamSelector)
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from hyperopt import hp

DATA_ROOT_FASTTEXT = Path("../data/MJFF/fasttext/")  # Note the relative path
RESULTS_ROOT = Path("../results/MJFF/")  # Note the relative path


def character_RNN(config,
                  language='english',
                  optimise=False):
    """
    Model to classify the long-format MJFF data, uses character-level embeddings
    to map to a numerical space.

    Parameters
    ----------
    config : [type]
        [description]
    language : str, optional
        [description], by default 'english'
    optimise : bool, optional
        [description], by default False

    Raises
    ------
    ValueError
        [description]
    """

    assert torch.cuda.is_available(), "Do not run this model with GPU support."
    # Specfiy the directory where all our results and data will be stored.
    lang_dir = Path(str(language) + "/")

    if language == 'english':
        # Note the name of the path (test, dev and train get identified automatically)
        corpus = ClassificationCorpus(DATA_ROOT_FASTTEXT / lang_dir)

        # Set word embeddings here _NOTE_: these are character-level embeddings
        word_embeddings = [CharacterEmbeddings()]

    else:
        raise ValueError("Language {} is not supported.".format(language))

    # Combine embeddings to make a "document"
    document_embeddings = DocumentRNNEmbeddings(embeddings=word_embeddings,
                                                hidden_size=config.hidden_size,
                                                rnn_layers=config.rnn_layers,
                                                reproject_words=config.reproject_words,
                                                reproject_words_dimension=config.reproject_words_dimension,
                                                bidirectional=config.bidirectional,
                                                dropout=config.dropout,
                                                word_dropout=config.word_dropout,
                                                locked_dropout=config.locked_dropout,
                                                rnn_type=config.rnn_type)

    # Classify said document using a TextClassifer
    classifier = TextClassifier(document_embeddings,
                                label_dictionary=corpus.make_label_dictionary(),
                                multi_label=False)

    # Specify a training instance
    trainer = ModelTrainer(classifier, corpus)

    # Train model
    trainer.train(RESULTS_ROOT / lang_dir,
                  learning_rate=config.learning_rate,
                  mini_batch_size=config.mini_batch_size,
                  patience=config.patience,
                  max_epochs=config.max_epochs)


def pdnet_mjff(config, language='english', optimise=False):

    assert torch.cuda.is_available(), "Do not run this model with GPU support."

    # English

    if language == 'english':
        language = Path(str(language) + "/")
        corpus = NLPTaskDataFetcher.load_classification_corpus(
            DATA_ROOT_FASTTEXT / language, test_file='test.csv', dev_file='dev.csv', train_file='train.csv')

        # Set word embeddings here
        word_embeddings = [WordEmbeddings('glove'),
                           CharacterEmbeddings(),
                           FlairEmbeddings('news-forward-fast'),
                           FlairEmbeddings('news-backward-fast')]

    elif language == 'english_turk':
        # English MJFF data with mechanical turk data as well
        language = Path("english/")
        corpus = NLPTaskDataFetcher.load_classification_corpus(
            DATA_ROOT_FASTTEXT / language, test_file='test_w_mt.csv', dev_file='dev_w_mt.csv', train_file='train_w_mt.csv')

        word_embeddings = [WordEmbeddings('glove'),
                           CharacterEmbeddings(),
                           FlairEmbeddings('news-forward-fast'),
                           FlairEmbeddings('news-backward-fast')]

    elif language == 'spanish':
        language = Path(str(language) + "/")
        corpus = NLPTaskDataFetcher.load_classification_corpus(
            DATA_ROOT_FASTTEXT / language, test_file='test.csv', dev_file='dev.csv', train_file='train.csv')

        word_embeddings = [CharacterEmbeddings(),
                           FlairEmbeddings('spanish-forward'),
                           FlairEmbeddings('spanish-backward')]

    elif language == 'mixed':
        # Use both English and Spanish MJFF data
        eng = Path(str("english") + "/")
        span = Path(str("spanish") + "/")
        corpus_eng = NLPTaskDataFetcher.load_classification_corpus(
            DATA_ROOT_FASTTEXT / eng, test_file='test.csv', dev_file='dev.csv', train_file='train.csv')

        corpus_span = NLPTaskDataFetcher.load_classification_corpus(
            DATA_ROOT_FASTTEXT / span, test_file='test.csv', dev_file='dev.csv', train_file='train.csv')

        # Combine corpuses, to form a multi-language corpus
        corpus = MultiCorpus([corpus_eng, corpus_span])

        # Combine Spanish and English
        word_embeddings = [
            #    WordEmbeddings('glove'),
            BertEmbeddings('bert-base-multilingual-cased'),
            #    FlairEmbeddings('news-forward-fast'),
            #    FlairEmbeddings('news-backward-fast')
        ]
    else:
        raise ValueError("Language {} is not supported.".format(language))

    # Combine embeddings to make a "document"
    document_embeddings = DocumentRNNEmbeddings(word_embeddings,
                                                hidden_size=config.hidden_size,
                                                reproject_words=True,
                                                reproject_words_dimension=128)

    # Classify said document using a TextClassifer
    classifier = TextClassifier(document_embeddings,
                                label_dictionary=corpus.make_label_dictionary(),
                                multi_label=False)

    # Specify a training instance
    trainer = ModelTrainer(classifier,
                           corpus)

    # Train model
    trainer.train(DATA_ROOT_FASTTEXT / language,
                  mini_batch_size=config.batch_size,
                  max_epochs=config.epochs)

# Auxiliary functions


def optimise_mjff_english_only():

    # Define the search space
    search_space = SearchSpace()
    search_space.add(Parameter.EMBEDDINGS, hp.choice, options=[
        [CharacterEmbeddings()],
        [WordEmbeddings('glove'), FlairEmbeddings('news-forward'), FlairEmbeddings('news-backward')],
        [WordEmbeddings('glove'), FlairEmbeddings('news-forward-fast'), FlairEmbeddings('news-backward-fast')]
    ])
    search_space.add(Parameter.HIDDEN_SIZE, hp.choice, options=[32, 64, 128, 256])
    search_space.add(Parameter.RNN_LAYERS, hp.choice, options=[1, 2, 3])
    search_space.add(Parameter.DROPOUT, hp.uniform, low=0.0, high=0.5)
    search_space.add(Parameter.LEARNING_RATE, hp.choice, options=[0.05, 0.1, 0.15, 0.2])
    search_space.add(Parameter.MINI_BATCH_SIZE, hp.choice, options=[8, 16, 32])

    language = Path("english/")
    corpus = NLPTaskDataFetcher.load_classification_corpus(
        DATA_ROOT_FASTTEXT / language, test_file='test.csv', dev_file='dev.csv', train_file='train.csv')

    # create the parameter selector
    param_selector = TextClassifierParamSelector(
        corpus,
        False,
        DATA_ROOT_FASTTEXT / language,
        'lstm',
        max_epochs=10,
        training_runs=3,
        optimization_value=OptimizationValue.DEV_SCORE)

    # start the optimization
    param_selector.optimize(search_space, max_evals=100)


def make_data_flair_readable(data):
    """
    Function to create a pandas dataframe that can properly be read
    by flair's sentence classifier.

    Parameters
    ----------
    data : pandas dataframe
        Dataframe containing all the labelled sentences

    Returns
    -------
    pandas dataframe
        Dataframe containing all the labelled sentences but flair enabled
    """

    # Special column labelling required
    data = data[['Diagnosis',
                 'Preprocessed_typed_sentence']].rename(columns={"Diagnosis": "label",
                                                                 "Preprocessed_typed_sentence": "text"})
    # We need to make this reformating since flair is based on Facebook's FastText format which requires the labels to have this appearence.
    data['label'] = '__label__' + data['label'].astype(str)
    return data


def make_train_test_dev(df,
                        separator,
                        save_dir):
    """
    Function to create train, test and dev datasets for the FLAIR model.

    Parameters
    ----------
    df : pandas dataframe
        Pandas dataframe which contains sentences and labels
    separator : str
        Which symbol we use the separate the labels and the sentences
    save_dir : str
        Path to the save location
    """

    # Shuffle the rows of the dataframe
    df = df.sample(frac=1).reset_index(drop=True)
    len_df = len(df)

    # Create train, test and dev datasets
    df.iloc[0:int(len_df*0.8)].to_csv(save_dir + 'train.csv',
                                      sep=separator,
                                      index=False,
                                      header=False)
    df.iloc[int(len_df*0.8):int(len_df*0.9)].to_csv(save_dir + 'test.csv',
                                                    sep=separator,
                                                    index=False,
                                                    header=False)
    df.iloc[int(len_df*0.9):].to_csv(save_dir + 'dev.csv',
                                     sep=separator,
                                     index=False,
                                     header=False)


class Config(dict):
    """
    Simple class to hold the model and training values.

    Example
    -------
    config = Config(
        testing=True,
        seed=1,
        batch_size=8,
        lr=1e-4,
        bidirectional=True,
        patience=3,
        epochs=10,
        hidden_size=256,
    )
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)

    def set(self, key, val):
        self[key] = val
        setattr(self, key, val)

from torch.utils.data import DataLoader, Dataset
import torch
import json
import csv


class MJFF_original(Dataset):
    def __init__(self,
                 label_data_path: str,
                 alphabet_path: str,
                 max_sample_length: int,
                 load_very_long_sentences: bool = False):
        """
        Create MJFF dataset object.

        Parameters
        ----------
        label_data_path : str
            The path of label and data file in csv format
        alphabet_path : str
            Max length of a sample
        max_sample_length : int
           The path of alphabet json file
        do_not_load_very_long_sentences: bool, optional
            Parameter tells method to filter out sentences that are above the threshold
        """
        self.label_data_path = label_data_path
        self.max_sample_length = max_sample_length
        self.load_very_long_sentences = load_very_long_sentences
        # read alphabet
        self.load_alphabet(alphabet_path)
        self.load(label_data_path)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        X = self.one_hot_encode_chars_in_sentence(idx)
        y = self.y[idx]
        return X, y

    def load_alphabet(self, alphabet_path):
        with open(alphabet_path) as f:
            self.alphabet = ''.join(json.load(f))

    def load(self, label_data_path, lowercase=True):
        self.label = []
        self.data = []
        with open(label_data_path, 'r') as f:
            rdr = csv.reader(f, delimiter=',', quotechar='"')
            # num_samples = sum(1 for row in rdr)
            for index, row in enumerate(rdr):
                txt = ' '.join(row[1:])
                if lowercase:
                    txt = txt.lower()
                # XXX: temporary fix for loading too long sentences
                if self.load_very_long_sentences is False:
                    if len(txt) <= self.max_sample_length:
                        self.label.append(int(row[0]))  # Labels
                        self.data.append(txt)  # Typed sentence
                elif self.load_very_long_sentences:
                    self.label.append(int(row[0]))
                    self.data.append(txt)

        self.y = torch.LongTensor(self.label)

    def one_hot_encode_chars_in_sentence(self, idx):
        # X = (batch, 70, sequence_length)
        X = torch.zeros(len(self.alphabet), self.max_sample_length)
        sequence = self.data[idx]
        for index_char, char in enumerate(sequence[::-1]):
            if self.char2index(char) != -1:
                X[self.char2index(char)][index_char] = 1.0
        return X

    def char2index(self, character):
        return self.alphabet.find(character)

    def get_class_weight(self):
        num_samples = self.__len__()
        label_set = set(self.label)
        num_class = [self.label.count(c) for c in label_set]
        class_weight = [num_samples/float(self.label.count(c)) for c in label_set]
        return class_weight, num_class


def save_checkpoint(model, state, filename):
    model_is_cuda = next(model.parameters()).is_cuda
    # model = model.module if model_is_cuda else model  # Only use if we have parallel GPUs available
    state['state_dict'] = model.state_dict()
    torch.save(state, filename)


def count_trainable_parameters(model):
    """Number of trainable model-parameters.

    Parameters
    ----------
    model : PyTorch model
        A predefined pytorch model

    Returns
    -------
    int
        The total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_MJFF_data_loader(dataset_path: str,
                          alphabet_path: str,
                          max_sample_length: int,
                          load_very_long_sentences: bool,
                          batch_size: int,
                          num_workers: int):
    """
    Function to create a Torch readable data object.

    Parameters
    ----------
    dataset_path : str
        [description]
    alphabet_path : str
        [description]
    max_sample_length : int
        [description]
    batch_size : int
        [description]
    num_workers : int
        [description]
    """
    print("Loading data from {}".format(dataset_path))
    dataset = MJFF_original(label_data_path=dataset_path,
                            alphabet_path=alphabet_path,
                            max_sample_length=max_sample_length, load_very_long_sentences=load_very_long_sentences)

    # Torch dataloader created here
    dataset_loader = DataLoader(dataset,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                drop_last=True,
                                shuffle=True)

    return dataset, dataset_loader

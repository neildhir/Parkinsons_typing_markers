"""
CNNs for text-classification and character-based CNNs for language modelling.

Inspiration: https://github.com/srviest/char-cnn-text-classification-pytorch
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# TODO: add attention (maybe): https://mlwhiz.com/blog/2019/03/09/deeplearning_architectures_text_classification/
# TODO: add grad cam: https://medium.com/@stepanulyanin/implementing-grad-cam-in-pytorch-ea0937c31e82


class SimpleCharCNN(nn.Module):
    """
    [18/12/19] : new model for prototyping the character CNN approach.
    """

    def __init__(self, args):
        super(SimpleCharCNN, self).__init__()
        """
        Addendums for convolutions:

        - kernel_size: anything you want. In Zhang's picture, there are 6 different filters with sizes 4, 4, 3, 3, 2, 2, represented by the "vertical" dimension.

        - in_channels (automatically the embedding_size): the size of the embedding - this is somwehat mandatory (in Keras this is automatic and almost invisible), otherwise the multiplications wouldn't use the entire embedding, which is pointless. In the picture, the "horizontal" dimension of the filters is constantly 5 (the same as the word size - this is not a spatial dimension).

        - output_channels (filters): anything you want, but it seems the picture is talking about 1 channel only per filter, since it's totally ignored, and if represented would be something like "depth".
        """
        self.convolution_layer_1 = nn.Sequential(
            # Number of features is the size of the alphabet
            nn.Conv1d(in_channels=args.alphabet_size, out_channels=16, kernel_size=8, stride=1),
            nn.ReLU(),
            # Non-overlapping pooling note
            nn.MaxPool1d(kernel_size=8, stride=8),  # XXX: why max pool here and not other pooling
        )
        self.fully_connected_1 = nn.Linear(2, 1)
        self.log_softmax = nn.LogSoftmax()

    def forward(self, x):
        x = self.convolution_layer_1(x)
        # collapse
        x = x.view(x.size(0), -1)
        x = self.fully_connected_1(x)
        x = self.log_softmax(x)  # TODO: check purpose

        return x


class CharCNN(nn.Module):
    """
    Implementation: https://github.com/srviest/char-cnn-text-classification-pytorch
    Paper: Character-level Convolutional Networks for Text Classification
    """

    # def __init__(self, args):
    #     super(CharCNN, self).__init__()
    #     self.conv1 = nn.Sequential( # OLD
    #         nn.Conv1d(args.num_features, 256, kernel_size=7, stride=1),
    #         nn.ReLU(),
    #         nn.MaxPool1d(kernel_size=3, stride=3)
    #     )

    #     self.conv2 = nn.Sequential(
    #         nn.Conv1d(256, 256, kernel_size=7, stride=1),
    #         nn.ReLU(),
    #         nn.MaxPool1d(kernel_size=3, stride=3)
    #     )

    #     self.conv3 = nn.Sequential(
    #         nn.Conv1d(256, 256, kernel_size=3, stride=1),
    #         nn.ReLU()
    #     )

    #     self.conv4 = nn.Sequential(
    #         nn.Conv1d(256, 256, kernel_size=3, stride=1),
    #         nn.ReLU()
    #     )

    #     self.conv5 = nn.Sequential(
    #         nn.Conv1d(256, 256, kernel_size=3, stride=1),
    #         nn.ReLU()
    #     )

    #     self.conv6 = nn.Sequential(
    #         nn.Conv1d(256, 256, kernel_size=3, stride=1),
    #         nn.ReLU(),
    #         nn.MaxPool1d(kernel_size=3, stride=3)
    #     )

    def __init__(self, args):
        super(CharCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(args.num_features, 64, kernel_size=16, stride=1), nn.ReLU(), nn.MaxPool1d(kernel_size=8, stride=8)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=16, stride=1), nn.ReLU(), nn.MaxPool1d(kernel_size=4, stride=4)
        )
        self.fc1 = nn.Sequential(nn.Linear(65408, 512), nn.ReLU(), nn.Dropout(p=args.dropout))
        # self.fc2 = nn.Sequential(
        #     nn.Linear(1024, 512),
        #     nn.ReLU(),
        #     nn.Dropout(p=args.dropout)
        # )

        self.fc2 = nn.Linear(512, 2)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        # x = self.conv2(x)
        # x = self.conv3(x)
        # x = self.conv4(x)
        # x = self.conv5(x)
        # x = self.conv6(x)

        # collapse
        x = x.view(x.size(0), -1)
        # linear layer
        x = self.fc1(x)
        # linear layer
        # x = self.fc2(x)
        # linear layer
        x = self.fc2(x)

        # output layer
        x = self.log_softmax(x)  # We should consider looking into other options here

        return x


class CNN_Text(nn.Module):
    """
    Implementation: https://github.com/Shawn1993/cnn-text-classification-pytorch
    Paper: "Convolutional Neural Networks for Sentence Classification"
    """

    def __init__(self, args):
        super(CNN_Text, self).__init__()
        self.args = args

        V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        Ci = 1
        Co = args.kernel_num
        Ks = args.kernel_sizes

        self.embed = nn.Embedding(V, D)
        # self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        """
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        """
        self.dropout = nn.Dropout(args.dropout)
        self.fc1 = nn.Linear(len(Ks) * Co, C)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        x = self.embed(x)  # (N, W, D)

        if self.args.static:
            x = Variable(x)

        x = x.unsqueeze(1)  # (N, Ci, W, D)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)

        """
        x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
        x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
        x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
        x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
        """
        x = self.dropout(x)  # (N, len(Ks)*Co)
        logit = self.fc1(x)  # (N, C)
        return logit


class Highway(nn.Module):
    """Highway network
    https://github.com/lucko515/deep-convolutional-highway-network
    http://www.cs.toronto.edu/~fidler/teaching/2015/slides/CSC2523/renjie_highwayNNs.pdf
    --very useful for training many multi-layered CNN
    """

    def __init__(self, input_size):
        super(Highway, self).__init__()
        self.fc1 = nn.Linear(input_size, input_size, bias=True)
        self.fc2 = nn.Linear(input_size, input_size, bias=True)

    def forward(self, x):
        t = F.sigmoid(self.fc1(x))
        return torch.mul(t, F.relu(self.fc2(x))) + torch.mul(1 - t, x)


class charLM(nn.Module):
    """
    Implementation: https://github.com/FengZiYjun/CharLM
    Paper: "Character-Aware Neural Language Models"

    CNN + highway network + LSTM

    # Input:
        4D tensor with shape [batch_size, in_channel, height, width]
    # Output:
        2D Tensor with shape [batch_size, vocab_size]
    # Arguments:
        char_emb_dim: the size of each character's embedding
        word_emb_dim: the size of each word's embedding
        vocab_size: num of unique words
        num_char: num of characters
        use_gpu: True or False
    """

    def __init__(self, char_emb_dim, word_emb_dim, vocab_size, num_char, use_gpu):
        super(charLM, self).__init__()
        self.char_emb_dim = char_emb_dim
        self.word_emb_dim = word_emb_dim
        self.vocab_size = vocab_size

        # char embedding layer
        self.char_embed = nn.Embedding(num_char, char_emb_dim)

        # convolutions of filters with different sizes
        self.convolutions = []

        # list of tuples: (the number of filter, width)
        self.filter_num_width = [(25, 1), (50, 2), (75, 3), (100, 4), (125, 5), (150, 6)]

        for out_channel, filter_width in self.filter_num_width:
            self.convolutions.append(
                nn.Conv2d(
                    1,  # in_channel
                    out_channel,  # out_channel
                    kernel_size=(char_emb_dim, filter_width),  # (height, width)
                    bias=True,
                )
            )

        self.highway_input_dim = sum([x for x, y in self.filter_num_width])

        self.batch_norm = nn.BatchNorm1d(self.highway_input_dim, affine=False)

        # highway net
        self.highway1 = Highway(self.highway_input_dim)
        self.highway2 = Highway(self.highway_input_dim)

        # LSTM
        self.lstm_num_layers = 2

        self.lstm = nn.LSTM(
            input_size=self.highway_input_dim,
            hidden_size=self.word_emb_dim,
            num_layers=self.lstm_num_layers,
            bias=True,
            dropout=0.5,
            batch_first=True,
        )

        # output layer
        self.dropout = nn.Dropout(p=0.5)
        self.linear = nn.Linear(self.word_emb_dim, self.vocab_size)

        if use_gpu is True:
            for x in range(len(self.convolutions)):
                self.convolutions[x] = self.convolutions[x].cuda()
            self.highway1 = self.highway1.cuda()
            self.highway2 = self.highway2.cuda()
            self.lstm = self.lstm.cuda()
            self.dropout = self.dropout.cuda()
            self.char_embed = self.char_embed.cuda()
            self.linear = self.linear.cuda()
            self.batch_norm = self.batch_norm.cuda()

    def forward(self, x, hidden):
        # Input: Variable of Tensor with shape [num_seq, seq_len, max_word_len+2]
        # Return: Variable of Tensor with shape [num_words, len(word_dict)]
        lstm_batch_size = x.size()[0]
        lstm_seq_len = x.size()[1]

        x = x.contiguous().view(-1, x.size()[2])
        # [num_seq*seq_len, max_word_len+2]

        x = self.char_embed(x)
        # [num_seq*seq_len, max_word_len+2, char_emb_dim]

        x = torch.transpose(x.view(x.size()[0], 1, x.size()[1], -1), 2, 3)
        # [num_seq*seq_len, 1, max_word_len+2, char_emb_dim]

        x = self.conv_layers(x)
        # [num_seq*seq_len, total_num_filters]

        x = self.batch_norm(x)
        # [num_seq*seq_len, total_num_filters]

        x = self.highway1(x)
        x = self.highway2(x)
        # [num_seq*seq_len, total_num_filters]

        x = x.contiguous().view(lstm_batch_size, lstm_seq_len, -1)
        # [num_seq, seq_len, total_num_filters]

        x, hidden = self.lstm(x, hidden)
        # [seq_len, num_seq, hidden_size]

        x = self.dropout(x)
        # [seq_len, num_seq, hidden_size]

        x = x.contiguous().view(lstm_batch_size * lstm_seq_len, -1)
        # [num_seq*seq_len, hidden_size]

        x = self.linear(x)
        # [num_seq*seq_len, vocab_size]
        return x, hidden

    def conv_layers(self, x):
        chosen_list = list()
        for conv in self.convolutions:
            feature_map = F.tanh(conv(x))
            # (batch_size, out_channel, 1, max_word_len-width+1)
            chosen = torch.max(feature_map, 3)[0]
            # (batch_size, out_channel, 1)
            chosen = chosen.squeeze()
            # (batch_size, out_channel)
            chosen_list.append(chosen)

        # (batch_size, total_num_filers)
        return torch.cat(chosen_list, 1)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        # We don't want to train the resnet itself.
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        # The last fully connected layer is used for prediction, so we drop it.
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        features = self.bn(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, dropout=0):
        super().__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, 
                            dropout=dropout if num_layers > 1 else 0,
                            batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        
        # TODO: Dig into nn.LogSoftmax() here
        self.softmax = nn.Softmax(dim=2)

    def forward(self, features, captions):
        batch_size = features.size(0)
        
        captions_without_end = captions[:, :-1]
        captions = self.embed(captions_without_end)
        
        """Concatenate the features and caption inputs and feed to LSTM cell(s).
        permutation is not required due to batch_first=True."""
        features = features.unsqueeze(1)
        inputs = torch.cat((features, captions), 1)
        lstm_output, _ = self.lstm(inputs, None)
        
        # Convert LSTM outputs to word predictions
        outputs = self.linear(lstm_output)
        return outputs

    def sample(self, inputs, states=None, max_len=20, stop_idx=1):
        "Accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        sentence = []
        lstm_state = None
        for _ in range(max_len):
            lstm_out, lstm_state = self.lstm(inputs, lstm_state)
            output = self.linear(lstm_out)

            # Get the predicted word
            # TODO: Sample stochastically or perform a beam search
            prediction = torch.argmax(output, dim=2)
            predicted_index = prediction.item()
            sentence.append(predicted_index)
            
            # TODO: Training needs to include the STOP index, otherwise it won't be emitted.
            if predicted_index == stop_idx:
                break
            
            # Get the embeddings for the next cycle.
            inputs = self.embed(prediction)

        return sentence
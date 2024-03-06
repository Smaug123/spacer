import torch
import torch.nn as nn


def save_checkpoint(model, optimizer, epoch, filename="checkpoint.pth"):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        # Add more information here if needed
    }
    torch.save(checkpoint, filename)


def load_checkpoint(model, optimizer, filename="checkpoint.pth"):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return epoch


# Define the LSTM model
class SpacingLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, bidirectional=True):
        super(SpacingLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, bidirectional=bidirectional, batch_first=True)
        self.fc = nn.Linear(hidden_size * (2 if bidirectional else 1), output_size)

    def forward(self, input_seq):
        embedded = self.embedding(input_seq)
        output, _ = self.lstm(embedded)
        output = self.fc(output)

        return output

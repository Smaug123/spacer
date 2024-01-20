from typing import Tuple, Iterable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import argparse
import os


class SpaceInsertionModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward,
                 max_seq_length):
        super(SpaceInsertionModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_length, d_model))
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward,
                                          batch_first=True)
        self.output_linear = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        src = self.embedding(src) + self.positional_encoding[:, :src.size(1), :]
        tgt = self.embedding(tgt) + self.positional_encoding[:, :tgt.size(1), :]
        output = self.transformer(src, tgt)
        return self.output_linear(output)


# original_strings = ["hi bye", "the world is huge", ...]
def construct_dataset(original_strings: list[str]) -> Tuple[list[str], list[str]]:
    input_sequences = []
    target_sequences = []

    for string in original_strings:
        target_sequences.append(string)
        input_sequences.append(string.replace(" ", ""))

    return input_sequences, target_sequences


def create_char_to_index_map(sequences: list[str]) -> dict[str, int]:
    chars = set(char for seq in sequences for char in seq)
    return {char: i for i, char in enumerate(sorted(chars))}


def map_chars_to_indices(sequences: Iterable[str], char_to_index: dict[str, int]) -> Iterable[list[int]]:
    return ([char_to_index[char] for char in seq] for seq in sequences)


def pad_sequences(sequences: Iterable[list[int]], max_length: int) -> Iterable[list[int]]:
    return (seq + [0] * (max_length - len(seq)) for seq in sequences)


def to_tensor(s: str, char_to_index_map: dict[str, int]) -> torch.tensor:
    chars_to_indices = list(pad_sequences(map_chars_to_indices([s], char_to_index_map), 100))
    return torch.tensor(chars_to_indices, dtype=torch.long).to('mps')


def make_loader(english_sample: list[str], char_to_index_map: dict[str, int]) -> DataLoader:
    input_sequences, target_sequences = construct_dataset(english_sample)
    print("Input Sequence:", input_sequences[0])  # "hibye"
    print("Target Sequence:", target_sequences[0])  # "hi bye"

    input_sequences_tokenized = list(map_chars_to_indices(input_sequences, char_to_index_map))
    target_sequences_tokenized = list(map_chars_to_indices(target_sequences, char_to_index_map))

    max_length_input = max(len(seq) for seq in input_sequences_tokenized)
    max_length_target = max(len(seq) for seq in target_sequences_tokenized)
    max_length = max(max_length_input, max_length_target)

    input_sequences_padded = list(pad_sequences(input_sequences_tokenized, max_length))
    target_sequences_padded = list(pad_sequences(target_sequences_tokenized, max_length))

    input_sequences_tensor = torch.tensor(input_sequences_padded, dtype=torch.long).to('mps')
    target_sequences_tensor = torch.tensor(target_sequences_padded, dtype=torch.long).to('mps')

    # Assuming `input_sequences` and `target_sequences` are your tokenized and padded data
    train_data = TensorDataset(input_sequences_tensor, target_sequences_tensor)
    return DataLoader(train_data, batch_size=32, shuffle=True)


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


def convert_to_readable(output: torch.Tensor, token_to_index: dict[str, int]):
    index_to_token = {index: token for token, index in token_to_index.items()}
    # Apply softmax to convert logits to probabilities
    probabilities = torch.nn.functional.softmax(output, dim=-1)

    # Find the token with the highest probability
    # The output is likely in the shape [sequence length, batch size, vocab size]
    # We are interested in the most likely token at each position in the sequence
    most_likely_tokens = probabilities.argmax(dim=-1)

    # Convert indices to tokens
    readable_tokens = [index_to_token[idx] for idx in most_likely_tokens.tolist()]

    # Join tokens into a single string or handle as needed
    readable_output = ' '.join(readable_tokens)
    return readable_output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a space insertion bot")
    parser.add_argument('--input', type=str, required=True, help='Path to the folder to scrape for text files')
    args = parser.parse_args()

    file_contents = []
    for filename in os.listdir(args.input):
        filepath = os.path.join(args.input, filename)
        # Check if it's a file and not a directory
        if os.path.isfile(filepath) and filepath.endswith(".txt"):
            with open(filepath, 'r') as file:
                content = file.read()
                file_contents.append(content.replace('\n', ' ').replace('  ', ' '))

    # Split up the text at random into 100-char-or-less sequences
    english_sample = []
    for content in file_contents:
        for i in range(0, len(content), 100):
            english_sample.append(content[i:i + 100])

    char_to_index_map = create_char_to_index_map(english_sample)

    train_loader = make_loader(english_sample, char_to_index_map)

    model = SpaceInsertionModel(
        vocab_size=100,  # Size of your vocabulary
        d_model=512,  # Size of embeddings and transformer hidden size
        nhead=8,  # Number of attention heads
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        max_seq_length=100  # Max sequence length
    ).to('mps')

    criterion = nn.CrossEntropyLoss()  # Cross-Entropy Loss is commonly used for classification tasks
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer is a good starting point

    # Training
    num_epochs = 10
    start_epoch = -1
    checkpoint_path = "checkpoint.pth"

    # If a checkpoint exists, load it
    if os.path.exists(checkpoint_path):
        print("Reloading checkpoint")
        start_epoch = load_checkpoint(model, optimizer, checkpoint_path)
        print(f"Start epoch: {start_epoch}")

    for epoch in range(start_epoch + 1, num_epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()

            input_seq, target_seq = batch
            output = model(input_seq, target_seq)

            target = target_seq

            # Calculate loss
            loss = criterion(output, target)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        save_checkpoint(model, optimizer, epoch, checkpoint_path)
        print(f'Epoch {epoch}, Loss: {total_loss / len(train_loader)}')

    # Validation:
    print('Validation')
    model.eval()
    total_loss = 0


    def pad(s: str) -> str:
        return s + " " * (100 - len(s))


    english_sample_validation = [pad("hi bye"), pad("the world is huge")]
    validation_loader = make_loader(english_sample_validation, char_to_index_map)

    with torch.no_grad():
        for batch in validation_loader:
            input_seq, target_seq = batch
            output = model(input_seq, target_seq)
            target = target_seq

            # Calculate loss
            loss = criterion(output, target)

            total_loss += loss.item()

    print(f'Validation Loss: {total_loss / len(validation_loader)}')

    # Sample
    model.eval()
    with torch.no_grad():
        input_seq = to_tensor(pad("inthebeginningwasthewordandthewordwaswithgod"), char_to_index_map)
        print(input_seq)
        dummy_tgt = torch.zeros_like(input_seq)
        output = model(input_seq, dummy_tgt)[0]
        # print dimension of output tensor
        print(convert_to_readable(output, char_to_index_map))

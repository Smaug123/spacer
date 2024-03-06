import os
import sys

_i = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _i)
import model
import dataset

sys.path.remove(_i)

import re
import torch
from torch import nn
import argparse

SAMPLE_LEN = 100

DEVICE = 'mps'


def pad(s: str) -> str:
    return s + " " * (SAMPLE_LEN - len(s))


def chunk_data(raw: list[str]):
    result = []
    sample = ""
    for content in raw:
        content = re.sub(r'[^a-zA-Z ]', '', content.replace('\n', ' ')).replace('  ', ' ')
        for i in range(0, len(content), SAMPLE_LEN):
            result.append(content[i:i + SAMPLE_LEN - 1])
            if not sample:
                sample = result[-1]
    print(f"Sample: {sample}")
    return result


def training_data(input_dir: os.PathLike):
    file_contents = []
    for filename in os.listdir(input_dir):
        filepath = os.path.join(input_dir, filename)
        if os.path.isfile(filepath) and filepath.endswith(".txt"):
            with open(filepath, 'r') as file:
                content = file.read()
                file_contents.append(content)
    print(f"Number of samples: {len(file_contents)}")

    # Split up the text at random into (SAMPLE_LEN-1)-char-or-less sequences
    english_sample = chunk_data(file_contents)

    char_to_index_map = dataset.create_char_to_index_map(english_sample)

    return english_sample, char_to_index_map


def train(training_data, char_to_index_map, num_epochs=64):
    train_loader = dataset.make_loader(training_data, char_to_index_map, SAMPLE_LEN, DEVICE)

    # Instantiate the model
    input_size = len(char_to_index_map)  # Size of the character vocabulary
    hidden_size = 256  # Hidden size of the LSTM
    output_size = 2  # Binary classification (space or no space)
    my_model = model.SpacingLSTM(input_size, hidden_size, output_size).to(DEVICE)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(my_model.parameters())

    # Training

    start_epoch = -1
    checkpoint_path = "checkpoint.pth"

    # If a checkpoint exists, load it
    if not args.start_afresh and os.path.exists(checkpoint_path):
        print("Reloading checkpoint")
        start_epoch = model.load_checkpoint(my_model, optimizer, checkpoint_path)
        print(f"Start epoch: {start_epoch}")

    # Training loop
    for epoch in range(start_epoch + 1, num_epochs):
        total_loss = 0
        for input_seq, target_seq in train_loader:
            # Forward pass
            output = my_model(input_seq)
            loss = criterion(output.view(-1, output_size), target_seq.view(-1))

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.save_checkpoint(my_model, optimizer, epoch, checkpoint_path + ".new")
        os.rename(checkpoint_path + ".new", checkpoint_path)
        print(f'Epoch {epoch}, Loss: {total_loss / len(train_loader)}')

    return my_model


def inference(my_model, target: str):
    target = chunk_data([target])
    unspaced, _ = dataset.construct_dataset(target)
    for sample in unspaced:
        with torch.no_grad():
            my_model.eval()
            input_seq = dataset.to_tensor(sample, char_to_index_map, SAMPLE_LEN, DEVICE)
            output = my_model(input_seq)
            output = output.argmax(dim=2)
            spaced_text = dataset.convert_to_readable(input_seq, output, char_to_index_map)
            print(spaced_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a space insertion bot")
    parser.add_argument('--input', type=str, required=True, help='Path to the folder to scrape for text files')
    parser.add_argument('--start-afresh', action='store_true', default=False,
                        help='True to ignore previous checkpoints')
    args = parser.parse_args()
    training_data, char_to_index_map = training_data(args.input)
    model = train(training_data=training_data, char_to_index_map=char_to_index_map)
    inference(model,
              "The way you can go isn't the real way; the name you can say isn't the real name. Heaven and Earth begin in the unnamed; name's the mother of the ten thousand things.")

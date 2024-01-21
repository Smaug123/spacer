import os
import sys

_i = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _i)
import model
import dataset

sys.path.remove(_i)

import re
import torch
from torch import nn, optim
import argparse

# char ALPHABET_SIZE - 1 is the end-of-string terminator.
ALPHABET_SIZE = 100
SAMPLE_LEN = 100


def pad(s: str) -> str:
    return s + " " * (SAMPLE_LEN - len(s))


def main(args):

    sample = ""

    file_contents = []
    for filename in os.listdir(args.input):
        filepath = os.path.join(args.input, filename)
        if os.path.isfile(filepath) and filepath.endswith(".txt"):
            with open(filepath, 'r') as file:
                content = file.read()
                sanitised = re.sub(r'[^a-zA-Z ]', '', content.replace('\n', ' ')).replace('  ', ' ')
                file_contents.append(sanitised[0:200])
                sample = sanitised[0:100]
    print(f"Sample: {sample}")

    # Split up the text at random into (SAMPLE_LEN-1)-char-or-less sequences
    english_sample = []
    for content in file_contents:
        for i in range(0, len(content), SAMPLE_LEN):
            english_sample.append(content[i:i + SAMPLE_LEN - 1])

    char_to_index_map = dataset.create_char_to_index_map(english_sample)

    train_loader = dataset.make_loader(english_sample, char_to_index_map, SAMPLE_LEN)

    my_model = model.SpaceInsertionModel(
        vocab_size=ALPHABET_SIZE,  # Size of your vocabulary
        d_model=128,  # Size of embeddings and transformer hidden size
        nhead=8,  # Number of attention heads
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=512,
        max_seq_length=SAMPLE_LEN  # Max sequence length
    ).to('mps')

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(my_model.parameters(), lr=0.0001)

    # Training
    num_epochs = 100
    start_epoch = -1
    checkpoint_path = "checkpoint.pth"

    # If a checkpoint exists, load it
    if not args.start_afresh and os.path.exists(checkpoint_path):
        print("Reloading checkpoint")
        start_epoch = model.load_checkpoint(my_model, optimizer, checkpoint_path)
        print(f"Start epoch: {start_epoch}")

    for epoch in range(start_epoch + 1, num_epochs):
        my_model.train()
        total_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()

            input_seq, target_seq = batch
            output = my_model(input_seq, target_seq)

            # Calculate loss
            loss = criterion(output, target_seq.float())

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        model.save_checkpoint(my_model, optimizer, epoch, checkpoint_path)
        print(f'Epoch {epoch}, Loss: {total_loss / len(train_loader)}')

    index_to_token = {index: token for token, index in char_to_index_map.items()}

    my_model.eval()
    example = sample.replace(' ', '')
    output_sequence = []
    with torch.no_grad():
        input_seq = dataset.to_tensor(example, char_to_index_map, SAMPLE_LEN)
        print(input_seq)
        tgt = torch.zeros_like(input_seq)
        for i in range(1, len(input_seq[0])):
            tgt[0, i] = 1
        for i in range(len(input_seq[0])):
            output = my_model(input_seq, tgt)[0].sigmoid()
            print(f"{i}: {output}")
            output_sequence.append(index_to_token[input_seq[0, i].item()])
            if output[i].item() > 0.5:
                output_sequence.append(' ')
                tgt[0, i] = 1
            else:
                tgt[0, i] = 0
        print(''.join(output_sequence))
        # print(convert_to_readable(input_seq[0], output, char_to_index_map))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a space insertion bot")
    parser.add_argument('--input', type=str, required=True, help='Path to the folder to scrape for text files')
    parser.add_argument('--start-afresh', action='store_true', default=False,
                        help='True to ignore previous checkpoints')
    main(parser.parse_args())

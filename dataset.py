from typing import Tuple, Iterable

import torch
from torch.utils.data import DataLoader, TensorDataset


# original_strings = ["hi bye", "the world is huge", ...]
# output: ["hibye", "theworldishuge"], [[0,1,0,0,1], [...]]
def construct_dataset(original_strings: list[str]) -> Tuple[list[str], list[list[int]]]:
    input_sequences = []
    target_sequences = []

    for string in original_strings:
        target_sequence = []
        input_sequence = []
        count = 0
        while count < len(string):
            while count < len(string) and string[count] == ' ':
                count += 1
            if count >= len(string):
                break
            if count == len(string) - 1:
                input_sequence.append(string[count])
                target_sequence.append(1)
                break

            input_sequence.append(string[count])
            if string[count + 1] == ' ':
                target_sequence.append(1)
            else:
                target_sequence.append(0)

            count += 1

        input_sequences.append(''.join(input_sequence))
        target_sequences.append(target_sequence)

    return input_sequences, target_sequences


def create_char_to_index_map(sequences: list[str]) -> dict[str, int]:
    chars = set(char for seq in sequences for char in seq if char != ' ')
    result = {char: i + 1 for i, char in enumerate(sorted(chars))}
    result[' '] = 0
    return result


def map_chars_to_indices(sequences: Iterable[str], char_to_index: dict[str, int]) -> Iterable[list[int]]:
    return ([char_to_index[char] for char in seq] for seq in sequences)


def pad_sequences(sequences: Iterable[list[int]], max_length: int) -> Iterable[list[int]]:
    return (seq + [0] * (max_length - len(seq)) for seq in sequences)


def to_tensor(s: str, char_to_index_map: dict[str, int], max_length: int, device: str) -> torch.tensor:
    chars_to_indices = list(pad_sequences(map_chars_to_indices([s], char_to_index_map), max_length))
    return torch.tensor(chars_to_indices, dtype=torch.long).to(device)


def make_loader(english_sample: list[str], char_to_index_map: dict[str, int], sample_len: int,
                device: str) -> DataLoader:
    input_sequences, target_sequences = construct_dataset(english_sample)
    print("Input Sequence:", input_sequences[0])  # "hibye"
    print("Target Sequence:", target_sequences[0])  # [0,1,0,0,1]

    input_sequences_tokenized = list(map_chars_to_indices(input_sequences, char_to_index_map))

    print('Tokenised:', input_sequences_tokenized[0])

    input_sequences_padded = list(pad_sequences(input_sequences_tokenized, sample_len))
    target_sequences_padded = list(pad_sequences(target_sequences, sample_len))
    print('Padded:', input_sequences_padded[0])
    print('Padded target:', target_sequences_padded[0])

    input_sequences_tensor = torch.tensor(input_sequences_padded, dtype=torch.long).to(device)
    target_sequences_tensor = torch.tensor(target_sequences_padded, dtype=torch.long).to(device)

    # Assuming `input_sequences` and `target_sequences` are your tokenized and padded data
    train_data = TensorDataset(input_sequences_tensor, target_sequences_tensor)
    return DataLoader(train_data, batch_size=32, shuffle=True)


def convert_to_readable(input_seq: torch.Tensor, output: torch.Tensor, token_to_index: dict[str, int]):
    index_to_token = {index: token for token, index in token_to_index.items()}
    result = []
    for i in range(input_seq.size(0)):
        input_chars = [index_to_token[input_seq[i, j].item()] for j in range(input_seq.size(1)) if
                       input_seq[i, j].item() != 0]
        output_chars = [' ' if output[i, j].item() > 0.5 else '' for j in range(output.size(1))]
        result.append(''.join(char + output_char for char, output_char in zip(input_chars, output_chars)))

    return ''.join(result)


if __name__ == "__main__":
    actual = construct_dataset(
        ["hi bye", "the world is huge", "in the beginning was the Word and the Word was with God"])
    expected = (["hibye", "theworldishuge", "inthebeginningwastheWordandtheWordwaswithGod"],
                [[0, 1, 0, 0, 1], [0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
                 [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1,
                  0, 0, 1, 0, 0, 0, 1, 0, 0, 1]])
    if expected != actual:
        print("Expected:", expected)
        print("Actual:", actual)
        raise ValueError("construct_dataset is incorrect")

    actual = create_char_to_index_map(["hi bye", "the world is huge", "in the beginning was the Word and the Word was "
                                                                      "with God"])

# METHODS TO PREPROCESS DATA INTO TENSORS
import torch
import string
import unicodedata
import io

# process strings from unicode into ascii
allowed_char = string.ascii_letters + ' .,;\'_'
def unicode_to_ascii(unicode_string:str) -> str:
    return ''.join(curr_char for curr_char in unicodedata.normalize('NFD', unicode_string) 
                   if unicodedata.category != 'Mn' and curr_char in allowed_char)

# turn letter into index for one hot tensor of len(allowed_char)
def letter_to_index(letter:str) -> int: 
    index = allowed_char.find(letter)
    return len(allowed_char)-1 if index == -1 else index

# turn string into "list" of one hot tensors (actually just a big tensor)
def string_to_tensor(line:str) -> torch.Tensor:
    line = unicode_to_ascii(line) # remove unwanted characters

    tensor = torch.zeros(len(line), 1, len(allowed_char))
    for index, letter in enumerate(line):
        tensor[index][0][letter_to_index(letter)] = 1
    return tensor
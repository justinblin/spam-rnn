import torch
import math

# turn output tensor into a human-readable label
# def label_from_output(output:torch.Tensor, output_labels:list[str], threshold) -> tuple[str, int]:
#     top_number, top_index = output.topk(1) # get list of 1 largest element and index
#     guess_index = top_index[0].item() # pull out index and convert from tensor into standard python
#     return output_labels[guess_index], guess_index
def label_from_output(output:torch.Tensor, output_labels:list[str], threshold:float = 0.5) -> tuple[str, int]:
    # print(math.e**output[0][1])
    if math.e**output[0][1] > threshold: guess_index = 1
    else: guess_index = 0
    return output_labels[guess_index], guess_index
import torch
from dataset import MyDataset
import preprocess

# SETUP DATASET
all_data = MyDataset('data/spam.csv')
train_set, test_set, extra_set = torch.utils.data.random_split(all_data, [.7, .2, .1], generator = torch.Generator(device = torch.device('cpu')).manual_seed(123))

print(extra_set[0])
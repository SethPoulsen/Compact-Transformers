import torch
from torch.utils.data import Dataset
import pandas as pd
from torchtext.vocab import GloVe

# https://coderzcolumn.com/tutorials/artificial-intelligence/how-to-use-glove-embeddings-with-pytorch
from torchtext.data import get_tokenizer
tokenizer = get_tokenizer("basic_english") ## We'll use tokenizer available from PyTorch
from torchtext.vocab import GloVe

max_words = 25
embed_len = 300
global_vectors = GloVe(name='840B', dim=embed_len)
def vectorize_batch(X):
    # Y, X = list(zip(*batch))
    X = [tokenizer(x) for x in X]
    X = [tokens+[""] * (max_words-len(tokens))  if len(tokens)<max_words else tokens[:max_words] for tokens in X]
    X_tensor = torch.zeros(len(X), max_words, embed_len)
    for i, tokens in enumerate(X):
        X_tensor[i] = global_vectors.get_vecs_by_tokens(tokens)
    return X_tensor #.reshape(len(batch), -1), torch.tensor(Y) - 1 ## Subtracted 1 from labels to bring in range [0,1,2,3] from [1,2,3,4]


class AgNewsCSVDataset(Dataset):
    def __init__(self):
        self.test_data = pd.read_csv('./data_manual/ag_news/test.csv')
        self.labels = torch.tensor(self.test_data['Class Index'].values)
        # self.descriptions = torch.tensor(self.test_data.Description.values)
        descriptions = self.test_data.Description.values
        # self.descriptions = torch.tensor([global_vectors.get_vecs_by_tokens(tokenizer(d), lower_case_backup=True) for d in descriptions])
        self.descriptions = vectorize_batch(descriptions)

    def __len__(self):
        return len(self.test_data)

    def __getitem__(self, idx):
        return self.descriptions[idx], self.labels[idx]
        # return self.test_data.Description[idx], self.test_data['Class Index'][1]


# from torch.utils.data import DataLoader
# from torchtext.data.functional import to_map_style_dataset

# max_words = 25
# embed_len = 300


# target_classes = ["World", "Sports", "Business", "Sci/Tech"]

# train_dataset, test_dataset  = torchtext.datasets.AG_NEWS()
# train_dataset, test_dataset = to_map_style_dataset(train_dataset), to_map_style_dataset(test_dataset)

# train_loader = DataLoader(train_dataset, batch_size=1024, collate_fn=vectorize_batch)
# test_loader  = DataLoader(test_dataset, batch_size=1024, collate_fn=vectorize_batch)
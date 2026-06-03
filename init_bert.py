import torch
from torch.utils.data import Dataset,DataLoader
from config import Config
import pandas as pd

config = Config()

def load_raw_data(file_path):
    data = []
    df = pd.read_csv(file_path)
    for _,line in df.iterrows():
        text = line['review']
        label = int(line['label'])
        data.append((text,label))
    return data

class TextDataSet(Dataset):
    def __init__(self,data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        x = self.data[item][0]
        y = self.data[item][-1]
        return x,y

def collate_fn(batch):
    texts = [i[0] for i in batch]
    labels = [i[-1] for i in batch]

    # 使用 HuggingFace tokenizer
    token_encode = config.tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors='pt'  # 直接返回 PyTorch tensor
    )
    
    input_ids = token_encode['input_ids']
    attention_mask = token_encode['attention_mask']
    labels = torch.tensor(labels, dtype=torch.long)

    return input_ids, attention_mask, labels

def build_dataloader():
    train_data = load_raw_data(config.train_path)
    test_data = load_raw_data(config.test_path)
    dev_data = load_raw_data(config.dev_path)

    train_data = TextDataSet(train_data)
    test_data = TextDataSet(test_data)
    dev_data = TextDataSet(dev_data)

    train_loader = DataLoader(train_data,batch_size=config.batch_size,shuffle=True,collate_fn=collate_fn)
    test_loader = DataLoader(test_data,batch_size=config.batch_size,shuffle=True,collate_fn=collate_fn)
    dev_loader = DataLoader(dev_data,batch_size=config.batch_size,shuffle=True,collate_fn=collate_fn)

    return train_loader,test_loader,dev_loader



if __name__ == '__main__':
    train_loader, test_loader , dev_loader= build_dataloader()
    for x,y,z in train_loader:
        print(f"input_ids shape: {x.shape}")
        print(f"attention_mask shape: {y.shape}")
        print(f"labels shape: {z.shape}")
        break

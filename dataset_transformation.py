import torch
import torchvision
import transformers
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device:", device)

dataset = load_dataset("MCIC")

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor()
])

class LoadData(Dataset):    
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]['image']
        label = self.dataset[idx]['label']
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

train_dataset = LoadData(dataset['train'], transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
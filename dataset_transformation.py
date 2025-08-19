import torch
import torchvision
import transformers
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device:", device)

dataset = load_dataset("archive\Veri-Kumesi_V1-100\Train")

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((384, 384)), # can also consider 512x512
    torchvision.transforms.ToTensor(),
    # torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class LoadData(Dataset):    
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        # mapping from folder names to numerical labels
        self.class_to_idx = {}
        self.idx_to_class = {}
        unique_labels = set()
        
        # extract folder names for class labels
        for item in dataset:
            if 'label' in item:
                unique_labels.add(item['label'])
        
        # Create mappings
        for idx, class_name in enumerate(sorted(unique_labels)):
            self.class_to_idx[class_name] = idx
            self.idx_to_class[idx] = class_name

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]['image']
        folder_name = self.dataset[idx]['label']  # folder name
        label = self.class_to_idx[folder_name]    # numerical label
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

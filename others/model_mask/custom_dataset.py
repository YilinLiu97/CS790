from torch.utils.data.dataset import Dataset
from torchvision import transforms
import torchvision.datasets as datasets
import torch


class CustomDataset(Dataset):
    def __init__(self, args, transform=None):
        self.data = datasets.ImageFolder(args.data_path, transform=transform)
        print(self.data[0][0].shape)
        self.x = torch.randn([len(self.data), 3, args.input_size, args.input_size])

    def __getitem__(self, index):
        # stuff
        ...
        y, _ = self.data[index]
        x = self.x[index]
        return x, y

    def __len__(self):
        return len(self.x)  # of how many data(images?) you have

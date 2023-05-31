from torch.utils.data import Dataset
from torchvision import transforms, datasets


class ImageDataset(Dataset):
    def __init__(self,data_path):
        self.data = datasets.ImageFolder(data_path)
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=45),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __getitem__(self, idx):
        x, y = self.data[idx]
        x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.data)

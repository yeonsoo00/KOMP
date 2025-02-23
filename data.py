import os
from PIL import Image
from torch.utils.data import Dataset


# Custom dataset class to handle binary images and masks
class CustomDataset(Dataset):
    def __init__(self, root, target_size, transform=None):
        self.root = root
        self.target_size = target_size
        self.transform = transform
        self.image_paths = os.listdir(os.path.join(self.root, 'Original'))  
        # Populate self.image_paths with paths to your images

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root, 'Original',self.image_paths[idx])
        image = Image.open(image_path).convert("L")
        mask_path = image_path.replace("Original", "Mask")
        mask = Image.open(mask_path.split(".jpg")[0] + '_template.jpg').convert("L")  # Convert to grayscale for binary masks

        # resize
        image = image.resize(self.target_size, Image.BILINEAR)
        mask = mask.resize(self.target_size, Image.BILINEAR)

        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask
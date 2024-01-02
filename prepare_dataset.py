import os
from natsort import natsorted
from torch.utils.data import Dataset
import PIL.Image as Image
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

## Create a custom Dataset class
class CelebADataset(Dataset):
  def __init__(self, root_dir, transform=None):
    #Args:
    #  root_dir (string): Directory with all the images
    #  transform (callable, optional): transform to be applied to each image sample
    # Read names of images in the root directory
    image_names = os.listdir(root_dir)

    self.root_dir = root_dir
    self.transform = transform
    self.image_names = natsorted(image_names)

  def __len__(self):
    return len(self.image_names)

  def __getitem__(self, idx):
    # Get the path to the image
    img_path = os.path.join(self.root_dir, self.image_names[idx])
    # Load image and convert it to RGB
    img = Image.open(img_path).convert('RGB')
    # Apply transformations to the image
    if self.transform:
      img = self.transform(img)

    return img

def prepare_dataset(root='data_faces/img_align_celeba', device='cpu', image_size=64, batch_size=128):
    img_list = os.listdir(root)
    print("Number of images:", len(img_list))

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)]
    )

    num_workers = 0 if device != 'cpu' else 2 # Number of workers for the dataloader
    pin_memory = True if device != 'cpu' else False # Whether to put fetched data tensors to pinned memory
    celeba_dataset = CelebADataset(root, transform)
    data_loader = DataLoader(celeba_dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=num_workers,
                             pin_memory=pin_memory)
    print("Image shape:", celeba_dataset[0].shape)

    return data_loader

if __name__ == '__main__':
    prepare_dataset()
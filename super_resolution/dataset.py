from torch.utils.data import DataLoader, Dataset


# create dataset
class LRDataset(Dataset):
    def __init__(self, root, img_list, transform=None):
        super(LRDataset, self).__init__()
        self.root = root
        self.img_list = img_list
        self.transform = transform
    
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.img_list[idx])
        
        img = Image.open(img_path)
        img = img.convert("RGB")
        img = np.array(img)
        
        if self.transform is not None:
            lr_aug = self.transform(image=img)
            lr_img = lr_aug['image']
        return lr_img, self.img_list[idx]

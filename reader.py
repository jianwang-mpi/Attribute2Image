from torchvision import datasets, transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os

class LFWDataset(Dataset):
    def __init__(self,csv_dir, image_dir, transformers = None):
        super(LFWDataset,self).__init__()
        self.attribute_frame = pd.read_csv(csv_dir)
        self.image_dir = image_dir
        self.transformers = transformers

    def __len__(self):
        return len(self.attribute_frame)
    def __getitem__(self, index):
        person_name = self.attribute_frame.ix[index, 0]
        person_name = str.replace(person_name, old=' ', new='_')
        image_name = person_name+'_'+'%04d'%int(self.attribute_frame.ix[index,1])+'.jpg'
        image_path = os.path.join(self.image_dir, person_name, image_name)
        with Image.open(image_path) as img:
            image = img.convert('RGB')
        attributes = self.attribute_frame.as_matrix()[index,2:].astype('float')
        if self.transformers is not None:
            image = self.transformers(image)
        return image, attributes

trans = transforms.Compose(transforms=[
    transforms.CenterCrop(size=64),

])

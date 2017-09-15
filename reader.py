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
        image_path = os.path.join(self.image_dir, image_name)
        with Image.open(image_path) as img:
            image = img.convert('RGB')
        attributes = self.attribute_frame.as_matrix()[index,2:].astype('float')
        if self.transformers is not None:
            image = self.transformers(image)
        return image, attributes

trans = transforms.Compose(transforms=[
    transforms.RandomHorizontalFlip(),
    transforms.Normalize(mean=[121.3,102.7,89.6], std=[71.1,64.5,63.3]),
    transforms.ToTensor()
])

train_dataset = LFWDataset(csv_dir='lfw_attr_train.csv',image_dir='lfw_cropped',transformers=trans)
test_dataset = LFWDataset(csv_dir='lfw_attr_test.csv', image_dir='lfw_cropped', transformers=trans)

train_loader = DataLoader(dataset = train_dataset, batch_size=32, shuffle=True, num_workers=4)
test_loader = DataLoader(dataset = test_dataset, batch_size=32, shuffle=True, num_workers=4)
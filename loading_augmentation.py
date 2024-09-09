import torch
import glob
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import dataset



# define the global variables
input_dim=(256,256) # set the input dimension of the images to 256x256
channel_dim=1 # 1 for greyscale, 3 for RGB (We are using greyscale)


# use the GPU if available to speed up the training (especially for google colab)
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA")
else:
    device = torch.device("cpu")
    print("Using CPU")



#We have to generate a dataset using de Dataset class. We got this code from the Exercise 5 solution. 
class CustomDataset(Dataset):
    def __init__(self, img_size, class_names, path=None, transformations=None, num_per_class: int = -1):
        self.img_size = img_size
        self.path = path
        self.num_per_class = num_per_class
        self.class_names = class_names
        self.transforms = transformations
        self.data = []
        self.labels = []

        if path:
            self.readImages()

        self.standard_transforms = transforms.Compose([
            transforms.ToTensor()
            ])

    def readImages(self):
        for id, class_name in self.class_names.items():
            print(f'Loading images from class: {id} : {class_name}')
            img_path = glob.glob(f'{self.path}{class_name}/*.jpg')
            if self.num_per_class > 0:
                img_path = img_path[:self.num_per_class]
            self.labels.extend([id] * len(img_path))
            for filename in img_path:
                img = Image.open(filename).convert('L')
                img = img.resize(self.img_size)
                self.data.append(img)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]

        if self.transforms:
            img = self.transforms(img)
        else:
            img = self.standard_transforms(img)

        label = torch.tensor(label, dtype=torch.long)

        return img, label
    



#After testing several types of data augmentation we found out the performance only got worse, so we used none.
train_transform = transforms.Compose([
    transforms.ToTensor(),
    #We tested image flipping, noise addition, gaussian bluring, image scaling, contrast enhancement...
])

train_path = "data/training/"
test_path = "data/testing/"
validation_path = "data/validation/"

#Getting the class names from the training set
class_names = [name[len(train_path):] for name in glob.glob(f'{train_path}*')]
class_names = dict(zip(range(len(class_names)), class_names))

#Generating the datasets and dataloaders for the model
train_dataset = CustomDataset(img_size=input_dim, path=train_path, class_names=class_names, transformations=train_transform) #We only applied transformations to the training set
test_dataset = CustomDataset(img_size=input_dim, path=test_path, class_names=class_names)
validation_dataset = CustomDataset(img_size=input_dim, path=validation_path, class_names=class_names)

train_dataloader = DataLoader(train_dataset, shuffle=True)
test_dataloader = DataLoader(test_dataset, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, shuffle=True)
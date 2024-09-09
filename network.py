#Imports again... Just in case
import torch
import torch.nn as nn
import torch.optim as optim
import glob
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import loading_augmentation
import dataset


# Define global variables
input_dim = (256, 256)  # Input dimension of the images (e.g., 256x256 pixels)
channel_dim = 1         # Number of input channels (1 for grayscale, 3 for RGB)

class ai_model(nn.Module):
    """
    A convolutional neural network (CNN) model for binary classification.

    This class defines a neural network designed for classifying grayscale chest x-ray images 
    into two categories: healthy and pneumatic. It consists of two convolutional layers followed 
    by a series of fully connected layers. The network includes dropout layers for regularization 
    and ReLU activations for non-linearity.

    Pneumonia causes inflammation of the lungs' air sacs, which fill up with fluid. 
    This fluid is denser than the air in the lungs and absorbs more x-rays, resulting in white spots 
    on the x-ray image. The model aims to detect these opaque spots. Convolutional layers with 
    different kernel sizes allow the model to capture patterns at various scales. After the convolutional 
    layers, the data is flattened and passed through three hidden layers. Dropout layers are used 
    between hidden layers to prevent overfitting. The chosen dropout rate is 0.3, based on experimentation 
    and results. The model's input dimensions are 256x256 to balance accuracy and training efficiency.

    Architecture:
    - Convolutional Layer (conv1): Applies 32 convolutional filters of size 5x5 to the input.
    - Convolutional Layer (conv2): Applies 64 convolutional filters of size 3x3.
    - Max Pooling Layer (maxpool1): Reduces spatial dimensions by pooling with a 2x2 kernel.
    - Fully Connected Layer (fc1): Transforms the flattened output to 512 units.
    - Fully Connected Layer (fc2): Further transforms the data to 128 units.
    - Fully Connected Layer (fc3): Reduces the data to 64 units.
    - Fully Connected Layer (fc4): Outputs 2 units representing the final class scores.
    - Dropout Layers: Applied after each fully connected layer to prevent overfitting.
    - ReLU Activations: Introduce non-linearity after each convolutional and fully connected layer.
    - Softmax Activation: Outputs probability distribution over the two classes.

    Attributes:
        conv1 (nn.Conv2d): The first convolutional layer.
        conv2 (nn.Conv2d): The second convolutional layer.
        maxpool1 (nn.MaxPool2d): The max pooling layer.
        fc1 (nn.Linear): The first fully connected layer.
        fc2 (nn.Linear): The second fully connected layer.
        fc3 (nn.Linear): The third fully connected layer.
        fc4 (nn.Linear): The final fully connected layer.
        dropout (nn.Dropout): Dropout layer for regularization.
        relu (nn.ReLU): ReLU activation function.
        softmax (nn.Softmax): Softmax activation function for output probabilities.
    """

    def __init__(self):
        """
        Initializes the network layers and components.
        
        Sets up the convolutional layers, max pooling, fully connected layers, dropout layers,
        and activation functions according to the architecture defined for the network.
        """
        super(ai_model, self).__init__()
        
        # Two convolutional layers to detect patterns in the images
        self.conv1 = nn.Conv2d(in_channels=channel_dim, out_channels=32, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        # Correct the input size for fc1 based on the calculations
        self.fc1 = nn.Linear(64 * 125 * 125, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 2)
        # 0.3 dropout to reduce overfitting
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        Defines the forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channel_dim, height, width), where
                                batch_size is the number of images in a batch, channel_dim is the number
                                of input channels (e.g., grayscale), and height and width are the dimensions 
                                of the images.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 2) where each value represents the 
                            probability of the input belonging to one of the two classes.
        """
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.maxpool1(x)
        x = x.view(x.size(0), -1)

        # Fully Connected Layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return self.softmax(x)
    

model = ai_model()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

model.to(device)




from tqdm.notebook import tqdm
pbar = None
def train(model, num_epochs: int = 3):
    for epoch in range(num_epochs):
        correct = 0
        total = 0
        model.train()

        pbar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch", leave=True)

        for data, targets in train_dataloader:
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(data)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            current_accuracy = correct / total * 100

            pbar.update(1)
            pbar.set_postfix(accuracy=f"{current_accuracy:.2f}%")

        pbar.close()
        tqdm.write(f"Epoch {epoch + 1}/{num_epochs}, Training accuracy: {current_accuracy:.2f}%")


        model.eval()
        correct_validation = 0
        total_validation = 0
        with torch.no_grad():
            for data, targets in validation_dataloader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total_validation += targets.size(0)
                correct_validation += (predicted == targets).sum().item()

        validation_accuracy = 100 * correct_validation / total_validation
        print(f'Validation accuracy: {validation_accuracy}%')
        torch.save(model.state_dict(), f'drive/MyDrive/SDU_Data/models/model_weights_3_{epoch}.pth')



def test(model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, targets in test_dataloader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")


train(model=model, num_epochs=15)
test(model=model)
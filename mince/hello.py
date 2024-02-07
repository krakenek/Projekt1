import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import models

# Definice transformací
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(360),      # Přidání augmentace dat
    transforms.ToTensor(),
])

# Cesta k datasetu
data_path = 'C:\\Users\\krake\\mince\\Dataset'

# Vytvoření ImageFolder datasetu
dataset = ImageFolder(root=data_path, transform=transform)

# Rozdělení datasetu na trénovací, validační a testovací část
train_size = int(0.60 * len(dataset))
val_size = int(0.30 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

# DataLoader pro trénovací, validační a testovací množinu
batch_size = 256  # Zvolte vhodnou velikost dávky
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Definice modelu
class SimpleModel(nn.Module):
    def __init__(self, num_classes):
        super(SimpleModel, self).__init__()
        self.features = models.resnet18(pretrained=True)
        in_features = self.features.fc.in_features
        self.features.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.features(x)

# Inicializace modelu
num_classes = len(dataset.classes)
model = SimpleModel(num_classes)

# Definice loss funkce a optimalizačního algoritmu
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Trénování modelu
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Validace modelu
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in val_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = correct / total
        print(f'Epoch {epoch+1}/{num_epochs}, Validation Accuracy: {accuracy:.4f}')

# Testování modelu
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    test_accuracy = correct / total
    print(f'Test Accuracy: {test_accuracy:.4f}')
    
    
    
torch.save(model.state_dict(), 'C:\\Users\\krake\\mince\\model60_11.pth')

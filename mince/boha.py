import cv2 
import os
import shutil
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models

def extract_coins(img):
    
    # Převedení obrázku na odstíny šedi
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Binarizace obrázku
    _ , binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binary = cv2.dilate(binary, kernel, iterations=1)

    # Nalezne kontury na binárním obrázku
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Vytvoření slovníku pro ukládání výsledků
    coins = {}

    # Minimální plocha pro filtrování malých kousků mincí
    min_contour_area = 20000

    # Procházení kontur
    for i, contour in enumerate(contours):
        # Vypočet plochy kontury
        area = cv2.contourArea(contour)

        # Filtrování malých kontur
        if area > min_contour_area:
            # Vytvoření ohraničujícího obdélníku kolem kontury
            x, y, w, h = cv2.boundingRect(contour)

            # Oříznutí původního obrázku podle ohraničujícího obdélníku
            result = img[y:y+h, x:x+w]

            # Uložení výsledného obrázku mince
            coins[f'coin{i+1}.jpg'] = result

    return coins

def extract_coins_plus(image_path):
    

    img = cv2.imread(image_path)

    if img is None:
        print(f"Error: Unable to load image at {image_path}")
    else:
        # Specify the directory where you want to save the coins
        output_directory = 'C:\\Users\\krake\\mince\\output_coins'

        # Remove the existing output directory if it exists
        if os.path.exists(output_directory):
            shutil.rmtree(output_directory)

        # Create the output directory
        os.makedirs(output_directory)

        # Call the extract_coins function
        coins_dict = extract_coins(img)

        # Iterate through the extracted coins and save them as JPG files
        for coin_name, coin_img in coins_dict.items():
            coin_path = os.path.join(output_directory, coin_name)
            cv2.imwrite(coin_path, coin_img)

        # Print a message indicating the successful extraction and saving
        print(f'Coins extracted and saved in the directory: {output_directory}')

image = 'C:\\Users\\krake\\mince\\testtest2.jpg'
extract_coins_plus(image)

class SimpleModel(nn.Module):
    def __init__(self, num_classes):
        super(SimpleModel, self).__init__()
        self.features = models.resnet18(pretrained=True)
        in_features = self.features.fc.in_features
        self.features.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.features(x)


# Define the transformation for the input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
data_path = r'C:\\Users\\krake\\mince\\Dataset'

final = 0

dataset = ImageFolder(root=data_path, transform=transform)
# Load the trained model
num_classes = len(dataset.classes)
model = SimpleModel(num_classes)
model.load_state_dict(torch.load('C:\\Users\\krake\\mince\\model60_11_56.pth'))
model.eval()

# Specify the path to the folder containing cutout coins
folder_path = 'C:\\Users\\krake\\mince\\output_coins'

# Iterate through the cutout coins in the folder
for coin_file in os.listdir(folder_path):
    # Construct the full path to the cutout coin image
    coin_path = os.path.join(folder_path, coin_file)

    # Load and preprocess the input coin image
    coin_image = Image.open(coin_path)
    coin_input_tensor = transform(coin_image).unsqueeze(0)

    # Perform inference
    with torch.no_grad():
        output = model(coin_input_tensor)

    # Get the predicted class
    _, predicted_class = torch.max(output, 1)

    # Display the results
    class_names = dataset.classes
    predicted_class_name = class_names[predicted_class.item()]

    final = final + int(predicted_class_name)

    plt.imshow(coin_image)
    plt.title(f'Predicted Class: {predicted_class_name}')
    plt.show()

THEimage = Image.open(image)

plt.imshow(THEimage)
plt.title(f'Final Count: {final}')
plt.show()
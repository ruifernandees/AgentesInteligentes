# predict.py
# Exemplo de predição usando o modelo treinado em model.py

import os
import torch
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50
from PIL import Image
import numpy as np
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Color to class mapping
COLOR2CLASS = {
    (60, 16, 152): 0,      # Building: #3C1098
    (132, 41, 246): 1,     # Land: #8429F6
    (110, 193, 228): 2,    # Road: #6EC1E4
    (254, 221, 58): 3,     # Vegetation: #FEDD3A
    (226, 169, 41): 4,     # Water: #E2A929
    (155, 155, 155): 5     # Unlabeled: #9B9B9B
}
NUM_CLASSES = 6

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

def predict_image(model, image_path, device):
    model.eval()
    image = Image.open(image_path).convert('RGB')
    image = image.resize((256, 256))
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)['out']
        pred_mask = torch.argmax(output.squeeze(), dim=0).cpu().numpy()
    return pred_mask

if __name__ == "__main__":
    images_dir = 'dataset/images'
    model_path = 'satellite_segmentation.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = deeplabv3_resnet50(weights=None, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    # Use a imagem de teste (troque o nome se quiser)
    test_image_path = os.path.join(images_dir, os.listdir(images_dir)[0])
    pred_mask = predict_image(model, test_image_path, device)
    # Salvar máscara predita como imagem
    try:
        from matplotlib import pyplot as plt
        plt.imsave('predicted_mask.png', pred_mask, cmap='tab20')
        print('Predição realizada e salva em predicted_mask.png')
    except ImportError:
        print('Predição realizada. Instale matplotlib para salvar a máscara como imagem.')

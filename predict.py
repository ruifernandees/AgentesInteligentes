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
    import sys
    images_dir = 'dataset/images'
    model_path = 'satellite_segmentation.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = deeplabv3_resnet50(weights=None, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)

    # CLI: escolha da imagem
    if len(sys.argv) > 1:
        image_name = sys.argv[1]
        test_image_path = os.path.join(images_dir, image_name)
        if not os.path.exists(test_image_path):
            print(f"Arquivo '{image_name}' não encontrado em {images_dir}.")
            sys.exit(1)
    else:
        test_image_path = os.path.join(images_dir, os.listdir(images_dir)[0])
        print(f"Nenhum nome de imagem fornecido. Usando: {os.path.basename(test_image_path)}")

    pred_mask = predict_image(model, test_image_path, device)
    # Salvar máscara predita como imagem
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        # Cores originais das classes
        CLASS_COLORS = [
            (60/255, 16/255, 152/255),      # Building
            (132/255, 41/255, 246/255),     # Land
            (110/255, 193/255, 228/255),    # Road
            (254/255, 221/255, 58/255),     # Vegetation
            (226/255, 169/255, 41/255),     # Water
            (155/255, 155/255, 155/255)     # Unlabeled
        ]
        CLASS_NAMES = ['Building', 'Land', 'Road', 'Vegetation', 'Water', 'Unlabeled']

        # Converter máscara para RGB
        rgb_mask = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.float32)
        for idx, color in enumerate(CLASS_COLORS):
            rgb_mask[pred_mask == idx] = color

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(rgb_mask)
        ax.axis('off')

        # Criar legenda
        patches = [mpatches.Patch(color=color, label=label) for color, label in zip(CLASS_COLORS, CLASS_NAMES)]
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.tight_layout()
        plt.savefig('predicted_mask.png', bbox_inches='tight')
        print('Predição realizada e salva em predicted_mask.png com legenda.')
    except ImportError:
        print('Predição realizada. Instale matplotlib para salvar a máscara como imagem com legenda.')

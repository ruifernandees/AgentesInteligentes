# api.py
# API REST para predição de segmentação de imagens de satélite
# Endpoint: POST /predict (upload de imagem) → retorna máscara predita PNG

from flask import Flask, request, send_file, jsonify
import os
import torch
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50
from PIL import Image
import numpy as np
import io
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

app = Flask(__name__)

# Configuração do modelo
NUM_CLASSES = 6
MODEL_PATH = 'satellite_segmentation.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Carrega o modelo uma vez ao iniciar a API
model = deeplabv3_resnet50(weights=None, num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# Transformações
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Cores das classes (RGB normalizadas 0-1)
CLASS_COLORS = [
    (60/255, 16/255, 152/255),      # Building
    (132/255, 41/255, 246/255),     # Land
    (110/255, 193/255, 228/255),    # Road
    (254/255, 221/255, 58/255),     # Vegetation
    (226/255, 169/255, 41/255),     # Water
    (155/255, 155/255, 155/255)     # Unlabeled
]

def predict_image(image_bytes):
    """
    Recebe bytes de imagem, faz predição e retorna máscara RGB como array numpy.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = image.resize((256, 256))
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)['out']
        pred_mask = torch.argmax(output.squeeze(), dim=0).cpu().numpy()
    
    # Converte máscara de classes para RGB
    rgb_mask = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
    for idx, color in enumerate(CLASS_COLORS):
        rgb_mask[pred_mask == idx] = [int(c * 255) for c in color]
    
    return rgb_mask

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint POST /predict
    Espera um arquivo de imagem no campo 'image' (multipart/form-data).
    Retorna a máscara predita como PNG.
    """
    if 'image' not in request.files:
        return jsonify({'error': 'Nenhum arquivo de imagem enviado. Use o campo "image".'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Nome de arquivo vazio.'}), 400
    
    try:
        # Lê os bytes da imagem
        image_bytes = file.read()
        
        # Faz a predição
        rgb_mask = predict_image(image_bytes)
        
        # Converte o array numpy para imagem PIL e salva em buffer
        mask_image = Image.fromarray(rgb_mask, mode='RGB')
        buffer = io.BytesIO()
        mask_image.save(buffer, format='PNG')
        buffer.seek(0)
        
        return send_file(buffer, mimetype='image/png', as_attachment=False, download_name='predicted_mask.png')
    
    except Exception as e:
        return jsonify({'error': f'Erro ao processar imagem: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health():
    """
    Endpoint de health check.
    """
    return jsonify({'status': 'ok', 'model': 'DeepLabV3-ResNet50', 'classes': NUM_CLASSES, 'device': str(device)})

if __name__ == '__main__':
    # Verifica se o modelo existe
    if not os.path.exists(MODEL_PATH):
        print(f"ERRO: Modelo '{MODEL_PATH}' não encontrado. Execute model.py para treinar o modelo antes de iniciar a API.")
        exit(1)
    
    print(f"API iniciada. Modelo carregado em {device}.")
    print("Endpoints disponíveis:")
    print("  POST /predict  - Upload de imagem para predição")
    print("  GET  /health   - Health check")
    app.run(host='0.0.0.0', port=10000, debug=True)

# Segmentação de Imagens de Satélite (DeepLabV3)

Este projeto treina um modelo de segmentação semântica (DeepLabV3-ResNet50) para classificar cada pixel de imagens de satélite em 6 classes: Building, Land, Road, Vegetation, Water e Unlabeled. Inclui pipeline de treino e script de predição com saída visual com legenda.

## Estrutura do projeto

```
```
AgentesInteligentes/
├── model.py              # Treinamento e função de predição básica
├── predict.py            # Script de predição com CLI e legenda
├── api.py                # API REST para predição via upload de imagem
├── dataset/
│   ├── images/           # Imagens RGB (.jpg)
│   └── masks/            # Máscaras anotadas (.png ou .jpg) com cores padronizadas
└── README.md             # Este guia
```
```

### Classes e cores para treino do modelo com máscaras (mapeamento RGB → classe)
- Building:    #3C1098  → (60, 16, 152)  → classe 0
- Land:        #8429F6  → (132, 41, 246) → classe 1
- Road:        #6EC1E4  → (110, 193, 228)→ classe 2
- Vegetation:  #FEDD3A  → (254, 221, 58) → classe 3
- Water:       #E2A929  → (226, 169, 41) → classe 4
- Unlabeled:   #9B9B9B  → (155, 155, 155)→ classe 5

As máscaras devem usar exatamente essas cores para que o conversor RGB→classe funcione corretamente.

## Requisitos

- macOS com Python 3.9+ (recomendado)
- GPU CUDA opcional (acelera o treino)
- Pacotes Python:
  - torch, torchvision
  - Pillow, numpy
  - matplotlib (opcional, para salvar imagem com legenda no predict)
  - Flask (para a API REST)

Instalação dos pacotes (exemplos):

```zsh
# Opcional: criar e ativar um ambiente virtual
python3 -m venv .venv
source .venv/bin/activate

# Instalar dependências
pip install -r requirements.txt
```

Observação: o código define um bypass SSL para evitar erros de certificado em ambientes locais. Em produção/remoto, remova-o.

## Git LFS (Large File Storage)

Este projeto utiliza Git LFS para versionar arquivos grandes como o modelo treinado (`.pth`), imagens do dataset e outros binários.

### Instalar Git LFS

```zsh
# macOS (via Homebrew)
brew install git-lfs

# Inicializar Git LFS no repositório
git lfs install
```

### Configurar rastreamento de arquivos

O arquivo `.gitattributes` já está configurado para rastrear:
- Modelos treinados: `*.pth`
- Imagens: `*.jpg`, `*.png`
- Outros arquivos grandes conforme necessário

### Clonar o repositório

Ao clonar o repositório, o Git LFS baixará automaticamente os arquivos grandes:

```zsh
git clone https://github.com/ruifernandees/AgentesInteligentes.git
cd AgentesInteligentes
```

Se você já clonou o repositório antes de instalar o Git LFS, execute:

```zsh
git lfs pull
```

## Como preparar os dados

Coloque seus arquivos nas pastas:

```
dataset/
├── testing/
│   ├── image0001.jpg
│   ├── image0002.jpg
│   └── ...
├── training/
│   ├── image0001.jpg
│   ├── image0002.jpg
│   └── ...
└── masks/
    ├── image0001.png  # mesma ordem/alinhamento com images/
    ├── image0002.png
    └── ...
```

- As listas de arquivos são ordenadas por nome; garanta correspondência 1:1 entre `images/` e `masks/`.
- As imagens e máscaras são redimensionadas para 256×256 durante o pipeline (pode ajustar em `model.py`).

## Treinar o modelo

O script `model.py` treina o DeepLabV3-ResNet50 (sem pesos pré-treinados) com `NUM_CLASSES=6` e salva os pesos em `satellite_segmentation.pth`.

```zsh
# Executar treino (padrão: 10 épocas, batch_size=4)
python3 model.py
```

Parâmetros relevantes em `model.py`:
- Tamanho da entrada: `transforms.Resize((256, 256))`
- Lote: `batch_size=4`
- Épocas: `train(model, train_loader, epochs=10)`
- Otimizador: Adam com learning rate `1e-4`
- Loss: `nn.CrossEntropyLoss()` (por pixel, 6 classes)

Saída do treino:
- Checkpoint do modelo: `satellite_segmentation.pth`

## Fazer predições (CLI)

Use `predict.py` para gerar a máscara predita e salvar uma imagem com legenda:

```zsh
# Predizer para uma imagem específica da pasta dataset/images
python3 predict.py image0050.jpg

# Sem argumento → usa o primeiro arquivo encontrado em dataset/images
python3 predict.py
```

Saídas:
- `predicted_mask.png` — visualização colorida da máscara predita com legenda das classes.

Como funciona:
- O script carrega `satellite_segmentation.pth`, redimensiona a entrada para 256×256, roda a inferência e converte o mapa de classes (0–5) em cores RGB originais, adicionando uma legenda com nomes.

## Usar a API REST

A API REST (`api.py`) permite fazer predições via HTTP. Ideal para integração com outras aplicações ou front-ends.

### Iniciar a API

```zsh
python3 api.py
```

A API roda em `http://localhost:3338` com os seguintes endpoints:

- **POST /predict**: Recebe uma imagem via upload e retorna a máscara predita em formato PNG.
- **GET /health**: Health check do serviço.

### Fazer uma predição via API

Usando `curl`:

```zsh
curl -X POST -F "image=@dataset/images/image0050.jpg" http://localhost:3338/predict --output predicted_mask_api.png
```

Usando Python (requests):

```python
import requests

url = 'http://localhost:3338/predict'
files = {'image': open('dataset/images/image0050.jpg', 'rb')}
response = requests.post(url, files=files)

if response.status_code == 200:
    with open('predicted_mask_api.png', 'wb') as f:
        f.write(response.content)
    print('Máscara salva em predicted_mask_api.png')
else:
    print('Erro:', response.json())
```

A API retorna diretamente a imagem PNG com as cores das classes (sem legenda sobreposta, apenas a máscara RGB).

## Arquivos principais

- `model.py`
  - Define o dataset customizado (converte máscara RGB → classes via mapeamento).
  - Constrói o modelo DeepLabV3-ResNet50 (`num_classes=6`).
  - Loop de treino (Adam + CrossEntropyLoss).
  - Função `predict_image` (inferência básica por arquivo).
- `predict.py`
  - Carrega o checkpoint `satellite_segmentation.pth`.
  - CLI para escolher a imagem: `python3 predict.py <arquivo.jpg>`.
  - Gera `predicted_mask.png` com cores originais e legenda.
- `api.py`
  - API REST com Flask.
  - Endpoint POST `/predict` para upload de imagem e retorno da máscara predita em PNG.
  - Endpoint GET `/health` para verificar status do serviço.
  - Carrega o modelo uma vez ao iniciar, mantendo-o em memória para inferências rápidas.


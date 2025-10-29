import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import numpy as np

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50
from PIL import Image

COLOR2CLASS = {
	(60, 16, 152): 0,      # Building: #3C1098
	(132, 41, 246): 1,     # Land: #8429F6
	(110, 193, 228): 2,    # Road: #6EC1E4
	(254, 221, 58): 3,     # Vegetation: #FEDD3A
	(226, 169, 41): 4,     # Water: #E2A929
	(155, 155, 155): 5     # Unlabeled: #9B9B9B
}

NUM_CLASSES = 6

class SatelliteDataset(Dataset):
	def __init__(self, images_dir, masks_dir, transform=None):
		self.images_dir = images_dir
		self.masks_dir = masks_dir
		self.transform = transform
		self.images = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg')])
		self.masks = sorted([f for f in os.listdir(masks_dir) if f.endswith('.png') or f.endswith('.jpg')])

	def __len__(self):
		return len(self.images)

	def __getitem__(self, idx):
		img_path = os.path.join(self.images_dir, self.images[idx])
		mask_path = os.path.join(self.masks_dir, self.masks[idx])
		image = Image.open(img_path).convert('RGB')
		mask = Image.open(mask_path).convert('RGB')
		# Redimensiona a máscara para 256x256
		mask = mask.resize((256, 256), resample=Image.NEAREST)
		mask = self.mask_to_class(mask)
		if self.transform:
			image = self.transform(image)
		return image, mask

	def mask_to_class(self, mask_img):
		mask_np = torch.from_numpy(np.array(mask_img)).long()
		class_mask = torch.zeros(mask_np.shape[:2], dtype=torch.long)
		for rgb, cls in COLOR2CLASS.items():
			matches = (mask_np == torch.tensor(rgb)).all(dim=-1)
			class_mask[matches] = cls
		return class_mask

transform = transforms.Compose([
	transforms.Resize((256, 256)),
	transforms.ToTensor(),
])

images_dir = 'dataset/training'
masks_dir = 'dataset/masks'
train_dataset = SatelliteDataset(images_dir, masks_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

model = deeplabv3_resnet50(weights=None, num_classes=NUM_CLASSES)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

def train(model, loader, epochs=5):
	model.train()
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
	criterion = nn.CrossEntropyLoss()
	for epoch in range(epochs):
		for images, masks in loader:
			images = images.to(device)
			masks = masks.to(device)
			optimizer.zero_grad()
			outputs = model(images)['out']
			loss = criterion(outputs, masks)
			loss.backward()
			optimizer.step()
		print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

if __name__ == "__main__":
	train(model, train_loader, epochs=10)
	torch.save(model.state_dict(), 'satellite_segmentation.pth')

def predict_image(model, image_path):
	model.eval()
	image = Image.open(image_path).convert('RGB')
	image = image.resize((256, 256))
	input_tensor = transform(image).unsqueeze(0).to(device)
	with torch.no_grad():
		output = model(input_tensor)['out']
		pred_mask = torch.argmax(output.squeeze(), dim=0).cpu().numpy()
	return pred_mask

if __name__ == "__main__":
	model.load_state_dict(torch.load('satellite_segmentation.pth', map_location=device))
	test_image_path = os.path.join(images_dir, train_dataset.images[0])  # Use a primeira imagem do dataset
	pred_mask = predict_image(model, test_image_path)
	# Salvar máscara predita como imagem
	from matplotlib import pyplot as plt
	plt.imsave('predicted_mask.png', pred_mask, cmap='tab20')
	print('Predição realizada e salva em predicted_mask.png')

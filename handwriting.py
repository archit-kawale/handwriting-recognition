import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

data_location = 'resources/words'
words_txt_location = 'resources/words.txt'

def get_paths_and_gts(partition_split_file):
    paths_and_gts = []
    with open(partition_split_file) as f:
        for line in f:
            if not line or line.startswith('#'):
                continue
            line_split = line.strip().split(' ')
            directory_split = line_split[0].split('-')
            image_location = f'{data_location}/{directory_split[0]}/{directory_split[0]}-{directory_split[1]}/{line_split[0]}.png'
            gt_text = ' '.join(line_split[8:])
            paths_and_gts.append([image_location, gt_text])
    return paths_and_gts

def add_padding(img, old_w, old_h, new_w, new_h):
    h1, h2 = int((new_h - old_h) / 2), int((new_h - old_h) / 2) + old_h
    w1, w2 = int((new_w - old_w) / 2), int((new_w - old_w) / 2) + old_w
    img_pad = np.ones([new_h, new_w, 3]) * 255
    img_pad[h1:h2, w1:w2, :] = img
    return img_pad

def fix_size(img, target_w, target_h):
    h, w = img.shape[:2]
    if w < target_w and h < target_h:
        img = add_padding(img, w, h, target_w, target_h)
    elif w >= target_w and h < target_h:
        new_w = target_w
        new_h = int(h * new_w / w)
        new_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        img = add_padding(new_img, new_w, new_h, target_w, target_h)
    elif w < target_w and h >= target_h:
        new_h = target_h
        new_w = int(w * new_h / h)
        new_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        img = add_padding(new_img, new_w, new_h, target_w, target_h)
    else:
        ratio = max(w / target_w, h / target_h)
        new_w = max(min(target_w, int(w / ratio)), 1)
        new_h = max(min(target_h, int(h / ratio)), 1)
        new_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        img = add_padding(new_img, new_w, new_h, target_w, target_h)
    return img

def preprocess(path, img_w, img_h):
    img = cv2.imread(path)
    img = fix_size(img, img_w, img_h)
    img = np.clip(img, 0, 255)
    img = np.uint8(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32)
    img /= 255
    return img

letters = [' ', '!', '"', '#', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/',
           '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?',
           'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
           'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
           'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
           'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
num_classes = len(letters) + 1
print(num_classes)

def text_to_labels(text):
    return [letters.index(x) for x in text]

def labels_to_text(labels):
    return ''.join([letters[int(x)] for x in labels])

train_files = get_paths_and_gts('resources/train_files.txt')
valid_files = get_paths_and_gts('resources/valid_files.txt')
test_files = get_paths_and_gts('resources/test_files.txt')
print(len(train_files), len(valid_files), len(test_files))

class HandwritingDataset(Dataset):
    def __init__(self, data, img_w, img_h, max_text_len):
        self.img_h = img_h
        self.img_w = img_w
        self.max_text_len = max_text_len
        self.samples = data
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_filepath, text = self.samples[idx]
        img = preprocess(img_filepath, self.img_w, self.img_h)
        img = self.transform(img).squeeze(0)
        
        label = text_to_labels(text)
        label_length = len(label)
        
        # Pad the label if it's shorter than max_text_len
        if label_length < self.max_text_len:
            label = label + [0] * (self.max_text_len - label_length)
        
        return img, torch.tensor(label), label_length

# Hyperparameters
batch_size = 64
input_length = 30
max_text_len = 16
img_w = 128
img_h = 64

# Create datasets and dataloaders
train_dataset = HandwritingDataset(train_files, img_w, img_h, max_text_len)
valid_dataset = HandwritingDataset(valid_files, img_w, img_h, max_text_len)
test_dataset = HandwritingDataset(test_files, img_w, img_h, max_text_len)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the model
class HandwritingCNN(torch.nn.Module):
    def __init__(self, num_classes):
        super(HandwritingCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = torch.nn.Linear(128 * 8 * 16, 512)
        self.fc2 = torch.nn.Linear(512, num_classes)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.max_pool2d(x, 2)
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.max_pool2d(x, 2)
        x = torch.nn.functional.relu(self.conv3(x))
        x = torch.nn.functional.max_pool2d(x, 2)
        x = x.view(-1, 128 * 8 * 16)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Instantiate the model and move it to GPU
model = HandwritingCNN(num_classes)
model = model.to(device)

# Define loss function and optimizer
criterion = torch.nn.CTCLoss(blank=num_classes-1, reduction='mean')
optimizer = torch.optim.Adam(model.parameters())

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (images, labels, label_lengths) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        input_lengths = torch.full(size=(outputs.size(0),), fill_value=outputs.size(1), dtype=torch.long)
        loss = criterion(outputs.transpose(0, 1), labels, input_lengths, label_lengths)
        
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')

    # Validation loop
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, labels, label_lengths in valid_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            input_lengths = torch.full(size=(outputs.size(0),), fill_value=outputs.size(1), dtype=torch.long)
            loss = criterion(outputs.transpose(0, 1), labels, input_lengths, label_lengths)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(valid_loader)
    print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_loss:.4f}')

print("Training completed!")
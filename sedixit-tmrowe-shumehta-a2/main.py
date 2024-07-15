# Authors: Tyler Rowe (tmrowe), Sean Dixit (sedixit)
# (based on skeleton code for CSCI-B 657, Feb 2024)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
from dataset_class import PatchShuffled_CIFAR10
from matplotlib import pyplot as plt
import argparse
import csv

# Packages added 
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# Define the model architecture for CIFAR10
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 64 * 8 * 8)
        x = self.fc_layers(x)
        return x
    
    def get_embedding(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 64 * 8 * 8)
        return x
    
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.max_pool = nn.MaxPool2d(2, 2)
        self.avg_pool = nn.AvgPool2d(2, 2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = self.max_pool(F.relu(self.conv1(x)))
        x = self.avg_pool(F.relu(self.conv2(x)))
        x = self.max_pool(F.relu(self.conv3(x)))
        x = self.global_pool(x)
        x = x.view(-1, 128)
        x = self.fc(x)
        return x
    
class ShallowCNN(nn.Module):
    def __init__(self):
        super(ShallowCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 32 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class CNNDropout(nn.Module):
    def __init__(self):
        super(CNNDropout, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(32 * 8 * 8, 120)
        self.fc2 = nn.Linear(120, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x.view(-1, 32 * 8 * 8))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class CNNBatchNorm(nn.Module):
    def __init__(self):
        super(CNNBatchNorm, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(32 * 8 * 8, 120)
        self.fc2 = nn.Linear(120, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 32 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class SkipConnectionCNN(nn.Module):
    def __init__(self):
        super(SkipConnectionCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.fc = nn.Linear(32 * 16 * 16, 10)
        self.match_dimensions = nn.Conv2d(3, 16, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x))
        identity = self.match_dimensions(identity)
        out = F.relu(self.conv2(out) + identity)
        out = F.relu(self.conv3(out))
        out = out.view(-1, 32 * 16 * 16)
        out = self.fc(out)
        return out

class UNetLike(nn.Module):
    def __init__(self):
        super(UNetLike, self).__init__()
        self.down_conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.down_pool = nn.MaxPool2d(2)
        self.down_conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        
        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_conv1 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        
        self.conv_final = nn.Conv2d(16, 10, kernel_size=1)

    def forward(self, x):
        # Downsample 
        x = F.relu(self.down_conv1(x))
        x = self.down_pool(x)
        x = F.relu(self.down_conv2(x))
        # Upsample 
        x = self.up_sample(x)
        x = F.relu(self.up_conv1(x))
        # Classification
        x = self.conv_final(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))  # Global average pool
        x = x.view(-1, 10)  # Flatten 
        return x

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 64 * 8 * 8)
        x = self.fc_layers(x)
        return x

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()

        # STORING IMAGE_SIZE, PATCH_SIZE AND DIM AS ATTR OF CLASS FOR FORWARD METHOD
        self.image_size = image_size
        self.patch_size = patch_size
        self.dim = dim

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        # RANDOMIZE POSITIONAL EMBEDDING EVERY FORWARD PASS
        image_height, image_width = pair(self.image_size)
        patch_height, patch_width = pair(self.patch_size)
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, self.dim))

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
    
    
# Define the model architecture for D-shuffletruffle
class Net_D_shuffletruffle(nn.Module):
    def __init__(self, image_size=32, patch_size=16, num_classes=10, dim=124, depth=10, heads=1, mlp_dim=32, pool = 'cls', channels = 3, dim_head = 32, dropout = 0, emb_dropout = 0):
        super().__init__()

        # STORING IMAGE_SIZE, PATCH_SIZE AND DIM AS ATTR OF CLASS FOR FORWARD METHOD
        self.image_size = image_size
        self.patch_size = patch_size
        self.dim = dim

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        # RANDOMIZE POSITIONAL EMBEDDING EVERY FORWARD PASS
        image_height, image_width = pair(self.image_size)
        patch_height, patch_width = pair(self.patch_size)
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, self.dim))

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
    
    def get_embedding(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        return x

# Define the model architecture for N-shuffletruffle
class Net_N_shuffletruffle(nn.Module):
    def __init__(self, image_size=32, patch_size=8, num_classes=10, dim=124, depth=10, heads=1, mlp_dim=32, pool = 'cls', channels = 3, dim_head = 32, dropout = 0, emb_dropout = 0):
        super().__init__()

        # STORING IMAGE_SIZE, PATCH_SIZE AND DIM AS ATTR OF CLASS FOR FORWARD METHOD
        self.image_size = image_size
        self.patch_size = patch_size
        self.dim = dim

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        # RANDOMIZE POSITIONAL EMBEDDING EVERY FORWARD PASS
        image_height, image_width = pair(self.image_size)
        patch_height, patch_width = pair(self.patch_size)
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, self.dim))

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
    
    def get_embedding(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        return x

def eval_model(model, data_loader, criterion, device):
    # Evaluate the model on data from valloader
    correct = 0
    total = 0
    val_loss = 0
    model.eval()
    with torch.no_grad():
        for data in data_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    
    return val_loss / len(data_loader), 100 * correct / len(data_loader.dataset)


def main(epochs = 100,
         model_class = 'Plain-Old-CIFAR10',
         batch_size = 128,
         learning_rate = 1e-4,
         l2_regularization = 0.0):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    
    # Load and preprocess the dataset, feel free to add other transformations that don't shuffle the patches. 
    # (Note - augmentations are typically not performed on validation set)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4913, 0.4822, 0.4467), (0.2114, 0.2088, 0.2122))
        ])

    # Initialize training, validation and test dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainset, valset = torch.utils.data.random_split(trainset, [40000, 10000], generator=torch.Generator().manual_seed(0))

    # def compute_mean_std(loader):
    #     mean = 0.0
    #     var = 0.0
    #     for images, _ in loader:
    #         batch_samples = images.size(0)  # batch size (the last batch can have smaller size!)
    #         images = images.view(batch_samples, images.size(1), -1)
    #         mean += images.mean(2).sum(0)
    #         var += images.var(2).sum(0)
        
    #     mean /= len(loader.dataset)
    #     var /= len(loader.dataset)
    #     std = torch.sqrt(var)

    #     return mean, std

    # dataloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=False, num_workers=2)

    # mean, std = compute_mean_std(dataloader)
    # print(f'Mean: {mean}')
    # print(f'Std: {std}')
    # Mean: tensor([0.4913, 0.4822, 0.4467])
    # Std: tensor([0.2114, 0.2088, 0.2122])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Initialize Dataloaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size= batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    # models = {
    #     'ShallowCNN': ShallowCNN(),
    #     'CNNDropout': CNNDropout(),
    #     'CNNBatchNorm': CNNBatchNorm(),
    #     'SkipConnectionCNN': SkipConnectionCNN(),
    #     'UNetLike': UNetLike(),
    #     'VGG': VGG()
    # }
    # path = '/Users/tylerrowe/Desktop/College/24Spring/Computer Vision/Assignments/sedixit-tmrowe-shumehta-a2/model_results/'
    # metrics_filename = 'model_metrics.csv'
    # with open(path + metrics_filename, 'w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(['Model', 'Epoch', 'Train Loss', 'Train Accuracy', 'Validation Loss', 'Validation Accuracy'])
    # # save test loss and test accuracy to a csv file
    # test_filename = 'test_metrics.csv'
    # with open(path + test_filename, 'a', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(['Model', 'Test Loss', 'Test Accuracy'])
    # for model_name, model in models.items():
    # Initialize the model, the loss function and optimizer
    if model_class == 'Plain-Old-CIFAR10':
        net = Net().to(device)
        patch_size = 16
    elif model_class == 'D-shuffletruffle': 
        net = Net_D_shuffletruffle().to(device)
        patch_size = 16
    elif model_class == 'N-shuffletruffle':
        net = Net_N_shuffletruffle().to(device)
        patch_size = 8
    # print(f'Training {model_name}')
    print(net) # print model architecture
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr = learning_rate, weight_decay= l2_regularization)

    patch_shuffle_testset = PatchShuffled_CIFAR10(data_file_path = f'test_patch_{patch_size}.npz', transforms = transform)
    patch_shuffle_testloader = torch.utils.data.DataLoader(patch_shuffle_testset, batch_size=batch_size, shuffle=False)
    
    # Initialize CSV file and headers
    csv_file = open('C:/Users/postw/Important/Computer Vision/assignment_2/sedixit-tmrowe-shumehta-a2/model_results/model_results.csv', 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Epoch', 'Training Accuracy', 'Validation Accuracy', 'Training Loss', 'Validation Loss', 'Patch_16 Accuracy', 'Patch_16 Loss'])

    # Train the model
    try:
        train_loss, validation_loss = [], []
        train_acc, validation_acc = [], []
        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            net.train()
            for data in trainloader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()

            if epoch % 1 == 0:
                val_loss, val_acc = eval_model(net, valloader, criterion, device)
                patch_16_val_loss, patch_16_val_acc = eval_model(net, patch_shuffle_testloader, criterion, device)   # added getting patch validation acc every epoch
                print('epoch - %d loss: %.3f accuracy: %.3f val_loss: %.3f val_acc: %.3f patch_16_acc: %.3f' % (epoch, running_loss / len(trainloader), 100 * correct / len(trainloader.dataset), val_loss, val_acc, patch_16_val_acc))
            else:
                print('epoch - %d loss: %.3f accuracy: %.3f' % (epoch, running_loss / len(trainloader), 100 * correct / len(trainloader.dataset)))

            train_loss.append(running_loss / len(trainloader))
            train_acc.append(100 * correct / len(trainloader.dataset))

            validation_loss.append(val_loss)
            validation_acc.append(val_acc)

            # Write data to CSV file
            #csv_writer.writerow([epoch, train_acc[-1], validation_acc[-1], train_loss[-1], validation_loss[-1], patch_16_val_acc, patch_16_val_loss])

        print('Finished training')
        # with open(path + metrics_filename, 'a', newline='') as file:
        #     writer = csv.writer(file)
        #     for epoch in range(epochs):
        #         writer.writerow([model_name, epoch+1, train_loss[epoch], train_acc[epoch], validation_loss[epoch], validation_acc[epoch]]) 
        # print(f'Metrics for {model_name} written to {metrics_filename}')
    except KeyboardInterrupt:
        pass
    
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    net.eval()
    # Evaluate the model on the test set

    test_loss, test_acc = eval_model(net, testloader, criterion, device)
    print('Test loss: %.3f accuracy: %.3f' % (test_loss, test_acc))

    # save model name, test loss, test accuracy to a csv file
    # with open(path + test_filename, 'a', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow([model_name, test_loss, test_acc])

    # Evaluate the model on the patch shuffled test data

    patch_size = 16
    patch_shuffle_testset = PatchShuffled_CIFAR10(data_file_path = f'test_patch_{patch_size}.npz', transforms = transform)
    patch_shuffle_testloader = torch.utils.data.DataLoader(patch_shuffle_testset, batch_size=batch_size, shuffle=False)
    patch_shuffle_test_loss, patch_shuffle_test_acc = eval_model(net, patch_shuffle_testloader, criterion, device)
    print(f'Patch shuffle test loss for patch-size {patch_size}: {patch_shuffle_test_loss} accuracy: {patch_shuffle_test_acc}')

    patch_size = 8
    patch_shuffle_testset = PatchShuffled_CIFAR10(data_file_path = f'test_patch_{patch_size}.npz', transforms = transform)
    patch_shuffle_testloader = torch.utils.data.DataLoader(patch_shuffle_testset, batch_size=batch_size, shuffle=False)
    patch_shuffle_test_loss, patch_shuffle_test_acc = eval_model(net, patch_shuffle_testloader, criterion, device)
    print(f'Patch shuffle test loss for patch-size {patch_size}: {patch_shuffle_test_loss} accuracy: {patch_shuffle_test_acc}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', 
                        type=int, 
                        default= 25,
                        help= "number of epochs the model needs to be trained for")
    parser.add_argument('--model_class', 
                        type=str, 
                        default= 'Plain-Old-CIFAR10', 
                        choices=['Plain-Old-CIFAR10','D-shuffletruffle','N-shuffletruffle'],
                        help="specifies the model class that needs to be used for training, validation and testing.") 
    parser.add_argument('--batch_size', 
                        type=int, 
                        default= 100,
                        help = "batch size for training")
    parser.add_argument('--learning_rate', 
                        type=float, 
                        default = 0.001,
                        help = "learning rate for training")
    parser.add_argument('--l2_regularization', 
                        type=float, 
                        default= 0.0,
                        help = "l2 regularization for training")
    
    args = parser.parse_args()
    main(**vars(args))

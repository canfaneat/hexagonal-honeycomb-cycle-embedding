import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random

# 配置参数
cfg = {
    'data_path': r'C:\Users\14398\Documents\Tencent Files\1439840558\FileRecv\flower_photos\flower_photos',
    'image_size': 224,
    'batch_size': 64,
    'num_workers': 4,
    'num_classes': 5,
    'lr': 0.0007,  # 略微提升学习率
    'weight_decay': 0.02,
    'epochs': 60,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'save_path': './best_flower_resnet18.pth',
    'label_smoothing': 0.1,
    'early_stop_patience': 12,  # 增大patience
}

# 数据增强与预处理
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(cfg['image_size'], scale=(0.6, 1.0), ratio=(0.75, 1.333)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.2))  # 放在最后，概率适中
])
test_transform = transforms.Compose([
    transforms.Resize((cfg['image_size'], cfg['image_size'])),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def replace_relu_with_silu(model):
    for name, module in model.named_children():
        if isinstance(module, nn.ReLU):
            setattr(model, name, nn.SiLU(inplace=True))
        else:
            replace_relu_with_silu(module)

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return running_loss / total, correct / total

def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.axis('off')

if __name__ == '__main__':
    # 数据集加载
    train_dir = os.path.join(cfg['data_path'], 'train')
    val_dir = os.path.join(cfg['data_path'], 'validate')
    test_dir = os.path.join(cfg['data_path'], 'test')

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=test_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers'])
    test_loader = DataLoader(test_dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers'])

    class_names = train_dataset.classes
    print("类别映射：", class_names)

    # 构建模型（ResNet18+SiLU+Dropout=0.1）
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    replace_relu_with_silu(model)
    model.fc = nn.Sequential(
        nn.Dropout(0.1),  # Dropout概率调低
        nn.Linear(model.fc.in_features, cfg['num_classes'])
    )
    model = model.to(cfg['device'])

    # 损失函数：带标签平滑的交叉熵
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg['label_smoothing'])

    # 优化器：AdamW
    optimizer = optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])

    # 学习率调度器：CosineAnnealingWarmRestarts
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=8, T_mult=2)

    # 日志记录
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    best_acc = 0.0
    patience = cfg['early_stop_patience']
    patience_counter = 0

    for epoch in range(cfg['epochs']):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, cfg['device'])
        val_loss, val_acc = evaluate(model, val_loader, criterion, cfg['device'])
        scheduler.step(epoch + 1)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        print(f"Epoch [{epoch+1}/{cfg['epochs']}]: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), cfg['save_path'])
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}!")
                break

    print("Training finished. Best validation accuracy: {:.4f}".format(best_acc))

    # 训练日志可视化
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 测试集评估
    model.load_state_dict(torch.load(cfg['save_path']))
    test_loss, test_acc = evaluate(model, test_loader, criterion, cfg['device'])
    print("Test accuracy: {:.4f}".format(test_acc))

    # 随机挑选9张测试图片进行可视化
    model.eval()
    all_indices = list(range(len(test_dataset)))
    random.shuffle(all_indices)
    sample_indices = all_indices[:9]
    images = []
    labels = []
    for idx in sample_indices:
        img, label = test_dataset[idx]
        images.append(img)
        labels.append(label)
    images = torch.stack(images)
    labels = torch.tensor(labels)
    images = images.to(cfg['device'])
    labels = labels.to(cfg['device'])
    outputs = model(images)
    _, preds = torch.max(outputs, 1)

    plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i+1)
        imshow(images[i].cpu(), title=f"Pred:{class_names[preds[i]]}\nTrue:{class_names[labels[i]]}")
    plt.tight_layout()
    plt.show()
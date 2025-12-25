import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.optim.lr_scheduler import ReduceLROnPlateau
from net import UNet
from data import COCOSegmentationDataset

# 数据路径设置
train_dir = './dataset/train'
val_dir = './dataset/valid'
test_dir = './dataset/test'

train_annotation_file = './dataset/train/_annotations.coco.json'
test_annotation_file = './dataset/test/_annotations.coco.json'
val_annotation_file = './dataset/valid/_annotations.coco.json'

# 加载COCO数据集
train_coco = COCO(train_annotation_file)
val_coco = COCO(val_annotation_file)
test_coco = COCO(test_annotation_file)

# 超参数
BATCH_SIZE = 32
LEARNING_RATE = 5e-5  
NUM_EPOCHS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加权Dice Loss
def weighted_dice_loss(pred, target, weight=None, smooth=1e-6):
    """
    :param pred: 模型预测结果
    :param target: 真实标签
    :param weight: 权重张量，shape与target一致，或为None
    :param smooth: 平滑项，防止分母为零
    :return:
    """
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    if weight is not None:
        weight_flat = weight.view(-1)
        intersection = (weight_flat * pred_flat * target_flat).sum()
        union = (weight_flat * pred_flat).sum() + (weight_flat * target_flat).sum()
    else:
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
    return 1 - ((2. * intersection + smooth) / (union + smooth))

# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, weight=None):
    best_val_loss = float('inf')
    patience = 8
    patience_counter = 0
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    train_dice_scores = []
    val_dice_scores = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_acc = 0
        train_dice = 0
        
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks, weight)
            loss.backward()
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_acc += (outputs.round() == masks).float().mean().item()
            # 计算Dice分数
            w = weight[0] * (1 - masks) + weight[1] * masks if weight is not None else None
            dice = weighted_dice_loss(outputs, masks, w)
            train_dice += (1 - dice.item())
            
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        train_dice /= len(train_loader)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        train_dice_scores.append(train_dice)
        
        # 验证
        model.eval()
        val_loss = 0
        val_acc = 0
        val_dice = 0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks, weight)
                
                val_loss += loss.item()
                val_acc += (outputs.round() == masks).float().mean().item()
                w = weight[0] * (1 - masks) + weight[1] * masks if weight is not None else None
                dice = weighted_dice_loss(outputs, masks, w)
                val_dice += (1 - dice.item())
                
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        val_dice /= len(val_loader)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        val_dice_scores.append(val_dice)
        
        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train Dice: {train_dice:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val Dice: {val_dice:.4f}')
        
        # 学习率调度
        scheduler.step(val_loss)
        
        # 早停
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model_251218_01.pth')
            print("✓ 保存最优模型")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("早停触发")
                break
        print()

    # 绘制训练曲线
    plot_training_curves(train_losses, train_accuracies, val_losses, val_accuracies, 
                         train_dice_scores, val_dice_scores)

    # 返回记录的指标用于可能的进一步处理
    return train_losses, train_accuracies, val_losses, val_accuracies, train_dice_scores, val_dice_scores


def plot_training_curves(train_losses, train_accuracies, val_losses, val_accuracies, 
                         train_dice_scores=None, val_dice_scores=None):
    """
    绘制训练和验证的损失、准确率以及Dice曲线
    """
    epochs = range(1, len(train_losses) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 绘制损失曲线
    axes[0, 0].plot(epochs, train_losses, 'bo-', label='Train Loss')
    axes[0, 0].plot(epochs, val_losses, 'ro-', label='Val Loss')
    axes[0, 0].set_title('Train and Val Loss')
    axes[0, 0].set_xlabel('Epochs')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # 绘制准确率曲线
    axes[0, 1].plot(epochs, train_accuracies, 'bo-', label='Train Acc')
    axes[0, 1].plot(epochs, val_accuracies, 'ro-', label='Val Acc')
    axes[0, 1].set_title('Train and Val Accuracy')
    axes[0, 1].set_xlabel('Epochs')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # 绘制Dice分数曲线（如果有）
    if train_dice_scores and val_dice_scores:
        axes[1, 0].plot(epochs, train_dice_scores, 'go-', label='Train Dice')
        axes[1, 0].plot(epochs, val_dice_scores, 'mo-', label='Val Dice')
        axes[1, 0].set_title('Train and Val Dice Score')
        axes[1, 0].set_xlabel('Epochs')
        axes[1, 0].set_ylabel('Dice Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

    # 绘制过拟合间隙（Val Loss - Train Loss）
    overfit_gap = [v - t for v, t in zip(val_losses, train_losses)]
    axes[1, 1].plot(epochs, overfit_gap, 'co-', label='Overfit Gap (Val-Train)')
    axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1, 1].set_title('Overfitting Gap')
    axes[1, 1].set_xlabel('Epochs')
    axes[1, 1].set_ylabel('Loss Difference')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig('training_curves_251218_01.png', dpi=100)
    plt.show()


def main():
    device = DEVICE
    
    # 使用Albumentations，确保image/mask同步
    transform_train = A.Compose([
        A.HorizontalFlip(p=0.5), 
        A.VerticalFlip(p=0.5),
        A.Affine(scale=(0.9, 1.1), rotate=(-30, 30), p=0.8),
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    # 验证集和测试集只做标准处理
    transform_val = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    # 创建数据集
    train_dataset = COCOSegmentationDataset(train_coco, train_dir, transform=transform_train)
    val_dataset = COCOSegmentationDataset(val_coco, val_dir, transform=transform_val)
    test_dataset = COCOSegmentationDataset(test_coco, test_dir, transform=transform_val)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=0)

    # 初始化改进版模型
    model = UNet(n_filters=32).to(device)

    # 设置优化器（添加weight_decay L2正则化）
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # 设置学习率调度器（当验证损失不下降时自动降低学习率）
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)

    # 设置权重（前景/肿瘤权重较大）
    weight = torch.tensor([0.15, 0.85]).to(device)  # 0为背景，1为肿瘤
    
    # 损失函数
    bce_loss = nn.BCELoss()
    def criterion(pred, target, weight):
        w = weight[0] * (1 - target) + weight[1] * target
        dice = weighted_dice_loss(pred, target, w)
        bce = bce_loss(pred, target)
        return 0.7 * dice + 0.3 * bce
    
    # 训练模型
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=NUM_EPOCHS,
        device=device,
        weight=weight,
    )

    # 在测试集上评估
    print("\n" + "=" * 60)
    print("在测试集上评估:")
    model.eval()
    test_loss = 0
    test_acc = 0
    test_dice = 0

    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            
            w = weight[0] * (1 - masks) + weight[1] * masks
            dice = weighted_dice_loss(outputs, masks, w)
            bce = bce_loss(outputs, masks)
            loss = 0.7 * dice + 0.3 * bce
            
            test_loss += loss.item()
            test_acc += (outputs.round() == masks).float().mean().item()
            test_dice += (1 - dice.item())

    test_loss /= len(test_loader)
    test_acc /= len(test_loader)
    test_dice /= len(test_loader)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Dice Score: {test_dice:.4f}")


if __name__ == '__main__':
    main()

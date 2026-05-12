# --------------------------
# 第一步：导入所有需要的第三方库
# --------------------------
# torch：深度学习核心框架，提供张量计算、自动微分和模型定义功能
import torch
# torch.nn：PyTorch的神经网络模块，包含预定义的层（Linear、Conv2d）和损失函数
import torch.nn as nn
# Dataset：自定义数据集的基类
# DataLoader：批量数据加载器，用于自动分批次读取数据
# random_split：随机划分数据集的工具
# Subset：从完整数据集中取出指定索引的子集
from torch.utils.data import Dataset, DataLoader, random_split, Subset
# transforms：图像预处理和数据增强工具
# models：PyTorch官方预训练模型库
from torchvision import transforms, models
# pandas：数据处理库，用于读取和操作CSV表格文件
import pandas as pd
# numpy：数值计算库，用于处理数组和矩阵运算
import numpy as np
# matplotlib.pyplot：绘图库，用于绘制训练曲线
import matplotlib.pyplot as plt
# Image：PIL库的图像类，用于打开、处理和保存图片文件
from PIL import Image
# os：操作系统接口库，用于检查文件是否存在、路径拼接等
import os


# --------------------------
# 第二步：定义自定义数据集类（必须继承自torch.utils.data.Dataset）
# --------------------------
class affectnet_RAF(Dataset):
    """
    自定义数据集类，用于读取AffectNet/RAF格式的表情数据集
    """
    def __init__(self, csv_path, transform=None):
        """
        类的初始化函数，在创建对象时自动调用
        参数：
            csv_path：标签CSV文件的路径
            transform：要应用的图像预处理（数据增强/归一化）
        """
        # 1. 用pandas读取CSV标签文件，存储为DataFrame表格格式
        self.df = pd.read_csv(csv_path)
        # 2. 保存传入的图像预处理函数，供后面使用
        self.transform = transform
        # 3. 定义8种表情标签的列表（顺序必须和训练时完全一致！）
        self.emotion_labels = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        
        # 4. 创建一个字典，用于把表情字符串（如'happy'）转换成数字索引（如4）
        # 模型只能接受数字作为标签，不能直接接受字符串
        self.label_to_idx = {label: idx for idx, label in enumerate(self.emotion_labels)}

        # 5. 提前检查所有图片文件是否存在，过滤掉不存在的行
        print("正在检查文件是否存在...")
        # 创建一个空列表，只保存文件存在的行
        valid_rows = []
        # 遍历CSV里的每一行（idx是行号，row是行数据）
        for idx, row in self.df.iterrows():
            # 从当前行里提取图片路径
            img_path = row['pth']
            # 检查这个路径的文件是否真的存在于磁盘上
            if os.path.exists(img_path):
                # 如果存在，把这一行加入有效行列表
                valid_rows.append(row)
            else:
                # 如果不存在，打印警告信息，跳过这一行
                print(f"⚠️  文件不存在，已跳过：{img_path}")

        # 6. 用有效的行重新生成DataFrame，替换原来的df
        self.df = pd.DataFrame(valid_rows)
        # 打印检查完成的信息和有效文件数量
        print(f"✅ 检查完成，共 {len(self.df)} 个有效文件")

    def __len__(self):
        """
        必须实现的方法：返回数据集的总样本数
        """
        # 返回DataFrame的行数，也就是总样本数
        return len(self.df)

    def __getitem__(self, idx):
        """
        必须实现的方法：根据索引idx返回对应的（图片, 标签）对
        参数：
            idx：样本的索引（0到总样本数-1）
        返回：
            image：预处理后的图片张量
            label：对应的数字标签
        """
        # 1. 获取DataFrame里第idx行的数据
        row = self.df.iloc[idx]
        # 2. 从行里提取图片路径（对应CSV里的'pth'列）
        img_path = row['pth']
        # 3. 从行里提取表情字符串（对应CSV里的'label'列）
        label_str = row['label']
        # 4. 用之前创建的字典，把表情字符串转换成数字索引
        label = self.label_to_idx[label_str]
        # 5. 用PIL打开图片文件，并强制转换成RGB格式（确保是3通道彩色图）
        image = Image.open(img_path).convert('RGB')
        
        # 6. 如果传入了预处理函数，就把它应用到图片上
        if self.transform:
            image = self.transform(image)
        
        # 7. 返回（预处理后的图片张量, 数字标签）
        return image, label


# --------------------------
# 第三步：准备数据集和 DataLoader
# --------------------------
print("=== 准备数据集 ===")

# ✅ 分别定义训练集和验证集的预处理
# 训练集预处理：包含数据增强，用来提升模型的泛化能力
train_transform = transforms.Compose([
    # 1. 把图片缩放到224x224像素（和预训练模型的输入大小一致）
    transforms.Resize((224, 224)),
    # 2. 随机水平翻转：50%的概率把图片左右翻转，增加数据多样性
    transforms.RandomHorizontalFlip(p=0.5),
    # 3. 随机旋转：把图片随机旋转±10度
    transforms.RandomRotation(degrees=10),
    # 4. 随机亮度和对比度：轻微改变图片的亮度和对比度
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    # 5. 转换成Tensor：把PIL Image转换成PyTorch张量，并把像素值从0-255归一化到0-1
    transforms.ToTensor(),
    # 6. 标准化：用ImageNet数据集的均值和标准差进行标准化
    # 预训练模型是在ImageNet上训练的，必须用同样的统计量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 验证集预处理：不包含任何数据增强，只做最基本的缩放、转Tensor和标准化
# ⚠️ 验证集绝对不能用数据增强！否则验证准确率是假的，没有参考价值
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 创建一个不带transform的完整数据集，只用来获取索引和统计信息
full_dataset = affectnet_RAF(csv_path='labels.csv')

# 计算训练集和验证集的大小：80%用来训练，20%用来验证
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

# ✅ 先随机划分索引，而不是直接划分数据集
# 这样我们可以给训练集和验证集分别应用不同的transform
train_indices, val_indices = random_split(
    range(len(full_dataset)),          # 要划分的索引范围（0到总样本数-1）
    [train_size, val_size],            # 划分比例
    generator=torch.Generator().manual_seed(42) # 固定随机种子，保证每次运行划分结果一样
)

# ✅ 用Subset分别创建带不同transform的训练集和验证集
# Subset的作用：从完整数据集中取出指定的索引，并应用指定的transform
train_dataset = Subset(
    affectnet_RAF(csv_path='labels.csv', transform=train_transform), # 用训练集transform
    train_indices # 只取训练集的索引
)
val_dataset = Subset(
    affectnet_RAF(csv_path='labels.csv', transform=val_transform), # 用验证集transform
    val_indices # 只取验证集的索引
)

# 创建DataLoader：把数据集变成可迭代的批量数据加载器
train_loader = DataLoader(
    train_dataset,  # 要加载的数据集
    batch_size=32,  # 每一批的样本数（32张图片一批）
    shuffle=True,   # 训练集要打乱顺序，防止模型记住样本顺序
    num_workers=0   # 数据加载的进程数（Windows下建议设为0，避免报错）
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,  # 验证集不需要打乱顺序
    num_workers=0
)

# 打印数据集信息
print(f"总数据集大小：{len(full_dataset)}")
print(f"训练集大小：{len(train_dataset)}")
print(f"验证集大小：{len(val_dataset)}")
print("-" * 50)


# --------------------------
# 第四步：准备模型、损失函数、优化器
# --------------------------
print("=== 准备 MobileNetV3-Small 模型 ===")

# 1. 加载预训练的MobileNetV3-Small模型骨架
# weights=...DEFAULT：加载在ImageNet上预训练好的权重
model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)

# 2. 先冻结模型的所有层（把requires_grad设为False）
# 冻结意味着这些层的参数不会在训练中更新
for param in model.parameters():
    param.requires_grad = False

# ✅ 解冻最后两个特征块（用切片 [-2:]）
# model.features[-2:]：倒数第2层和倒数第1层（最后两个特征提取块）
# 越靠后的层学到的特征越高级，和表情识别越相关，所以我们只微调这几层
for param in model.features[-2:].parameters():
    param.requires_grad = True # 把requires_grad设为True，允许这些层的参数更新

# 3. 修改模型的最后一层（分类头）
# MobileNetV3的最后一层在 model.classifier[-1]
# in_features：最后一层的输入特征数
in_features = model.classifier[-1].in_features
# 把输出从原来的1000类（ImageNet的类别数）改成我们的8类表情
model.classifier[-1] = nn.Linear(in_features, 8)

# 4. 自动选择设备：如果有GPU就用GPU，没有就用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 把模型移动到指定设备上
model = model.to(device)
print(f"使用设备：{device}")

# 5. 计算类别权重，处理类别不平衡问题
# 有些表情的样本很多（比如happy），有些很少（比如disgust）
# 我们给样本少的类别更高的权重，让模型更关注这些少数类
class_counts = full_dataset.df['label'].value_counts().sort_index().values
# 计算权重公式：总样本数 / (类别数 * 该类样本数)
class_weights = torch.tensor(len(full_dataset) / (8 * class_counts), dtype=torch.float32).to(device)
# 用带权重的交叉熵损失函数
criterion = nn.CrossEntropyLoss(weight=class_weights)

# ✅ 分层设置学习率
# 前面解冻的层用更小的学习率，避免破坏预训练学到的特征
# 最后分类层用更大的学习率，因为它是从头开始学的
optimizer = torch.optim.Adam([
    {'params': model.features[-2].parameters(), 'lr': 0.000005}, # 倒数第二层：最小学习率
    {'params': model.features[-1].parameters(), 'lr': 0.00001},  # 倒数第一层：中等学习率
    {'params': model.classifier[-1].parameters(), 'lr': 0.0001}   # 最后分类层：最大学习率
])

# 6. 学习率调度器：随着训练轮数增加，逐渐降低学习率
# StepLR：每7轮（step_size），学习率乘以0.5（gamma）
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)

print("✅ 模型准备完成")
print("-" * 50)


# --------------------------
# 第五步：定义训练函数
# --------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    训练一个epoch的函数
    参数：
        model：要训练的模型
        loader：训练集DataLoader
        criterion：损失函数
        optimizer：优化器
        device：设备（CPU/GPU）
    返回：
        epoch_loss：本轮的平均训练损失
        epoch_acc：本轮的平均训练准确率
    """
    # 1. 把模型设置为训练模式
    # 训练模式会启用Dropout和BatchNorm的训练行为
    model.train()
    # 初始化统计变量
    running_loss = 0.0 # 累计损失
    correct = 0        # 预测正确的样本数
    total = 0          # 总样本数

    # 2. 遍历训练集的每一个batch
    for images, labels in loader:
        # 把数据移动到指定设备上
        images = images.to(device)
        labels = labels.to(device)
        
        # 3. 前向传播：把图片输入模型，得到预测结果
        outputs = model(images)
        # 4. 计算损失：比较预测结果和真实标签
        loss = criterion(outputs, labels)

        # 5. 反向传播 + 参数更新
        optimizer.zero_grad() # 第一步：清空上一轮的梯度（否则梯度会累加）
        loss.backward()       # 第二步：反向传播，计算当前梯度
        optimizer.step()      # 第三步：用优化器更新模型参数
          
        # 6. 统计本轮的损失和准确率
        running_loss += loss.item() * images.size(0) # 累计损失（乘以batch_size，因为loss是平均的）
        # 在输出的第1维（类别维）找最大值的索引，就是预测的类别
        _, predict = torch.max(outputs.data, 1)
        total += labels.size(0)                     # 累计总样本数
        correct += (predict == labels).sum().item() # 累计预测正确的样本数
    
    # 7. 计算本轮的平均损失和准确率
    epoch_loss = running_loss / total # 平均损失 = 总损失 / 总样本数
    epoch_acc = correct / total       # 准确率 = 正确数 / 总样本数

    return epoch_loss, epoch_acc


# --------------------------
# 第六步：定义验证函数
# --------------------------
def validate(model, loader, criterion, device):
    """
    验证一个epoch的函数（和训练函数类似，但不需要反向传播）
    """
    # 1. 把模型设置为评估模式
    # 评估模式会关闭Dropout，并使用BatchNorm的全局统计量
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    # 2. 验证时不需要计算梯度，节省内存和速度
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            # 3. 前向传播（只有前向，没有反向）
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 4. 统计损失和准确率（和训练时一样）
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


# --------------------------
# 第七步：主训练循环
# --------------------------
print("=== 开始训练 ===")

num_epochs = 30 # 总共训练30轮
# 初始化列表，用来记录每一轮的损失和准确率，方便后面画图
train_losses = []
train_accs = []
val_losses = []
val_accs = []
best_acc = 0.0 # 记录最高的验证准确率，用来保存最优模型

# 主循环：从第0轮到第29轮（共30轮）
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}") # 打印当前是第几轮

    # 1. 训练一轮
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    # 2. 验证一轮
    val_loss, val_acc = validate(model, val_loader, criterion, device)

    # 3. 更新学习率（每轮结束后调用）
    scheduler.step()

    # 4. 把本轮的结果记录到列表里
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    # 5. 打印本轮的结果
    print(f"  训练损失：{train_loss:.4f} | 训练准确率：{train_acc:.4f}")
    print(f"  验证损失：{val_loss:.4f} | 验证准确率：{val_acc:.4f}")
    print(f"  当前学习率：{scheduler.get_last_lr()[0]:.6f}") # 打印当前学习率

    # ✅ 只保存验证准确率最高的模型
    if val_acc > best_acc:
        # 如果当前验证准确率比历史最高还高，更新最高准确率
        best_acc = val_acc
        # 保存当前模型的权重
        torch.save(model.state_dict(), "best_mobilenet_affectnet.pth")
        print(f"  ✅ 新的最优模型已保存！最高验证准确率：{best_acc:.4f}")
    else:
        # 如果没有提升，只打印当前最高准确率
        print(f"  验证准确率未提升，当前最高：{best_acc:.4f}")
    
    print("-" * 50)

# 训练结束，打印最终结果
print(f"🎉 训练完成！最终最高验证准确率：{best_acc:.4f}")
print("-" * 50)


# --------------------------
# 第八步：绘制训练曲线
# --------------------------
print("=== 绘制训练曲线 ===")

# 创建一个12x5大小的画布
plt.figure(figsize=(12, 5))

# 左边的子图：损失曲线
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs+1), train_losses, label="Train Loss") # 训练损失
plt.plot(range(1, num_epochs+1), val_losses, label="Val Loss")     # 验证损失
plt.xlabel("Epoch")          # x轴标签
plt.ylabel("Loss")           # y轴标签
plt.title("Loss Curve")      # 标题
plt.legend()                  # 显示图例
plt.grid(True)                # 显示网格

# 右边的子图：准确率曲线
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs+1), train_accs, label="Train Acc") # 训练准确率
plt.plot(range(1, num_epochs+1), val_accs, label="Val Acc")     # 验证准确率
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy Curve")
plt.legend()
plt.grid(True)

# 自动调整子图间距，避免重叠
plt.tight_layout()
# 保存图片到本地
plt.savefig("training_curve_optimized.png")
# 显示图片
plt.show()

print("✅ 训练曲线已保存为 training_curve_optimized.png")
print("-" * 50)
print("所有任务完成！")
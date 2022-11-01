import torch
from torchvision import transforms, models, datasets
from torch import nn


def data_augmentation():
    '''
    数据增强
    :return: 含数据增强操作的变换器
    '''
    data_transform = {
        'train': transforms.Compose([
            transforms.RandomRotation(45),  # 随机旋转，角度在-45到45度之间
            transforms.CenterCrop(224),  # 从中心开始剪裁
            transforms.RandomHorizontalFlip(p=0.5),  # 以0.5的概率水平翻转
            transforms.RandomVerticalFlip(p=0.5),  # 以0.5的概率垂直翻转
            transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),  # 参数依次为亮度、对比度、饱和度、色相
            transforms.RandomGrayscale(p=0.025),  # 以0.025的概率变为灰度图像，3通道即R=G=B
            transforms.ToTensor(),  # 将0-255的像素进行归一化
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 使用均值和标准差标准化三个通道的数据
        ]),
        'test': transforms.Compose([  # 测试集如果也进行归一化则参数必须和训练集一致
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    return data_transform


def data_load(train_dir, test_dir, batch_size, data_transform):
    '''
    构建数据管道
    :param train_dir: 训练集所在文件夹
    :param test_dir: 验证集所在文件夹
    :param batch_size: 每次迭代的批量大小
    :param data_transform: 数据增强转换器
    :return: 返回数据管道
    '''
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, data_transform['train']),
        'test': datasets.ImageFolder(test_dir, data_transform['test'])
    }

    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True),
        'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=batch_size, shuffle=True)
    }
    return dataloaders


def get_model(features):
    '''
    构建模型
    :param features: 最后需要分类的类别数
    :return: 模型和需要训练的参数
    '''
    # 默认使用残差神经网络的预训练模型
    model = models.resnet152(pretrained=True)

    for params in model.parameters():
        # 冻结模型每一层
        params.requires_grad = False

    # 改变最后一层线性层的输出大小
    # .fc等参数都是根据打印的模型来的
    num_origin = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_origin, features), nn.LogSoftmax(dim=1))

    param_learn = []
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            param_learn.append(param)
    return model, param_learn



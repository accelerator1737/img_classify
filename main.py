from module import *
from torch import optim
from train_model import train_best

data_dir = r'102flowers\\'
train_dir = data_dir + 'train'
test_dir = data_dir + 'valid'

transform = data_augmentation()

dataloaders = data_load(train_dir, test_dir, 16, transform)

model, params_learn = get_model(102)

# 设置优化器对哪些参数进行优化
optimizer = optim.Adam(params_learn, lr=1e-2)
# 学习率衰减，每7个epoch学习率衰减为原来的0.1倍
sch = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
# 损失函数
criterion = nn.CrossEntropyLoss()

train_best(model, 5, dataloaders, optimizer, criterion, 'model.h5')


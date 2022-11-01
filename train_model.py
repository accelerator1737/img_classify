import copy
import torch
import pandas as pd
import datetime





def train_best(model, num_epoch, dataloaders, optimizer, loss_function, save_path, log_step=5):
    '''
    训练出最好的模型
    :param model: 构建的模型
    :param num_epoch: 总的要训练的轮次
    :param dataloaders: 数据管道
    :param optimizer: 优化器
    :param loss_function: 损失函数
    :param save_path: 保存的最优模型的路径
    :param log_step: 默认每5个训练batch打印一次数据
    :return: 每一个epoch的信息
    '''
    best_acc = 0
    # 最优模型
    best_model = copy.deepcopy(model.state_dict())

    # 保存每一个epoch的信息
    dfhistory = pd.DataFrame(columns=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print("Start Training...\n")
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("==========" * 8 + "%s\n" % nowtime)

    for i in range(num_epoch):

        # 1，训练循环----------------------------------------------------------------

        loss_sum = 0.0
        metric_sum = 0.0
        # 训练模式，可以更新参数
        model.train()
        # 将数据全部取完

        # 记录每一个batch
        step = 0
        # 记录取了多少个数据
        all_step = 0

        for inputs, labels in dataloaders['train']:
            inputs = inputs.to(device)
            labels = labels.to(device)
            # 梯度清零，防止累加
            optimizer.zero_grad()

            # 每一批次拿了多少张图像
            a = inputs.size(0)

            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            # 返回每一行的最大值和其索引
            _, pred = torch.max(outputs, 1)
            loss.backward()
            optimizer.step()

            # 学习损失
            loss_sum += loss.item() * inputs.size(0)
            metric_sum += torch.sum(pred == labels.data)

            step += 1
            all_step += a

            if step % log_step == 0:
                print("[step = {}]  train_loss = {:.3f}, train_acc = {:.3f}".
                      format(all_step, loss_sum / all_step, metric_sum.double() / all_step))

        train_loss = loss_sum / len(dataloaders['train'].dataset)
        train_acc= metric_sum.double() / len(dataloaders['train'].dataset)

        # 2，验证循环----------------------------------------------------------------

        val_loss_sum = 0.0
        val_metric_sum = 0.0

        step = 0
        all_step = 0

        # 验证模式，该模式下模型参数不能进行修改
        model.eval()

        for inputs, labels in dataloaders['test']:
            inputs = inputs.to(device)
            labels = labels.to(device)
            # 梯度清零，防止累加
            optimizer.zero_grad()

            a = inputs.size(0)

            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            # 返回每一行的最大值和其索引
            _, pred = torch.max(outputs, 1)

            # 学习损失
            val_loss_sum += loss.item() * inputs.size(0)
            val_metric_sum += torch.sum(pred == labels.data)

            step += 1
            all_step += a

            if step % log_step == 0:
                print("[step = {}]  val_loss = {:.3f}, val_acc = {:.3f}".

                      format(all_step, val_loss_sum / all_step, val_metric_sum.double() / all_step))

        val_loss = val_loss_sum / len(dataloaders['test'].dataset)
        val_acc = val_metric_sum.double() / len(dataloaders['test'].dataset)

        # 3. 打印epoch级别日志
        print("EPOCH = {}/{}  train_loss = {:.3f}, train_acc = {:.3f}, val_loss = {:.3f}, val_acc = {:.3f}\n".
              format(i, num_epoch, train_loss, train_acc, val_loss, val_acc))

        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("==========" * 8 + "%s\n" % nowtime)

        # 4. 保存最优的模型
        if val_acc > best_acc:
            best_acc = val_acc
            best_model = copy.deepcopy(model.state_dict())
            state = {
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict()
            }
            torch.save(state, save_path)

        dfhistory.loc[i] = (i, train_loss, train_acc, val_loss, val_acc)
    print('Finished Training...\n')

    return dfhistory
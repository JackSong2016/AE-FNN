import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import os
import time
import xlwt
from sklearn.metrics import *

start_time = time.time()

BATCH_SIZE=64
IMG_W=90
IMG_H=50
# number_conv1 = 128
# number_conv2 = 64
# number_conv3 = 32
# number_neuron = 256

number_conv1 = 64
number_conv2 = 128
number_conv3 = 256
number_neuron = 512

# number_conv1 = 32
# number_conv2 = 64‘
# number_conv3 = 80
# number_neuron = 128

seed=1314
#seed=1234
LR = 1e-3              # learning rate
EPOCH_COUNT = 3
fnn_epochs = 150000
#原始数据集
# train_dir=os.getcwd()+"\\Distrainsample"
# test_dir=os.getcwd()+"\\Dispresample"
train_dir=r"E:\disorder\AutoEncoder\AE+FNN\Distrainsample"
test_dir=r"E:\disorder\AutoEncoder\AE+FNN\Dispresample"
# 创建图片标签路径列表和标签列表,且一一对应
# train_images=[]                     # 训练集图片
# train_labels=os.listdir(train_dir)  # 训练集标签
# for item in train_labels:
#     train_images.append(os.path.join(train_dir,item,"{}.png".format(item)))

# 2023.2.3
train_images=[]
train_labels=[float(item.strip(" .png")) for item in os.listdir(train_dir)]
for it in train_labels:
    train_images.append(os.path.join(train_dir,"{} .png".format(it)))
# print(train_labels)

list_test=os.listdir(test_dir)
test_images=[]      # 测试集图片
test_labels=[]      # 测试集标签
for item in list_test:
    test_images.append(os.path.join(test_dir,item))
    test_labels.append(item.split(".png")[0])


# print(train_images)
# print(pre_labels)
# print(test_images)
# print(test_labels)
# 定义自己的数据集
# 图像预处理
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 自定义训练集
class MyTrainset(Dataset):
    imgs=[]
    labels=[]
    def __init__(self,transform=img_transform,target_transform=None):
        self.imgs=train_images
        self.labels=train_labels
        self.transform=transform
        self.target_transform=target_transform
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self,index):
        image=self.imgs[index]
        label=self.labels[index]
        img=Image.open(image).convert('RGB')
        img=img.resize((IMG_W,IMG_H))   # img.resize((width,height))
        if self.transform is not None:
            img=self.transform(img)
        label=np.array(label).astype(np.float)
        label=torch.from_numpy(label)   # <class 'torch.Tensor'>
        return img,label
trainSet=MyTrainset()   # 实例化该类
trainSets = torch.utils.data.DataLoader(
                dataset=trainSet,batch_size=BATCH_SIZE,shuffle=True,num_workers=6)

# 自定义测试集
class MyTestset(Dataset):
    imgs=[]
    labels=[]
    def __init__(self,transform=img_transform,target_transform=None):
        self.imgs=test_images
        self.labels=test_labels
        self.transform=transform
        self.target_transform=target_transform
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self,index):
        image=self.imgs[index]
        label=self.labels[index]
        img=Image.open(image).convert('RGB')
        img=img.resize((IMG_W,IMG_H))   # img.resize((width,height))
        # plt.imshow(img)
        # plt.show()
        if self.transform is not None:
            img=self.transform(img)
        label=np.array(label).astype(np.float)
        label=torch.from_numpy(label)   # <class 'torch.Tensor'>
        return img,label
testSet=MyTestset()
testSets=torch.utils.data.DataLoader(dataset=testSet,batch_size=1,shuffle=True,num_workers=6)

# 编码解码卷积神经网络
class AutoEncoder(nn.Module):
    def __init__(self):
        torch.manual_seed(seed=seed)
        super(AutoEncoder, self).__init__()
        #编码
        self.encoder = nn.Sequential(
            # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
            nn.Conv2d(in_channels=3, out_channels=number_conv1, kernel_size=3, stride=2, padding=1),  # b, 16, 45, 25
            #nn.ReLU(True),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # b, 16, 22, 12
            nn.Conv2d(in_channels=number_conv1, out_channels=number_conv2, kernel_size=3, stride=2, padding=1),  # b, 32, 11, 6
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # b, 8, 5, 3
            nn.Conv2d(in_channels=number_conv2, out_channels=number_conv3, kernel_size=3, stride=2, padding=1),  # b, 64, 3, 2
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # b, 64, 1, 1
        )
        #解码
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=number_conv3, out_channels=number_conv2,kernel_size=(4,5), stride=2,padding=1),  # b, 32, 3, 2
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=number_conv2, out_channels=number_conv1, kernel_size=3, stride=5,padding=1),  # b, 16, 11, 6
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=number_conv1, out_channels=number_conv1, kernel_size=5, stride=4),  # b, 3, 45, 25
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=number_conv1, out_channels=3, kernel_size=2, stride=2),  # b, 3, 50, 90
            nn.Tanh()
        )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


Autoencodermode = AutoEncoder()
# print(model)
optimizer = torch.optim.SGD(Autoencodermode.parameters(), lr=LR)
#optimizer = torch.optim.Adam(Autoencodermode.parameters(), lr=LR)
criterion = nn.MSELoss()

def save_matrix(matrix1,matrix2):
    book=xlwt.Workbook(encoding='utf-8',style_compression=0)
    sheet1=book.add_sheet('train_encoded',cell_overwrite_ok=True)
    for i in range(len(matrix1)):
        for j in range(len(matrix1[i])):
            sheet1.write(i,j,str(matrix1[i][j]))
    sheet2= book.add_sheet('test_encoded', cell_overwrite_ok=True)
    for i in range(len(matrix2)):
        for j in range(len(matrix2[i])):
            sheet2.write(i,j,str(matrix2[i][j]))
    book.save('encoded.xls')
# acu二维张量
def accuracy(acu,pre):
    acc=1-torch.abs(torch.sub(acu,pre)).div(acu)
    return 100*torch.mean(acc)

# 定义网络模型
class FnnNet(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(FnnNet, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.relu=nn.ReLU()
        self.predict = torch.nn.Linear(n_hidden, n_output)
    # 定义前向运算
    def forward(self, x):
        h1 = self.hidden(x)
        a1 = self.relu(h1)
        out = self.predict(a1)
        #a2 = F.relu(out)
        return out
# 参数依次是 n_feature, n_hidden, n_output
FnnModel = FnnNet(number_conv3,number_neuron,1)

# 自定义Loss
class CustomLoss(nn.Module):  # 注意继承 nn.Module
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, x, y):
        mse_loss = torch.mean(torch.pow((x - y), 2),dtype=torch.float32)
        return mse_loss       # 注意最后只能返回Tensor值，且带梯度，即 loss.requires_grad == True


loss_fn = CustomLoss()     # 实例化MSELoss，使用默认设置
#FnnOptimizer = torch.optim.Adam(FnnModel.parameters(), lr=LR)     # 前馈神经网络优化器
FnnOptimizer = torch.optim.SGD(FnnModel.parameters(), lr=LR)

def algorithm():
    for epoch in range(EPOCH_COUNT):
        # 循环分批次读取训练集中的所有数据
        if epoch + 1 == EPOCH_COUNT:
            list_tensor_train_feature = []
            list_tensor_train_labels = []
            list_tensor_test_feature = []
            list_tensor_test_labels = []
        # data原数据、labels热导率、encoded编码后的压缩空间、decoded解码后的数据
        for batch_id, (data, labels) in enumerate(trainSets, 1):
            Autoencodermode.train()
            optimizer.zero_grad()
            encoded, decoded = Autoencodermode(data)  # 每一批数据经过自动编码器
            if epoch + 1 == EPOCH_COUNT:
                list_tensor_train_feature.append(encoded)
                list_tensor_train_labels.append(labels)
            loss_auto = criterion(decoded, data)  # 自动编解码器loss
            loss_auto.backward()
            optimizer.step()

        print('epoch [{}/{}], loss:{:.4f}'
              .format(epoch + 1, EPOCH_COUNT, loss_auto.data))
        # 训练集
        if epoch + 1 == EPOCH_COUNT:
            # list_tensor_train_feature.append(encoded)
            # list_tensor_train_labels.append(labels)
            # 训练集编码后的压缩特征和对应的标签
            x_train = torch.cat(list_tensor_train_feature).reshape(-1, number_conv3)  # torch.Size([200, 64]) 编码后的特征
            y_train = torch.cat(list_tensor_train_labels).reshape(-1, 1)  # torch.Size([200, 1])   对应的热导率
            x_numpy = x_train.detach().numpy()
            y_numpy = y_train.detach().numpy()
            np.savetxt("x_numpy.dat", x_numpy.reshape(-1, number_conv3), fmt="%.4f", delimiter=",")
            x_train = torch.from_numpy(x_numpy)
            y_train = torch.from_numpy(y_numpy)
            # print(x_train)
            # print(y_train)
            # 模型训练
            loss_history = []  # 训练过程中的loss数据
            y_pred_history = []  # 中间的预测结果
            acc=[]
            with open("loss.dat", 'w') as files:
                with open("accuracy.dat",'w') as f:
                    for i in range(fnn_epochs):

                        # (1) 前向计算
                        FnnOptimizer.zero_grad()
                        y_pred = FnnModel(x_train)
                        # print(y_pred)
                        # print(y_pred.type)
                        # print(y_pred.dtype)
                        # print(x_train.dtype)
                        # (2) 计算loss
                        loss_fnn = loss_fn(y_pred, y_train)
                        accu=accuracy(y_train,y_pred)
                        # print(loss)
                        # loss=torch.tensor(loss,dtype=torch.float32,requires_grad=True)
                        # print(loss)
                        # print(loss.dtype)    # torch.float32
                        # FnnOptimizer.zero_grad()
                        # (3) 反向传播求梯度
                        loss_fnn.backward()
                        # torch.autograd.backward(loss)
                        # (4) 更新所有参数
                        FnnOptimizer.step()

                        # # (5) 复位优化器的梯度,将梯度初始化为零
                        # FnnOptimizer.zero_grad()

                        # 记录训练数据
                        loss_history.append(loss_fnn.item())
                        y_pred_history.append(y_pred.data)
                        acc.append(accu.item())

                        if (i % 1000 == 0):
                            print('epoch {}  loss {:.4f} accuracy {: .4f}'.format(i, loss_fnn.item(),accu.item()))
                            files.write(str(i))
                            files.write("   ")
                            files.write(str(loss_fnn.item()))
                            files.write('\n')

                            # f.write(str(i))
                            # f.write("   ")
                            f.write(str(accu.item()))
                            f.write('\n')
                    print("\n迭代完成")
                    print("final loss =", loss_fnn.item())
                    # print(type(loss_fnn.item()))  # <class 'float'>
                    # print(len(loss_history))
                    # print(len(y_pred_history))
                f.close()
            files.close()

            # 用训练集数据，检查训练效果
            y_train_pred = FnnModel.forward(x_train)
            print("1.************************")
            print(y_train_pred)  # 预测得到的热导率值
            loss_train = torch.mean((y_train_pred - y_train) ** 2)
            print("loss for train:", loss_train.data)  # 训练过程中的损失
            np.savetxt("train.dat", np.hstack((y_train.numpy(), y_train_pred.detach().numpy())), fmt="%.4f"
                       )

        # 测试集
        if epoch + 1 == EPOCH_COUNT:
            Autoencodermode.eval()
            with torch.no_grad():
                for batch_id, (test_data, test_labels) in enumerate(testSets, 1):
                    test_encoded, test_decoded = Autoencodermode(test_data)
                    list_tensor_test_feature.append(test_encoded)
                    list_tensor_test_labels.append(test_labels)
                    # loss_auto=criterion(decoded,data)   # 自动编解码器loss
                    # optimizer.zero_grad()
                    # loss_auto.backward()
                    # optimizer.step()
                # 测试集编码后的压缩特征和对应的标签
                x_test_feature = torch.cat(list_tensor_test_feature).reshape(-1, number_conv3)
                y_test_label = torch.cat(list_tensor_test_labels).reshape(-1, 1)
                test_numpy=x_test_feature.detach().numpy()
                print("########################")
                #print(x_numpy)      # 保存训练集和测试集压缩后的特征到excel中
                save_matrix(x_numpy,test_numpy)
                # 用测试集数据，验证测试效果
                y_test_pred = FnnModel.forward(x_test_feature)  # 测试集预测得到的结果
                print("测试集预测结果")
                print(y_test_pred)
                np.savetxt("test.dat", np.hstack((y_test_label.numpy(), y_test_pred.detach().numpy())), fmt="%.4f")

if __name__ == '__main__':
    algorithm()
    end_time = time.time()
    print("time:%d" % (end_time - start_time) + "秒")
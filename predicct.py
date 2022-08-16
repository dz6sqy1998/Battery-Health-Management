import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class FlattenLayer(torch.nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x): # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)
net=torch.load('bessmodel1.pl')['model']
#读的表就是提取特征的表
biao = pd.read_excel(r'C:\Users\一二三四的壹\Desktop\tezheng1.xlsx')


#这个意思是后边的三个特征不要，天数要放在第一列
inputx = biao.iloc[:, 1:-3]


#峰值电压时间
inputx = inputx.drop('峰值电压时间秒', axis=1).dropna()

# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# result = scaler.fit_transform(inputx)
# print(inputx)
values = inputx.values
# rl = torch.tensor(rl)
values = torch.tensor(values)

y=[]
z = []
# print(values[0])
for k in range(24):
    test=values[k].unsqueeze(0).float()
    pre = float(net(test)[0][0].data)
    y.append(pre)
    #
    # rel = float(rl[k] / 2000)
    # z.append(rel)
    # print(rel,end = ',')


    print('预测值{}'.format(int(float(net(test)[0][0].data)*2000)))

# x1 = []
# for i in range(400):
#     x1.append(i)
#
#
# plt.scatter(x1,y,c = 'r',label = 'pre')
# # # plt.scatter(z,c = 'g')
# plt.scatter(x1,z,c = 'b',label = 'true')
# plt.legend()
# plt.show()
# print(y)
# print(z)

#    print(k,pre)
#    print(rel)

# # aa = np.arange(550)
# plt.rcParams['font.sans-serif'] = [u'SimHei']
# plt.rcParams['axes.unicode_minus'] = False
# plt.plot(aa,color ='r',label = 'pre')
# plt.plot(bb,color = 'b',label = 'true')
# plt.xlabel('对应值')
# plt.ylabel('数值')
# plt.title('容量预测')
# plt.legend()
# plt.show()
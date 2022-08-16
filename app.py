from flask import Flask,render_template,request,redirect,url_for,jsonify
from forms import LoginForm
import pandas as pd
import pickle
import torch
import numpy as np
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from scipy import interpolate
from sklearn.model_selection import train_test_split
import math

import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=17, kernel_size=6, stride=1),
            nn.ReLU(), )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=17, out_channels=14, kernel_size=3, stride=1),
            nn.ReLU(), )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=14, out_channels=27, kernel_size=20, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3))

        self.fc1 = nn.Sequential(
            nn.Linear(1620, 30),
            # nn.Dropout(p=0.1)
            )
        self.fc2 = nn.Linear(30, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        out = self.fc2(x)
        return out

PATH=r"C:\Users\sqy\Desktop\验收\UI\移动项目\model_parameter.pkl"#读取质检模型
new_model = CNN().double()                          #建立新模型
new_model.load_state_dict(torch.load(PATH))   #将质检model中的参数加载到new_model中
model=pickle.load(open('savemodel_3class_4.sav','rb'))
app=Flask(__name__)
app.config['SECRET_KEY'] = '1456719640@qq.com'
app.config['JSON_AS_ASCII'] =False
feature1=float(10)
item=int(1)
data_zhijian_echarts=[]
data_guanli_echarts=[]
data_jiance_echarts=[]
battery_category=['优秀','良好','合格','较差','故障电池']
categories=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
class FlattenLayer(torch.nn.Module):
	def __init__(self):
		super(FlattenLayer, self).__init__()

	def forward(self, x):  # x shape: (batch, *, *, ...)
		return x.view(x.shape[0], -1)
@app.route('/login',methods=['GET','POST'])#在路由端设置GET请求和POST请求
def login():
	#GET设置页面输入，POST设置发送数据
	if request.method=='GET':
		return render_template("login_1.html",**locals())
	else:
		form=LoginForm(request.form)
		#格式验证是否为真
		if form.validate():
			return render_template("guanli_upload.html",**locals())
		else:
			return"邮箱或密码错误！"
@app.route('/register', methods=['GET', 'POST'])
def register():
	if request.method == 'POST':
		print(request.form.get('username'))
		print(request.form.get('password'))
		print(request.form.get('repassword'))
		return '注册成功'

	return render_template('register.html')
@app.route("/")
def index():
	result=''
	name1=''
	return render_template('index.html',**locals())


@app.route('/guanli_upload', methods=['POST'])
def upload():
	if request.method == 'POST':
		global feature1,item
		# feature1=float(request.form.get('days'))
		feature1=request.form.get('days')

		# item=int(request.form.get('id'))
		print("这是天数：",feature1)
		print("这是单体电池编号：",item)
		return render_template('guanli_upload.html',**locals())

@app.route('/guanli_tianshu', methods=['POST','GET'])
def guanli_tianshu():
	#GET设置页面输入，POST设置发送数据
	if request.method == 'POST':
		print("这是用户名：",request.form.get('username'))
		print("这是密码：" ,request.form.get('password'))
		return redirect(url_for('guanli_tianshu'))
		# print(request.form.get('repassword'))
	return render_template('guanli_tianshu.html')
@app.route('/guanli_predict', methods=['POST','GET'])
def success():
	if request.method == 'POST':
		f = request.files['file']
		# f1 = request.files['file1']

		f.save(f.filename)
		# f1.save(f1.filename)
		global item,feature1
		print("这是predict里的楼层编号：",item)
		print("这是predict里的天数变量：",feature1)
		data=pd.read_excel(f.filename,sheetname='电压数据表格1', header=None)#目前提取的是xlsx格式，但蓄电池软件生成的都是xls

#加入孙闯的原始特征提取
		# data1 = pd.read_excel(r"C:\Users\sqy\Desktop\data processing\放电数据\3f7-1.xls", sheetname='电压数据表格1', header=None)
		data2 = data.drop([1, 2, 3, 4], axis=0)
		data2
		data3 = data2.drop([0], axis=1)
		data2.set_axis([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
					   axis=0)
		data2
		data_1 = data2.iloc[:, 42:210].T
		data_1
		data_1[0] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
					 28, 29,
					 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
					 55, 56,
					 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
					 82, 83,
					 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106,
					 107, 108,
					 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128,
					 129,
					 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149,
					 150,
					 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168]

		# print(data_1)

		# In[53]:
		# 最小二乘法,求斜率w,截距b
		def fit(data_x, data_y):
			m = len(data_y)
			x_bar = np.mean(data_x)
			sum_yx = 0
			sum_x2 = 0
			sum_delta = 0
			for i in range(m):
				x = float(np.array(data_x[i]))
				y = float(np.array(data_y[i]))
				sum_yx += y * (x - x_bar)
				sum_x2 += x ** 2
			# 根据公式计算w
			w = sum_yx / (sum_x2 - m * (x_bar ** 2))

			for i in range(m):
				x = float(np.array(data_x[i]))
				y = float(np.array(data_y[i]))
				sum_delta += (y - w * x)
			b = sum_delta / m
			return w, b

		# In[54]:

		w_list = []
		b_list = []
		for p in range(1, 25):
			x = np.array(data_1[0])
			y = np.array(data_1[p])

			w, b = fit(x, y)
			w_list.append(w)
			b_list.append(b)

		w_list.insert(0, '斜率')
		b_list.insert(0, '截距')
		data3.insert(11, 'w', w_list)
		data3.insert(11, 'b', b_list)
		data3['xielv'] = w_list
		data3['jieju'] = b_list
		df = data3.loc[1:, ['jieju', 'xielv']]

		# df.reset_index(inplace=True)
		df.set_axis([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23], axis=0)
		af = df
		# df.values=float(df.values)
		# df_1 = pd.read_csv(r"C:\Users\sqy\Desktop\验收\UI\斜率截距\7-14-1.csv",engine='python')

		df_1 = pd.read_excel(f.filename, sheetname='电压数据表格1')

		df_2 = df_1.drop([0, 1, 2, 3], axis=0)
		df_3 = df_2.drop(['时间(HMS)'], axis=1)
		df_3 = df_3.iloc[:, 0:207]
		df_3.set_axis([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23], axis=0)

		# df_3.columns=['0:00:00','0:00:06','0:00:12','0:00:18','0:00:24','0:00:30','0:00:36','0:00:42','0:00:48','0:00:54','0:01:00','0:01:06','0:01:12','0:01:18','0:01:24','0:01:30','0:01:36','0:01:42','0:01:48','0:01:54','0:02:00','0:02:06','0:02:12','0:02:18','0:02:24','0:02:30','0:02:36','0:02:42','0:02:48','0:02:54','0:03:00','0:04:00','0:05:00','0:06:00','0:07:00','0:08:00','0:09:00','0:10:00','0:11:00','0:12:00','0:13:00','0:14:00','0:15:00','0:16:00','0:17:00','0:18:00','0:19:00','0:20:00','0:21:00','0:22:00','0:23:00','0:24:00','0:25:00','0:26:00','0:27:00','0:28:00','0:29:00','0:30:00','0:31:00','0:32:00','0:33:00','0:34:00','0:35:00','0:36:00','0:37:00','0:38:00','0:39:00','0:40:00','0:41:00','0:42:00','0:43:00','0:44:00','0:45:00','0:46:00','0:47:00','0:48:00','0:49:00','0:50:00','0:51:00','0:52:00','0:53:00','0:54:00','0:55:00','0:56:00','0:57:00','0:58:00','0:59:00','1:00:00','1:01:00','1:02:00','1:03:00','1:04:00','1:05:00','1:06:00','1:07:00','1:08:00','1:09:00','1:10:00','1:11:00','1:12:00','1:13:00','1:14:00','1:15:00','1:16:00','1:17:00','1:18:00','1:19:00','1:20:00','1:21:00','1:22:00','1:23:00','1:24:00','1:25:00','1:26:00','1:27:00','1:28:00','1:29:00','1:30:00','1:31:00','1:32:00','1:33:00','1:34:00','1:35:00','1:36:00','1:37:00','1:38:00','1:39:00','1:40:00','1:41:00','1:42:00','1:43:00','1:44:00','1:45:00','1:46:00','1:47:00','1:48:00','1:49:00','1:50:00','1:51:00','1:52:00','1:53:00','1:54:00','1:55:00','1:56:00','1:57:00','1:58:00','1:59:00','2:00:00','2:01:00','2:02:00','2:03:00','2:04:00','2:05:00','2:06:00','2:07:00','2:08:00','2:09:00','2:10:00','2:11:00','2:12:00','2:13:00','2:14:00','2:15:00','2:16:00','2:17:00','2:18:00','2:19:00','2:20:00','2:21:00','2:22:00','2:23:00','2:24:00','2:25:00','2:26:00','2:27:00','2:28:00','2:29:00','2:30:00','2:31:00','2:32:00','2:33:00','2:34:00','2:35:00','2:36:00','2:37:00','2:38:00','2:39:00','2:40:00','2:41:00','2:42:00','2:43:00','2:44:00','2:45:00','2:46:00','2:47:00','2:48:00','2:49:00','2:50:00','2:51:00','2:52:00','2:53:00','2:54:00','2:55:00','2:56:00','2:57:00','2:58:00','2:59:00','3:00:00','3:00:10']
		df_3.columns = ['0:00:00', '0:00:06', '0:00:12', '0:00:18', '0:00:24', '0:00:30', '0:00:36', '0:00:42',
						'0:00:48', '0:00:54', '0:01:00', '0:01:06', '0:01:12', '0:01:18', '0:01:24', '0:01:30',
						'0:01:36', '0:01:42', '0:01:48', '0:01:54', '0:02:00', '0:02:06', '0:02:12', '0:02:18',
						'0:02:24', '0:02:30', '0:02:36', '0:02:42', '0:02:48', '0:02:54', '0:03:00', '0:04:00',
						'0:05:00', '0:06:00', '0:07:00', '0:08:00', '0:09:00', '0:10:00', '0:11:00', '0:12:00',
						'0:13:00', '0:14:00', '0:15:00', '0:16:00', '0:17:00', '0:18:00', '0:19:00', '0:20:00',
						'0:21:00', '0:22:00', '0:23:00', '0:24:00', '0:25:00', '0:26:00', '0:27:00', '0:28:00',
						'0:29:00', '0:30:00', '0:31:00', '0:32:00', '0:33:00', '0:34:00', '0:35:00', '0:36:00',
						'0:37:00', '0:38:00', '0:39:00', '0:40:00', '0:41:00', '0:42:00', '0:43:00', '0:44:00',
						'0:45:00', '0:46:00', '0:47:00', '0:48:00', '0:49:00', '0:50:00', '0:51:00', '0:52:00',
						'0:53:00', '0:54:00', '0:55:00', '0:56:00', '0:57:00', '0:58:00', '0:59:00', '1:00:00',
						'1:01:00', '1:02:00', '1:03:00', '1:04:00', '1:05:00', '1:06:00', '1:07:00', '1:08:00',
						'1:09:00', '1:10:00', '1:11:00', '1:12:00', '1:13:00', '1:14:00', '1:15:00', '1:16:00',
						'1:17:00', '1:18:00', '1:19:00', '1:20:00', '1:21:00', '1:22:00', '1:23:00', '1:24:00',
						'1:25:00', '1:26:00', '1:27:00', '1:28:00', '1:29:00', '1:30:00', '1:31:00', '1:32:00',
						'1:33:00', '1:34:00', '1:35:00', '1:36:00', '1:37:00', '1:38:00', '1:39:00', '1:40:00',
						'1:41:00', '1:42:00', '1:43:00', '1:44:00', '1:45:00', '1:46:00', '1:47:00', '1:48:00',
						'1:49:00', '1:50:00', '1:51:00', '1:52:00', '1:53:00', '1:54:00', '1:55:00', '1:56:00',
						'1:57:00', '1:58:00', '1:59:00', '2:00:00', '2:01:00', '2:02:00', '2:03:00', '2:04:00',
						'2:05:00', '2:06:00', '2:07:00', '2:08:00', '2:09:00', '2:10:00', '2:11:00', '2:12:00',
						'2:13:00', '2:14:00', '2:15:00', '2:16:00', '2:17:00', '2:18:00', '2:19:00', '2:20:00',
						'2:21:00', '2:22:00', '2:23:00', '2:24:00', '2:25:00', '2:26:00', '2:27:00', '2:28:00',
						'2:29:00', '2:30:00', '2:31:00', '2:32:00', '2:33:00', '2:34:00', '2:35:00', '2:36:00',
						'2:37:00', '2:38:00', '2:39:00', '2:40:00', '2:41:00', '2:42:00', '2:43:00', '2:44:00',
						'2:45:00', '2:46:00', '2:47:00', '2:48:00', '2:49:00', '2:50:00', '2:51:00', '2:52:00',
						'2:53:00', '2:54:00', '2:55:00', '2:56:00', '2:57:00', '2:58:00', '2:59:00']

		# df_3
		# # lst=list(df_3.columns)
		# # lst

		df3_ = df_3.iloc[:, 0:31]
		df3_['gudivalue'] = df3_.min(axis=1)

		df3_['lieming'] = df3_.idxmin(axis=1)
		df3__ = df_3.iloc[:, 32:68]
		df3_['fengzhi'] = df3__.max(axis=1)
		# df3_['fengzhi']
		df3_['fzsj'] = df3__.idxmax(axis=1)
		# df3_['fengzhi']
		# df3_['lieming']
		# df3_['fzsj']

		df2 = df3_['fzsj'].str.split(':', expand=True).astype(int)

		# df4=df3_['lieming'].str.split(':',expand=True).astype(int)
		df4 = df3_['lieming'].str.split(':', expand=True).astype(int)

		# df4
		df3_['gudisj'] = df4[1] * 60 + df4[2]
		# df3_
		df3_['fengzhisj'] = df2[1] * 60
		df3_['jiezhi'] = df_3.iloc[:, 206]
		df3_['fuchong'] = df_3.iloc[:, 0]
		# df3_
		tezheng = df3_.loc[:, ['gudivalue', 'fengzhi', 'gudisj', 'fengzhisj', 'fuchong', 'jiezhi']]
		tezheng1 = pd.concat([tezheng, af], axis=1)
		tezheng1.set_axis(['谷底电压', '峰值电压', '谷底电压时间秒', '峰值电压时间秒', '浮充电压', '截止电压', '截距', '斜率'], axis=1)



		lst = feature1
		lst1 = []
		lst1 = [float(n) for n in lst.split()]

		Day=pd.DataFrame(np.array(lst1),columns=['天数'])
		new_data=pd.concat([Day,tezheng1[0:24]],axis=1)
		new_data['斜率']=new_data['斜率'].astype(float)
		new_data['截距']=new_data['截距'].astype(float)

		# print(new_data)

		# column=['天数','谷底电压值','峰值电压值','谷底电压时间（秒）','峰值电压时间（秒）','浮充电压值','截止电压','截距','斜率']
		# new_data=pd.concat([Day,data[:24]],axis=1)
		result1 = model.predict(new_data)

		#这个留白是为了加入孙闯那个处理原始特征，得到新的数据data1
		# feature2 = data.loc[item-1].values.flatten().tolist()#这个将来要修改,目前还不是自动提取
		# feature2.insert(0, feature1)
		# result1 = model.predict([feature2])
		# u1_0 = result1[0]
		result3 = []
		result4 = []
		result5 = []

		for i in range(24):
			u1_0 = result1[i]

			'''u2_0为电池的soh值的分类置信度'''
			u1_0_array = np.array(u1_0)
			u2_0=np.array([0.01508184, 0.01370481, 0.97121335])#尝试三分类
			# u2_0 = np.array([4.79029204e-02, 1.30396045e-01, 8.50501691e-02, 3.17260895e-04,7.36333604e-01])
			R = np.append(u1_0_array, u2_0)
			# R = R.reshape(2, 5)
			R = R.reshape(2, 3)
			A = np.array([0.97,0.03])
			R.shape

			'''法一'''

			def min_max_operator(A, R):
				'''
				利用最值算子合成矩阵
				:param A:评判因素权向量 A = (a1,a2 ,L,an )
				:param R:模糊关系矩阵 R
				:return:
				'''
				B = np.zeros((1, R.shape[1]))
				for column in range(0, R.shape[1]):
					list = []
					for row in range(0, R.shape[0]):
						list.append(min(A[row], R[row, column]))
					B[0, column] = max(list)
				return B

			y_prob = min_max_operator(A, R)
			# print(y_prob)
			y_pred = [list(x).index(max(x)) for x in y_prob]
			# y_pred

			# 2022.7.6按照移动修改5分类的评语集
			# def evaluate():
			# 	if y_pred[0] == 0:
			# 		return "综合健康状态为：优秀"
			# 	elif y_pred[0] == 1:
			# 		return "综合健康状态为：良好"
			# 	elif y_pred[0] == 2:
			# 		return "综合健康状态为：合格"
			# 	elif y_pred[0] == 3:
			# 		return "综合健康状态为：较差"
			# 	else:
			# 		return "综合健康状态为：故障电池"
			def evaluate():
				if y_pred[0] == 0:
					return "综合健康状态为：优秀"
				elif y_pred[0] == 1:
					return "综合健康状态为：合格"
				else:
					return "综合健康状态为：故障电池"
			result2=evaluate()
			result5=y_pred[0]
			result3.append(result2)

		for o, t in enumerate(result3):
			result4.append((o+1, t))
		result = result4
		name1 = f.filename
		global data_guanli_echarts
		data_guanli_echarts=result3
		return render_template('guanli_predict.html', **locals())
	return redirect(url_for('success'))#重定向，重新输入
@app.route('/index_1', methods=['POST', 'GET'])
def index_1():
	return render_template('index_1.html',**locals())
@app.route('/zhijian', methods=['POST', 'GET'])
def zhijian():
	return render_template('zhijian.html',**locals())
@app.route('/zhijian_upload', methods=['POST'])
def zhijian_upload():
	if request.method == 'POST':
		return render_template('zhijian_upload.html',**locals())
@app.route('/zhijian_predict', methods=['POST','GET'])
def zhijian_predict():
	if request.method == 'POST':
		f = request.files['file']
		# f1 = request.files['file1']

		f.save(f.filename)
		# f1.save(f1.filename)
		lst = []
		for i in range(24):
			lst.append(1)
		zhijian_1 = pd.read_excel(f.filename,
								  sheetname='电压数据表格1')
		zhijian_2 = zhijian_1.drop([0, 1, 2, 3], axis=0)
		zhijian_3 = zhijian_2.drop(['时间(HMS)'], axis=1)
		zhijian_4 = zhijian_3.iloc[:, :208]  # 将输入的表格格式化为训练模型中的训练数据（，208）格式
		zhijian_4['label'] = lst  # 增加一列假标签，为了满足数据格式
		train_list = zhijian_4.values[:, :-1]  # 208
		train_list = np.expand_dims(train_list, axis=1)
		y = zhijian_4.values[:, -1]
		x_test = torch.tensor(train_list, dtype=torch.double)
		y_test = torch.tensor(y, dtype=torch.double)
		test_dataset = TensorDataset(x_test, y_test)
		batch_size = 32
		test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
												  batch_size=batch_size,
												  shuffle=True)
		for (data, target) in test_loader:
			y_prob = new_model(data)
		# print("真实值为：", target)
		pred = torch.max(y_prob.data, 1)[1]  # torch.max(x,dim) 返回最大值和索引
		result_2=pred
		name3=f.filename
		global data_zhijian_echarts
		data_zhijian_echarts=result_2.tolist()
		# print("预测值为：", pred)
		return render_template('zhijian_predict.html', **locals())
	return redirect(url_for('zhijian'))#重定向，重新输入

@app.route('/guanli', methods=['POST','GET'])
def guanli():
	return render_template("guanli.html",**locals())

@app.route('/jiance', methods=['POST','GET'])
def jiance():
	return render_template("jiance.html",**locals())
@app.route('/jiance_tianshu', methods=['POST','GET'])
def jiance_tianshu():
	#GET设置页面输入，POST设置发送数据
	if request.method == 'POST':
		# print("这是用户名：",request.form.get('username'))#可以输出登录界面得到的用户名和密码
		# print("这是密码：" ,request.form.get('password'))
		return redirect(url_for('jiance_tianshu'))
	return render_template('jiance_tianshu.html')
@app.route('/jiance_upload', methods=['POST'])
def jiance_upload():
	if request.method == 'POST':
		global feature1, item
		# feature1=float(request.form.get('days'))
		feature1 = request.form.get('days')

		# item = int(request.form.get('id'))
		print("这是天数：", feature1)
		print("这是单体电池编号：", item)
		return render_template('jiance_upload.html',**locals())
@app.route('/jiance_predict',methods=['POST','GET'])
def jiance_predict():
	if request.method == 'POST':
		f = request.files['file']
		f.save(f.filename)
		global item,feature1
		lst = feature1
		lst1 = []
		lst1 = [float(n) for n in lst.split()]

		Day=pd.DataFrame(np.array(lst1),columns=['天数'])
	# ##实现上传状态检测文件
	# 	jiance_1 = pd.read_excel(f.filename, sheetname='电压数据表格1')
	# 	jiance_2 = jiance_1.drop([0, 1, 2, 3], axis=0)
	# 	jiance_3 = jiance_2.drop(['时间(HMS)'], axis=1)
	# 	jiance_3 = jiance_3.iloc[:, 0:43]
	# 	jiance_3.set_axis([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
	# 					  axis=0)
	# 	jiance_3.columns = ['0:00:00', '0:00:06', '0:00:12', '0:00:18', '0:00:24', '0:00:30', '0:00:36', '0:00:42',
	# 						'0:00:48', '0:00:54', '0:01:00', '0:01:06', '0:01:12', '0:01:18', '0:01:24', '0:01:30',
	# 						'0:01:36', '0:01:42', '0:01:48', '0:01:54', '0:02:00', '0:02:06', '0:02:12', '0:02:18',
	# 						'0:02:24', '0:02:30', '0:02:36', '0:02:42', '0:02:48', '0:02:54', '0:03:00', '0:04:00',
	# 						'0:05:00', '0:06:00', '0:07:00', '0:08:00', '0:09:00', '0:10:00', '0:11:00', '0:12:00',
	# 						'0:13:00', '0:14:00', '0:15:00']
	# 	jiance3_ = jiance_3.iloc[:, 0:31]
	# 	jiance3_['gudivalue'] = jiance3_.min(axis=1)
	#
	# 	jiance3_['lieming'] = jiance3_.idxmin(axis=1)
	# 	jiance3__ = jiance_3.iloc[:, 32:42]
	# 	jiance3_['fengzhi'] = jiance3__.max(axis=1)
	# 	jiance4 = jiance3_['lieming'].str.split(':', expand=True).astype(int)
	# 	jiance3_['gudisj'] = jiance4[1] * 60 + jiance4[2]
	# 	jiance3_['fuchong'] = jiance_3.iloc[:, 0]
	# 	zhuangtai = jiance3_.loc[:, ['gudivalue', 'fengzhi', 'gudisj', 'fuchong']]
	# 	zhuangtai1 = pd.concat([Day, zhuangtai[0:24]], axis=1)

		jiance_1 = pd.read_excel(f.filename, sheetname='电压数据表格1')
		jiance_2 = jiance_1.drop([0, 1, 2, 3], axis=0)
		jiance_3 = jiance_2.drop(['时间(HMS)'], axis=1)
		jiance_3 = jiance_3.iloc[:, 0:42]
		jiance_3.set_axis([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
						  axis=0)

		# df_3.columns=['0:00:00','0:00:06','0:00:12','0:00:18','0:00:24','0:00:30','0:00:36','0:00:42','0:00:48','0:00:54','0:01:00','0:01:06','0:01:12','0:01:18','0:01:24','0:01:30','0:01:36','0:01:42','0:01:48','0:01:54','0:02:00','0:02:06','0:02:12','0:02:18','0:02:24','0:02:30','0:02:36','0:02:42','0:02:48','0:02:54','0:03:00','0:04:00','0:05:00','0:06:00','0:07:00','0:08:00','0:09:00','0:10:00','0:11:00','0:12:00','0:13:00','0:14:00','0:15:00','0:16:00','0:17:00','0:18:00','0:19:00','0:20:00','0:21:00','0:22:00','0:23:00','0:24:00','0:25:00','0:26:00','0:27:00','0:28:00','0:29:00','0:30:00','0:31:00','0:32:00','0:33:00','0:34:00','0:35:00','0:36:00','0:37:00','0:38:00','0:39:00','0:40:00','0:41:00','0:42:00','0:43:00','0:44:00','0:45:00','0:46:00','0:47:00','0:48:00','0:49:00','0:50:00','0:51:00','0:52:00','0:53:00','0:54:00','0:55:00','0:56:00','0:57:00','0:58:00','0:59:00','1:00:00','1:01:00','1:02:00','1:03:00','1:04:00','1:05:00','1:06:00','1:07:00','1:08:00','1:09:00','1:10:00','1:11:00','1:12:00','1:13:00','1:14:00','1:15:00','1:16:00','1:17:00','1:18:00','1:19:00','1:20:00','1:21:00','1:22:00','1:23:00','1:24:00','1:25:00','1:26:00','1:27:00','1:28:00','1:29:00','1:30:00','1:31:00','1:32:00','1:33:00','1:34:00','1:35:00','1:36:00','1:37:00','1:38:00','1:39:00','1:40:00','1:41:00','1:42:00','1:43:00','1:44:00','1:45:00','1:46:00','1:47:00','1:48:00','1:49:00','1:50:00','1:51:00','1:52:00','1:53:00','1:54:00','1:55:00','1:56:00','1:57:00','1:58:00','1:59:00','2:00:00','2:01:00','2:02:00','2:03:00','2:04:00','2:05:00','2:06:00','2:07:00','2:08:00','2:09:00','2:10:00','2:11:00','2:12:00','2:13:00','2:14:00','2:15:00','2:16:00','2:17:00','2:18:00','2:19:00','2:20:00','2:21:00','2:22:00','2:23:00','2:24:00','2:25:00','2:26:00','2:27:00','2:28:00','2:29:00','2:30:00','2:31:00','2:32:00','2:33:00','2:34:00','2:35:00','2:36:00','2:37:00','2:38:00','2:39:00','2:40:00','2:41:00','2:42:00','2:43:00','2:44:00','2:45:00','2:46:00','2:47:00','2:48:00','2:49:00','2:50:00','2:51:00','2:52:00','2:53:00','2:54:00','2:55:00','2:56:00','2:57:00','2:58:00','2:59:00','3:00:00','3:00:10']
		jiance_3.columns = ['0:00:00', '0:00:06', '0:00:12', '0:00:18', '0:00:24', '0:00:30', '0:00:36', '0:00:42',
							'0:00:48', '0:00:54', '0:01:00', '0:01:06', '0:01:12', '0:01:18', '0:01:24', '0:01:30',
							'0:01:36', '0:01:42', '0:01:48', '0:01:54', '0:02:00', '0:02:06', '0:02:12', '0:02:18',
							'0:02:24', '0:02:30', '0:02:36', '0:02:42', '0:02:48', '0:02:54', '0:03:00', '0:04:00',
							'0:05:00', '0:06:00', '0:07:00', '0:08:00', '0:09:00', '0:10:00', '0:11:00', '0:12:00',
							'0:13:00', '0:14:00']
		jiance3_ = jiance_3.iloc[:, 0:31]
		jiance3_['gudivalue'] = jiance3_.min(axis=1)

		jiance3_['lieming'] = jiance3_.idxmin(axis=1)
		jiance3__ = jiance_3.iloc[:, 32:42]
		jiance3_['fzsj'] = jiance3__.idxmax(axis=1)
		jiance5 = jiance3_['fzsj'].str.split(':', expand=True).astype(int)
		jiance3_['fengzhi'] = jiance3__.max(axis=1)
		jiance4 = jiance3_['lieming'].str.split(':', expand=True).astype(int)
		jiance3_['gudisj'] = jiance4[1] * 60 + jiance4[2]
		jiance3_['fengzhisj'] = jiance5[1] * 60

		jiance3_['fuchong'] = jiance_3.iloc[:, 0]
		# zhuangtai = jiance3_.loc[:, ['gudivalue', 'fengzhi', 'gudisj','fuchong']]
		zhuangtai = jiance3_.loc[:, ['gudivalue', 'fengzhi', 'gudisj', 'fengzhisj', 'fuchong']]
		zhuangtai1 = pd.concat([Day, zhuangtai[0:24]], axis=1)

		# net = torch.load('bessmodel2.pl')['model']
		# values = zhuangtai1.values
		# values = torch.tensor(values)
		#
		# y = []
		# z = []
		# for k in range(24):
		# 	test = values[k].unsqueeze(0).float()
		# 	pre = int(float((net(test)[0][0].data) * 2000))
		# 	y.append(pre)
		model = pickle.load(open('rf2.sav', 'rb'))
		y_prob = model.predict(zhuangtai1)
		y=[int(x) for x in y_prob]
		result_1=y
		global data_jiance_echarts
		data_jiance_echarts=y
		name2 = f.filename
		return render_template('jiance_predict.html', **locals())
	return redirect(url_for('jiance_predict'))  # 重定向，重新输入
#连接echarts实现可视化
#输入数据

@app.route('/zhijian_echarts', methods=["GET","POST"]) #echarts 名字可以改为任意，但一定要与HTML文件中一至
def echarts():
	if request.method == 'GET':
		global data_zhijian_echarts
		value_1=data_zhijian_echarts.count(1)
	# print(data_echarts)
	return jsonify(categories = categories,data =value_1)
@app.route('/guanli_echarts', methods=["GET","POST"]) #echarts 名字可以改为任意，但一定要与HTML文件中一至
def guanli_echarts():
	if request.method == 'GET':
		global data_guanli_echarts,battery_category
		value_0=[]
		value_1=[]
		value_2=[]
		value_3=[]
		value_4=[]
		# index_0 = [i + 1 for i, val in enumerate(data_guanli_echarts) if val == "综合健康状态为：优秀"]
		# index_1 = [i + 1 for i, val in enumerate(data_guanli_echarts) if val == "综合健康状态为：良好"]
		# index_2 = [i + 1 for i, val in enumerate(data_guanli_echarts) if val == "综合健康状态为：合格"]
		# index_3 = [i + 1 for i, val in enumerate(data_guanli_echarts) if val == "综合健康状态为：较差"]
		# index_4 = [i + 1 for i, val in enumerate(data_guanli_echarts) if val == "综合健康状态为：故障电池"]

		index_0 = [i + 1 for i, val in enumerate(data_guanli_echarts) if val == "综合健康状态为：优秀"]
		index_1 = [i + 1 for i, val in enumerate(data_guanli_echarts) if val == "综合健康状态为：合格"]
		index_2 = [i + 1 for i, val in enumerate(data_guanli_echarts) if val == "综合健康状态为：故障电池"]




		for j in range(len(index_0)):
			value_0.append({'name': str(index_0[j]), 'value': 1})
		for j1 in range(len(index_1)):
			value_1.append({'name': str(index_1[j1]), 'value': 1})
		for j2 in range(len(index_2)):
			value_2.append({'name': str(index_2[j2]), 'value': 1})
		# for j3 in range(len(index_3)):
		# 	value_3.append({'name': str(index_3[j3]), 'value': 1})
		# for j4 in range(len(index_4)):
		# 	value_4.append({'name': str(index_4[j4]), 'value': 1})

	# return jsonify(battery_category = battery_category,value_0=value_0,value_1=value_1,value_2=value_2,value_3=value_3,value_4=value_4)
	return jsonify(battery_category = battery_category,value_0=value_0,value_1=value_1,value_2=value_2)

@app.route('/jiance_echarts', methods=["GET","POST"]) #echarts 名字可以改为任意，但一定要与HTML文件中一至
def jiance_echarts():
	if request.method == 'GET':
		global data_jiance_echarts
		global categories
	return jsonify(categories = categories,data_jiance_echarts =data_jiance_echarts)
@app.route('/zhijian_plot', methods=["GET",'POST']) #echarts 名字可以改为任意，但一定要与HTML文件中一至
def plot():
	if request.method == 'GET':
		return render_template('zhijian_plot.html',**locals())
@app.route('/guanli_plot', methods=["GET",'POST']) #echarts 名字可以改为任意，但一定要与HTML文件中一至
def guanli_plot():
	if request.method == 'GET':
		return render_template('guanli_plot.html',**locals())
@app.route('/jiance_plot', methods=["GET",'POST']) #echarts 名字可以改为任意，但一定要与HTML文件中一至
def jiance_plot():
	if request.method == 'GET':
		return render_template('jiance_plot.html',**locals())
if __name__ == '__main__':
    app.run(debug=True,port=100)

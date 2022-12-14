#加入孙闯的原始特征提取
import pandas as pd
import numpy as np
data = pd.read_excel(r"C:\Users\sqy\Desktop\验收\UI\移动项目\原始表格数据\八楼\8-1-1.xls", sheetname='电压数据表格1', header=None)#修改为自己的路径
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
df_1 = pd.read_excel(r"C:\Users\sqy\Desktop\验收\UI\移动项目\原始表格数据\八楼\8-1-1.xls", sheetname='电压数据表格1')#再次提取原始表，有表头,修改为自己的路径


# df_1 = pd.read_excel(f.filename, sheetname='电压数据表格1')

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
tezheng1.to_excel(r"C:\Users\sqy\Desktop\data processing\放电数据\new_data1.xls")#得到新的特征

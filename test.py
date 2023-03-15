# import json
#
# with open('./value_chart.txt', 'r') as f:
#     lines = f.readlines()  #["{"079835:1, "003075:1",...}"]
#     data = ''
#     for line in lines: #line:{"079835:1, "003075:1",...}
#         data += line
#     chartcode_value = json.loads(data) #243  将str类型的数据转换为dict类型
# print(type(chartcode_value))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdate
import matplotlib.ticker as ticker

'''收益率画图'''
# cnn=[0.0, 3.2800000000000002, 5.74, 7.175, 7.585, 15.58, 17.22, 23.985, 35.260000000000005, 38.129999999999995, 52.275000000000006, 60.68, 38.129999999999995, 28.904999999999998, 33.005, 33.415, 43.665, 41.615, 27.88, 33.825]
# lstm=[0.0, 3.2800000000000002, 5.74, 7.175, 7.585, 7.585, 7.585, 13.69, 23.865, 26.455000000000002, 26.455000000000002, 26.455000000000002, 8.855, 1.6549999999999998, 4.855, 4.855, 12.855, 11.254999999999999, 0.5349999999999999, 5.175]
# dqn3=[0.0, 0.0, 0.0, 1.225, 1.6150000000000002, 9.22, 10.780000000000001, 17.215, 27.939999999999998, 28.71, 28.71, 35.475, 16.775000000000002, 9.125, 12.525, 12.615000000000002,
#       14.865, 13.264999999999999, 2.545, 7.33]
# dqn4=[0.0, 0.0, 1.5, 1.5, 1.5, 6.375, 7.8549999999999995, 13.96, 15.885, 17.705000000000002, 26.674999999999997, 26.674999999999997, 26.674999999999997, 26.674999999999997, 28.075, 28.215, 31.715, 31.715, 28.03, 29.625]
# x1=['2019-03-04', '2019-03-05', '2019-03-06', '2019-03-07',
#                '2019-03-08',   '2019-03-11',
#                '2019-03-12', '2019-03-13', '2019-03-14', '2019-03-15',
#                 '2019-03-18', '2019-03-19',
#                '2019-03-20', '2019-03-21', '2019-03-22', '2019-03-25', '2019-03-26', '2019-03-27',
#                '2019-03-28', '2019-03-29']
# def draw():
#     fig= plt.figure(figsize=(15, 5))
#     ax = plt.gca()  # 表明设置图片的各个轴，plt.gcf()表示图片本身
#     #ax.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))  # 横坐标标签显示的日期格式
#     #plt.plot(twophotos, label='', color='red', linestyle=':')
#     #plt.xticks(pd.date_range('2019-3-4', '2019-3-29', freq='20d'))  # 横坐标日期范围及间隔
#     #x1=pd.date_range('2019-3-4', '2019-3-29', freq='D')
#
#     plt.xlabel('日期', fontproperties="simhei")
#     plt.ylabel('收益率', fontproperties="simhei")
#     plt.plot(x1,dqn3, label='model 1', color='red')
#     plt.plot(x1,dqn4, label='model 2', color='blue')
#     plt.plot(x1,cnn, label='cnn', color='yellow')
#     plt.plot(x1,lstm, label='lstm', color='green')
#     plt.legend(loc='upper left')
#     '''
#     plt.twinx()  # 将ax1的x轴也分配给ax2使用
#     plt.ylabel('股价收益率', fontproperties="simhei")
#     plt.plot(x1,truephotos, label='price',color='green')
#     '''
#     ax.xaxis.set_major_locator(ticker.MultipleLocator(base=2))  # 解决X轴密集问题
#     #plt.legend(loc='best')
#     #plt.savefig(save_name1)
#     plt.show()
#     plt.close()
# draw()


'''指标画图柱状图'''
# labels=['模型一','模型二','lstm','cnn']
# y60_data=[17950,32500,62500,74800]
# y520_data=[73300,296250,51750,338250]
# y860_data=[3350,45100,34800,27750]
# y2240_data=[-36150,16550,-48100,-102600]
# x_width = range(0, len(labels))
# x2_width = [i + 0.2 for i in x_width]
# x3_width = [i + 0.2 for i in x2_width]
# x4_width = [i + 0.2 for i in x3_width]
# fig= plt.figure()
#
# plt.rcParams['font.sans-serif']=['SimHei']
# plt.rcParams['axes.unicode_minus']=False  #防止乱码
#
# plt.ylabel("最终收益",fontproperties="simhei")
# # plt.xlabel("年份")
# # plt.ylabel("销量")
# plt.xticks([0.3,1.3,2.3,3.3], labels = ['模型一','模型二','lstm','cnn'],fontproperties="simhei")
#
# plt.bar(x_width, y60_data, lw=0.5, fc="r", width=0.2, label="000060")
# plt.bar(x2_width, y520_data, lw=0.5, fc="b", width=0.2, label="000520")
# plt.bar(x3_width, y860_data, lw=0.5, fc="y", width=0.2, label="000860")
# plt.bar(x4_width, y2240_data, lw=0.5, fc="g", width=0.2, label="002240")
#
# plt.legend(loc='upper left') #生成标签
# plt.show()



'''最大回撤率'''
def MaxDrawdown(return_list):
    return_list1=np.maximum.accumulate(return_list)
    #print(return_list1)
    res=np.array([])
    for i in return_list1:
        if i==0:
            res=np.append(res,[1])
        else:
            res=np.append(res,[i])
    #print(res)
    i = np.argmax((np.maximum.accumulate(return_list) - return_list) / np.maximum.accumulate(res))  # 结束位置

   # print(i)
    if i == 0:
        return 0
    #print((np.maximum.accumulate(return_list) - return_list) / np.maximum.accumulate(res))
    j = np.argmax(return_list[:i])  # 开始位置
    #print(j)
    return (return_list[j] - return_list[i]) / (return_list[j])



return_list = \
[0.0, 0.0, 3.7549600000000094, 5.3713999999999995, 5.3713999999999995, 5.3713999999999995, 5.2706, 9.100999999999999, 8.6978, 8.6978, 8.6978, 8.6978, 8.6978, 8.6978, 8.6978, 9.4328, 9.4328, 9.4328, 9.4328, 9.4328, 10.930919999999995, 10.006359999999987, 11.970919999999996, 14.102919999999997, 21.070919999999997, 21.694919999999996, 21.694919999999996, 21.694919999999996, 21.694919999999996, 21.694919999999996, 21.694919999999996, 21.694919999999996, 21.694919999999996, 21.694919999999996, 15.699919999999995, 15.699919999999995, 15.699919999999995, 8.279919999999995, 8.279919999999995, 8.279919999999995, 8.279919999999995, 8.279919999999995, 8.279919999999995, 8.170519999999994, 8.170519999999994, 8.170519999999994, 8.170519999999994, 9.541849, 11.456349, 11.456349, 9.547318999999995, 9.547318999999995, 9.547318999999995, 8.940670999999996, 8.940670999999996, 12.130031000000004, 12.130031000000004, 12.130031000000004, 12.130031000000004, 12.130031000000004, 17.852531000000006, 14.749301, 11.641166000000016, 11.641166000000016, 11.641166000000016, 11.641166000000016, 12.469166000000016, 12.24284600000002, 11.921582000000006, 13.573166000000015, 14.46188600000001, 14.732366000000017, 14.732366000000017, 14.732366000000017, 14.732366000000017, 14.732366000000017, 14.732366000000017, 14.732366000000017, 13.554126000000025, 12.098126000000024, 10.138126000000023, 6.498126000000025, 5.154126000000024, 9.24660600000003, 9.24660600000003, 10.148481000000029, 12.54053100000003, 12.54053100000003, 12.54053100000003, 12.54053100000003, 12.54053100000003, 12.317355000000028, 12.317355000000028, 18.93367900000003, 18.93367900000003, 18.93367900000003, 18.93367900000003, 18.93367900000003, 18.93367900000003, 18.93367900000003, 18.93367900000003, 18.93367900000003, 18.93367900000003, 18.93367900000003, 18.93367900000003, 18.93367900000003, 18.93367900000003, 18.76596900000004, 16.799434000000055, 16.33146900000004, 16.33146900000004, 16.33146900000004, 16.33146900000004, 16.33146900000004, 16.89861900000004, 19.750899000000047, 20.206899000000046, 17.755899000000046, 17.755899000000046, 17.755899000000046, 17.755899000000046, 17.755899000000046, 17.755899000000046, 17.755899000000046, 17.755899000000046, 17.755899000000046, 17.306674000000047, 17.306674000000047, 17.306674000000047, 17.306674000000047, 17.306674000000047, 17.306674000000047, 17.306674000000047, 15.598574000000045, 14.292172000000045, 10.827674000000046, 10.827674000000046, 10.827674000000046, 10.827674000000046, 10.827674000000046, 10.827674000000046, 14.860274000000045, 15.108340000000037, 12.59224200000004, 12.59224200000004, 12.59224200000004, 12.59224200000004, 12.59224200000004, 12.59224200000004, 12.59224200000004, 12.59224200000004, 12.59224200000004, 12.59224200000004, 12.59224200000004, 12.59224200000004, 12.59224200000004, 12.59224200000004, 17.00875400000005, 18.063634000000032, 13.443442000000038, 14.41624200000004, 13.49208200000003, 11.61944200000004, 16.11803400000003, 14.721458000000053, 14.721458000000053, 14.721458000000053, 14.659349000000068, 12.60914900000007, 13.096373000000044, 14.960849000000067, 14.59904900000007, 14.056349000000068, 11.222249000000069, 12.247349000000069, 14.588798000000045, 14.588798000000045, 14.588798000000045, 14.588798000000045, 20.498198000000045, 18.146498000000044, 18.146498000000044, 18.39835400000005, 19.41350000000005, 20.98700600000005, 20.98700600000005, 20.98700600000005, 18.707685000000033, 18.707685000000033, 18.707685000000033, 18.707685000000033, 18.707685000000033, 18.707685000000033, 18.707685000000033, 18.707685000000033, 18.707685000000033, 18.707685000000033, 18.707685000000033, 16.039185000000032, 17.029495000000043, 17.029495000000043, 17.029495000000043, 17.029495000000043, 17.029495000000043, 17.029495000000043, 15.240286000000033, 13.026610000000055, 14.523289000000036, 14.265385000000034, 14.265385000000034, 14.265385000000034, 14.265385000000034, 14.265385000000034, 14.265385000000034, 14.265385000000034, 14.265385000000034, 14.265385000000034, 14.265385000000034, 14.265385000000034, 14.265385000000034, 14.265385000000034, 14.265385000000034, 14.265385000000034, 14.265385000000034, 14.265385000000034, 14.265385000000034, 19.497871000000043, 19.79185700000003, 19.79185700000003, 19.79185700000003, 19.79185700000003, 19.79185700000003, 19.79185700000003, 19.79185700000003, 19.79185700000003, 19.79185700000003, 19.79185700000003, 19.79185700000003, 19.79185700000003, 19.79185700000003, 19.79185700000003]

mdd = MaxDrawdown(return_list)
print(mdd)


'''年化收益率=投资内收益/本金/投资天数×365×100%'''
res=0
for i in return_list:
    res+=i*10000
print(res/1000000/30*365)

'''模型'''
# import tensorflow.contrib.slim as slim
# import tensorflow as tf
#
#
# def normalized_columns_initializer(std=1.0):
#     def _initializer(shape, dtype=None, partition_info=None):
#         out = np.random.randn(*shape).astype(np.float32)
#         out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
#         return tf.constant(out)
#
#
# inImage = []
# for i in range(3):
#     inImage.append(tf.Variable(tf.random_normal([1, 400, 240, 3])))
#
# indata = tf.Variable(tf.random_normal([1, 20, 6]))
# init = tf.global_variables_initializer()  # 遍历tf.global_variables()返回的全局变量列表
# with tf.Session() as sess:
#     sess.run(init)  # 参数初始化
#
# is_train = True
# phase = True
#
# characteristic = []  # 特征融合
#
# for i in range(3):
#     with tf.variable_scope('main' + str(i)):  # 没有这个下面神经网络会出现重命(这是个命名函数)
#         conv1 = tf.contrib.layers.convolution2d(inputs=inImage[i], num_outputs=32, kernel_size=[14, 14],
#                                                 padding="VALID", biases_initializer=None, trainable=is_train)
#         batch_norm_1 = tf.contrib.layers.batch_norm(conv1, is_training=phase, scope='bn_1', center=True, scale=True)
#         print(batch_norm_1)
#         '''Variable bn_1/gamma already exists, disallowed. Did you mean to set reuse=True or reuse=tf.AUTO_REUSE in VarScope? Originally defined at:'''
#         # ?,388,121,32
#         mp1 = tf.contrib.layers.max_pool2d(batch_norm_1, kernel_size=[3, 3], stride=3)
#         # ?,386,119,32
#         conv2 = tf.contrib.layers.convolution2d(inputs=mp1, num_outputs=32, kernel_size=[7, 7],
#                                                 padding="VALID", biases_initializer=None, trainable=is_train)
#         batch_norm_2 = tf.contrib.layers.batch_norm(conv2, is_training=phase, scope='bn_2', center=True,
#                                                     scale=True)
#         # ?,192,59,32
#         mp2 = tf.contrib.layers.max_pool2d(batch_norm_2, kernel_size=[3, 3])
#
#         conv3 = tf.contrib.layers.convolution2d(inputs=mp2, num_outputs=32, kernel_size=[3, 3],
#                                                 padding="VALID", biases_initializer=None, trainable=is_train)
#         batch_norm_3 = tf.contrib.layers.batch_norm(conv3, is_training=phase, scope='bn_3', center=True,
#                                                     scale=True)
#         mp3 = tf.contrib.layers.max_pool2d(batch_norm_3, kernel_size=[3, 3])
#         conv4 = tf.contrib.layers.convolution2d(inputs=mp3, num_outputs=32, kernel_size=[3, 3],
#                                                 padding="VALID", biases_initializer=None, trainable=is_train)
#         batch_norm_4 = tf.contrib.layers.batch_norm(conv4, is_training=phase, scope='bn_4', center=True,
#                                                     scale=True)
#         mp4 = tf.contrib.layers.max_pool2d(batch_norm_4, kernel_size=[3, 3])
#
#         conv5 = tf.contrib.layers.convolution2d(inputs=mp4, num_outputs=64, kernel_size=[3, 3],
#                                                 padding="VALID", biases_initializer=None, trainable=is_train)
#         batch_norm_5 = tf.contrib.layers.batch_norm(conv5, is_training=phase, scope='bn_5', center=True,  # (n,11,4,64)
#                                                     scale=True)
#         img_result = tf.reshape(tf.transpose(batch_norm_5, perm=[0, 2, 1, 3]), shape=[-1, batch_norm_5.shape[2],
#                                                                                       batch_norm_5.shape[1] *
#                                                                                       batch_norm_5.shape[
#                                                                                           3]])  # (1,4,704)  行*过滤器个数
#
#         '''
#         CNN
#
#         img_result = tf.reshape(tf.transpose(batch_norm_5,perm=[0,2,1,3]),shape=[-1,batch_norm_5.shape[2]*batch_norm_5.shape[1]*batch_norm_5.shape[3]])  #(1,4,704)  行*过滤器个数
#         print(img_result)  #Tensor("Reshape:0", shape=(1, 2816), dtype=float32)
#         rnn_out= slim.fully_connected(img_result, 1024, activation_fn=tf.nn.leaky_relu,
#                                                           weights_initializer=normalized_columns_initializer(),
#                                                           biases_initializer=None)
#
#         print(rnn_out.shape) #(1, 1024)
#         '''
#
#         '''取消CNN
#         #img_result = tf.reshape( tf.transpose(inImage,perm=[0,2,1,3]),shape=[-1,inImage.shape[2],inImage.shape[1]*inImage.shape[3]])  #(1,4,704)  行*过滤器个数
#         #print(img_result)  #(1,240,1200)
#         '''
#
#         lstms = []
#         for i in range(2):
#             lstm_cell = tf.contrib.rnn.LSTMCell(1024, state_is_tuple=True, trainable=is_train, name="LSTM_%d" % i)
#             lstms.append(lstm_cell)
#         cell = tf.contrib.rnn.MultiRNNCell(lstms)  # 4层->最后一个cell输出
#         lstm_outputs, lstm_state = tf.nn.dynamic_rnn(cell, img_result,
#                                                      dtype=tf.float32)  # lstm_outputs:shape=(1, 4, 1024)
#
#         rnn_out = lstm_outputs[:, -1]  # (1,1024)  取最后一层输出
#
#         batch_norm = tf.contrib.layers.batch_norm(rnn_out, is_training=phase, scope='bn', center=True,
#                                                   # shape=(1, 1024)
#                                                   scale=True)
#         characteristic.append(batch_norm)  # 特征融合
#
# '''另外的神经网络'''
# # indata = tf.placeholder(tf.float32, shape=[None] + [20, 6])  # 时间序列输入    (?,20,6)
# lstms1 = []
# for i in range(2):
#     lstm_cell1 = tf.contrib.rnn.LSTMCell(1024, state_is_tuple=True, trainable=is_train, name="LSTM1_%d" % i)
#     lstms1.append(lstm_cell1)
# cell1 = tf.contrib.rnn.MultiRNNCell(lstms1)
#
# lstm_outputs, lstm_state = tf.nn.dynamic_rnn(cell1, indata, dtype=tf.float32)  # outputs：RNN的最后一层的输出
#
# # rnn_out = tf.reshape(lstm_outputs, [-1, 1024])[-1:]
# rnn_out1 = lstm_outputs[:, -1]  # 最后一层最后一个单元输出
# characteristic.append(rnn_out1)
#
# portfolio_state = [[1, 0.99065]]
#
# concat = tf.concat([norm for norm in characteristic] + [portfolio_state], 1)  # (1,1024)->(1,1024+x)
# print(concat)
# fc_layer_1 = slim.fully_connected(concat, 512, activation_fn=tf.nn.leaky_relu,
#                                   weights_initializer=normalized_columns_initializer(),
#                                   biases_initializer=None)  # (1,512)
# '''
# #lstm_state :shape(1,1024) *8
# outputs对应的是RNN的最后一层的输出，states对应的是每一层的最后一个step的输出，在完成了两层的定义后，
# outputs的shape并没有变化，而states的内容多了一层，分别对应RNN的两层输出。
# state中最后一层输出对应着outputs最后一步的输出。
# '''
# dropout = slim.dropout(fc_layer_1, 1.0)  # shape=(1, 512)
# streamAC, streamVC = tf.split(dropout, [256, 256], 1)  # shape=(1, 256)，shape=(1, 256)
#
# streamA = tf.contrib.layers.flatten(streamAC)  # shape=(1, 256)
# streamV = tf.contrib.layers.flatten(streamVC)  # shape=(1, 256)
#
# AW = tf.Variable(tf.random_normal([256, 3]))
# VW = tf.Variable(tf.random_normal([256, 1]))
#
# Advantage = tf.matmul(streamA, AW)  # （1，3）
# Value = tf.matmul(streamV, VW)  # （1，1）
#
# Qout = Value + tf.subtract(Advantage,  # (1,3)
#                            tf.reduce_max(Advantage, reduction_indices=1, keep_dims=True))
# predict = tf.argmax(Qout, 1)  # (1,)
# Q_dist = tf.nn.softmax(Qout / 0.1)  # (self.Qout/self.temp)  (1,3)
#
# targetQ = tf.placeholder(shape=[None], dtype=tf.float32)  # (?,)
# actions = tf.placeholder(shape=[None], dtype=tf.int32)
#
# actions_onehot = tf.one_hot(actions, depth=3)  # (?,3)
#
# Q = tf.reduce_sum(tf.multiply(Qout, actions_onehot), reduction_indices=1)  # (?,)
#
# td_error = tf.losses.huber_loss(labels=targetQ, predictions=Q)  # Tensor("huber_loss/value:0", shape=(), dtype=float32)
# # self.td_error = tf.square(self.targetQ - self.Q)
#
# loss = tf.reduce_mean(td_error)  # Tensor("Mean:0", shape=(), dtype=float32)
#
# update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, "1")
# with tf.control_dependencies(update_ops):
#     trainer = tf.train.AdamOptimizer(learning_rate=0.01)
#     updateModel = trainer.minimize(loss)

# for var in tf.global_variables():  #tf.trainable_variables():每一层参数设置
#     print(var) # 获取每个变量的shape，其类型为'tensorflow.python.framework.tensor_shape.TensorShape'
# for var in tf.global_variables():  #tf.trainable_variables():每一层参数设置
#     shape = var.shape # 获取每个变量的shape，其类型为'tensorflow.python.framework.tensor_shape.TensorShape'
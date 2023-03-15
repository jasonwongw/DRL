from network import updateTargetGraph, experience_buffer, updateTarget, TrendPredictNetwork
import os
import tensorflow as tf
from agent import Agent
import numpy as np
from environment import Environment
import sys
from time import sleep
import json
import random
import matplotlib.pyplot as plt
import pandas as pd

class PolicyLearner:
    def __init__(self, load_model=True, learning_rate=0.005, min_trading_unit=1000, max_trading_unit=25000,    #600031 30快 1000，50000  ，600519 500-1000
                 delayed_reward_threshold=.01, training=True):
        self.environment = Environment()
        self.agent = Agent(self.environment,
                           min_trading_unit=min_trading_unit,
                           max_trading_unit=max_trading_unit,
                           delayed_reward_threshold=delayed_reward_threshold)

        self.y = .99
        #self.discount_factor = .8  # 0.8**30 = 0.004
        self.startE = 1
        self.endE = 0.1
        self.anneling_steps = 20000.
        self.num_episodes = 200
        self.pre_train_steps = 50
        self.max_epLength = 300
        self.replay_memory = 20
        self.training_step = 5

        self.load_model = load_model
        self.path = './dqn'


        if not os.path.exists(self.path):
            os.makedirs(self.path)

        # self.h_size = 512
        self.tau = 0.001

        tf.reset_default_graph()

        self.network_type = 20  # , 6, 7]
        self.data_type = [Environment.TYPE_BASIC,Environment.TYPE_DMI ,Environment.TYPE_STO,Environment.TYPE_QINGGAN]

        self.buffer_size = 0
        for image_type in self.data_type:
            image_size = 1
            for shape in self.environment.RANGE_SHAPES[self.network_type][image_type]:
                image_size *= shape
            self.buffer_size += image_size

        self.buffer_size = ((10 * (1024 ** 3)) // (
        self.buffer_size * 2 * self.max_epLength)) // 10 * 10  # 10GB / Imagesize  #250   #360
        #print(self.buffer_size)
        self.mainQN = [TrendPredictNetwork(learning_rate=learning_rate, model_type=self.network_type,  #赋值self.mainQN类对象
                                           name='main_%s_%s' % (self.network_type, type), data_type=type) for type in
                       self.data_type]
        if training:
            self.targetQN = [TrendPredictNetwork(learning_rate=learning_rate, model_type=self.network_type,
                                                 name='target_%s_%s' % (self.network_type, type), data_type=type) for
                             type in self.data_type]

    def train(self):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver(max_to_keep=1, reshape=True)
        trainables = tf.trainable_variables() #[<tf.Variable 'main_20_0/Conv/weights:0' shape=(14, 14, 3, 32) dtype=float32_ref>, <tf.Variable 'main_20_0/bn_1/gamma:0' shape=(32,)...]
        targetOps = updateTargetGraph(trainables, self.tau) #[<tf.Tensor 'Assign:0' shape=(14, 14, 3, 32) dtype=float32_ref>, <tf.Tensor 'Assign_1:0' shape=(32,) dtype=float32_ref>, <tf.Tensor 'Assign_2:0' shape=(32,)...
        rList = []
        # portfolio_list=[]
        total_steps = 0
        myBuffer = experience_buffer(self.buffer_size)  #类对象
        episode_buffer = experience_buffer()
        e = self.startE

        stepDrop = (self.startE - self.endE) / self.anneling_steps
        with tf.Session() as sess:
            # 初始化变量.
            sess.run(init)
            if self.load_model == True:
                print('Loading Model...')

                ckpt = tf.train.get_checkpoint_state(self.path)
                saver.restore(sess, ckpt.model_checkpoint_path)
                e = self.endE

            # 与主要神经网络相同，设置目标神经网络
            updateTarget(targetOps, sess)

            for ii in range(self.num_episodes):  #200
                rAll = 0
                d = False
                j = 0
                episode_buffer.buffer = []
                episode_reward_buffer = []
                self.environment.reset()
                self.agent.reset()
                # print('%d 第二个 episode 初始化 :' % ii,self.environment.idx, self.environment.KOSPI_idx, 'total num :',total_steps, '项目代码',self.environment.chart_code)
                s = [self.environment.get_image(self.network_type, datatype) for datatype in self.data_type]  #读取照片数组0basic 4dmi 5sto
                #s是一开始的状态
                #print(len(s)) #idx的3张照片(0,4,5),ii每次迭代就是上张照片
                '''
                s为0，4，5组的相同idx各一张
                180 ./a/chart_images_other_basic/000060_20_180.jpg
                180 ./a/chart_images_other_STO/000060_20_180.jpg
                180 ./a/chart_images_other_DMI/000060_20_180.jpg
                3
                181 ./a/chart_images_other_basic/000060_20_181.jpg
                '''

                s_potfol = np.array(self.agent.get_states())  #[持股比例，投资组合价值百分比]  (2,)
                #print(s_potfol.shape)
                episode_step = 1

                while j < self.max_epLength and not d:  #max_epLength=30（照片）

                    j += 1

                    # 选择行为作为输入值（仅托架+螺栓）
                    all_Q_d = np.zeros([self.agent.NUM_ACTIONS])
                    for i, mainQN in enumerate(self.mainQN):  #self.mainQN有3个->0，4，5
                        Q_d = sess.run(mainQN.Q_dist, feed_dict={mainQN.inImage: [s[i]], #(1,3)  [[0.13918856 0.8154896  0.04532186]]
                                                                 mainQN.portfolio_state: [s_potfol],
                                                                 mainQN.temp: e,
                                                                 mainQN.keep_per: (1 - e) + 0.1,
                                                                 mainQN.phase: False})  #action概率:[[0.6562301  0.2436781  0.10009175]]
                        all_Q_d += Q_d[0]  #将三张图片的约车Q_d相加
                    # 将所有神经网络的概率值相加后除以
                    # print(np.sum(all_Q_d))
                    all_Q_d /= len(self.data_type)
                    all_Q_d /= np.sum(all_Q_d)
                    # print(np.sum(all_Q_d))
                    a = np.random.choice(all_Q_d, p=all_Q_d) #给出选取概率p   a=0.3777557866437602
                    action = np.argmax(all_Q_d == a)
                    # 将行动传达给策略(利润增长率delayed_reward)
                    delayed_reward = self.agent.act(action=action, confidence=all_Q_d[action]) #action所带来的reward #profitloss
                    d = self.environment.step() #idx+1  d=F OR T,每次调用idx+1
                    if e > self.endE and total_steps > self.pre_train_steps:
                        e -= stepDrop
                    '''
                    immediate_reward, delayed_reward = self.agent.act(action=action, confidence=all_Q_d[action])

                    if e > self.endE and total_steps > self.pre_train_steps:
                        e -= stepDrop


                    #转到下一个索引
                    d = self.environment.step()
                    if (delayed_reward == 0 and episode_step % 5 == 0) or d:
                        delayed_reward = immediate_reward
                        self.agent.base_portfolio_value = self.agent.portfolio_value
                    '''
                    # 下一张图片，获取阿波
                    # print('total step :', total_steps, 'current episode step : ', j, 'idx :', self.environment.idx, 'kospi_idx', self.environment.KOSPI_idx, '종목코드',self.environment.chart_code)
                    s1 = [self.environment.get_image(self.network_type, datatype) for datatype in self.data_type] #取照片
                    #print(len(s1))
                    '''
                    181 ./a/chart_images_other_basic/000060_20_181.jpg
                    181 ./a/chart_images_other_STO/000060_20_181.jpg
                    181 ./a/chart_images_other_DMI/000060_20_181.jpg
                    3
                    182 ./a/chart_images_other_basic/000060_20_182.jpg
                    182 ./a/chart_images_other_STO/000060_20_182.jpg
                    182 ./a/chart_images_other_DMI/000060_20_182.jpg
                    3
                    '''
                    s1_potfol = np.array(self.agent.get_states())

                    episode_reward_buffer.append(0) #
                    # 保存到缓冲区
                    # 原始缓冲区顺序：状态动作奖励下一个状态是否结束
                    # 修改缓冲区：当前图像动作奖励下一个图像，下一个Popole状态是否结束前一个状态LSTM，状态LSTM当前Popole
                    # 修复缓冲区：当前图像动作下一个图像，下一个波波状态是否结束前一个状态LSTM，状态LSTM当前波波补偿（设置折扣）
                    # episode_buffer.add([s, action, delayed_reward, s1, s1_potfol, d, before_rnn_state, rnn_state, s_potfol  ]  )
                    episode_buffer.buffer.append([s, action, s1, s1_potfol, d, s_potfol])  #s是当前照片（状态），s1是下一个
                    if total_steps > self.pre_train_steps and total_steps % self.training_step == 0:  #当超过2000次之后，每5次更新一次dqn网络参数
                        try:
                            # 从缓冲区获取数据
                            # 学习模式且存在延迟补偿时更新策略神经网络
                            # 原始缓冲区顺序：状态动作奖励下一个状态是否结束
                            # 修改缓冲区：当前图像动作奖励下一个图像，下一个Popole状态是否结束前一个状态LSTM，状态LSTM当前Popole
                            # 批处理学习数据大小

                            trainBatch, size = myBuffer.sample(self.replay_memory)  # (self.batch_size)
                            #trainBatch就是20个[s, action, s1, s1_potfol, d, s_potfol,...]

                            # print('训练数据提取结果 : ', trainBatch.shape)
                            # 奖励必须乘以折扣因子以影响前一行为


                            for i in range(len(self.data_type)):
                                # 下面执行Double-DQN，更新target Q-value
                                # 在主要神经网络中选择行为。
                                # 学习时不使用底座和螺栓
                                # 为了学习LSTM，从随机日期到replay memory，在随机故事中选择和使用任意数量的内容
                                # 修复缓冲区：当前图像动作下一个图像，下一个波波状态是否结束前一个状态LSTM，状态LSTM当前波波补偿（设置折扣）
                                feed_dict = {self.mainQN[i].inImage: [datas[i] for datas in trainBatch[:, 2]],
                                             #trainBatch[:,2]:索引第二列->(20,)其中每一个都包含n张s1图片，n为输入多少张图片数
                                             self.mainQN[i].portfolio_state: [data for data in trainBatch[:, 3]], #s1_potfol (20,)其中每一个都是(2,)
                                             self.mainQN[i].keep_per: 1.0,
                                             self.mainQN[i].phase: True}
                                Q1 = sess.run(self.mainQN[i].predict,  ##action概率:[[0.6562301  0.2436781  0.10009175]]
                                              feed_dict=feed_dict)
                                del feed_dict   #减少内存消耗
                                feed_dict_2 = {self.targetQN[i].inImage: [datas[i] for datas in trainBatch[:, 2]],
                                               self.targetQN[i].portfolio_state: [data for data in trainBatch[:, 3]],
                                               self.targetQN[i].keep_per: 1.0,
                                               self.targetQN[i].phase: True}
                                Q2 = sess.run(self.targetQN[i].Qout,  # feed_dict 需要修改
                                              feed_dict=feed_dict_2)
                                del feed_dict_2

                                # 根据结束与否制作假标签
                                end_multiplier = -(trainBatch[:, 4] - 1)
                                # 在目标神经网络的Q值中，从主要神经网络中获取行为均匀的第二个Q值。（这部分是doubleQ）
                                doubleQ = Q2[range(size), Q1]
                                # 对奖励加上双Q值。y是折扣因子
                                # #targetQ是即时奖励+下一状态的最大奖励（doubleQ）
                                targetQ = trainBatch[:, 6] + (self.y * doubleQ * end_multiplier)  #Double DQN的q
                                # 和我们的目标值一起更新神经网络。
                                # 对于行为，通过与targetQ值的差异来避免和更新损失
                                # 原始缓冲区顺序：状态动作奖励下一个状态是否结束
                                # 修改缓冲区：当前图像动作奖励下一个图像，下一个Popole状态是否结束前一个状态LSTM，状态LSTM当前Popole

                                feed_dict = {self.mainQN[i].inImage: [datas[i] for datas in trainBatch[:, 0]],  #s
                                             self.mainQN[i].portfolio_state: [data for data in trainBatch[:, 5]],  #s_potfol
                                             self.mainQN[i].targetQ: targetQ,
                                             self.mainQN[i].actions: trainBatch[:, 1],
                                             self.mainQN[i].keep_per: 1.0,
                                             self.mainQN[i].phase: True}
                                _ = sess.run(self.mainQN[i].updateModel, feed_dict=feed_dict)
                                del feed_dict
                                '''
                                _ = sess.run(self.mainQN[i].updateModel, \
                                             feed_dict={self.mainQN[i].inImage: np.vstack(trainBatch[:, 0]),
                                                        self.mainQN[i].targetQ: targetQ,
                                                        self.mainQN[i].actions: trainBatch[:, 1]})
                                '''
                            del trainBatch, size
                            updateTarget(targetOps, sess)
                        except IndexError as eee:
                            print(eee)

                    rAll += delayed_reward

                    # rAll = delayed_reward
                    # 改变状态
                    del s
                    s = s1
                    del s_potfol
                    s_potfol = s1_potfol
                    total_steps += 1
                    episode_step += 1

                # portfolio_list.append(self.agent.portfolio_value)
                # 将折扣奖励添加到花絮缓冲区
                accumulate = 0
                episode_reward_buffer[-1] = delayed_reward  #一次迭代
                episode_reward_buffer.reverse() #翻转
                # print('%s episode_reward_len : ' % ii, len(episode_reward_buffer), 'episode_buffer_len :', len(episode_buffer.buffer))
                for i, reward in enumerate(episode_reward_buffer): #
                    accumulate = self.y * accumulate + reward
                    idx = -(i + 1)
                    episode_buffer.buffer[idx] += [accumulate]
                    #print(idx, episode_buffer.buffer[idx])

                myBuffer.add(episode_buffer.buffer) #每轮30个   [s,a,s_,...]:7

                if len(rList) + 1 >= self.buffer_size:
                    # self.buffer[0:1] = []
                    del rList[0]
                rList.append(rAll)  #每一轮epoch的reward
                self.environment.chartcode_value[
                    self.environment.chart_code] += 1 if self.agent.portfolio_value > self.agent.initial_balance else -1
                print("%d  %s %s %d %d %d" % (
                ii,  self.environment.chart_code, delayed_reward,
                self.agent.portfolio_value, self.agent.minimum_portfolio_value, self.agent.maximum_portfolio_value))

                #输出
                # print("%d %4f %d %4f %4f %d %d"% (total_steps, np.mean(rList[-10:]), np.mean(portfolio_list), np.max(rList[-10:]),np.min(rList[-10:]),np.max(portfolio_list),np.min(portfolio_list)))#e)
                # print(sys.getsizeof(myBuffer.buffer), sys.getsizeof(episode_buffer.buffer))
                # portfolio_list= []
                if total_steps > self.pre_train_steps and ii % 50 == 0:  #每50次迭代
                    try:
                        saver.save(sess, self.path + '/model-' + str(ii) + '.cptk')
                        # print("Saved Model")
                    except:
                        pass
            # 显示学习结束平均奖励
            saver.save(sess, self.path + '/model-' + str(ii) + '.cptk')
            print("平均episode奖励值 : " + str(sum(rList) / self.num_episodes))
            #平均episode奖励值 : 7.589135000000003
                

    def test(self):  #idx就相当于时间，一天一天的往后
        init = tf.global_variables_initializer()
        saver = tf.train.Saver(max_to_keep=1, reshape=True)

        # portfolio_list=[]
        total_steps = 0

        e = self.endE

        with tf.Session() as sess:
            # 初始化变量。
            sess.run(init)
            print('Loading Model...')
            # 开启model
            ckpt = tf.train.get_checkpoint_state(self.path)
            saver.restore(sess, ckpt.model_checkpoint_path)
            # 开始
            for ii, code in enumerate(self.environment.chartcode_list):
                self.rAll = []
                self.stock_price=[]
                self.sharpe=[]
                d = False
                j = 0

                print('%d %s episode' % (ii, code))
                if not self.environment.reset(code):  #self.environment.reset(code)是经过的
                    continue
                chart_data = self.environment.chart_data
                trend = chart_data[:, 1]
                #print(self.environment.chart_data)  #看这里是否经过了self.environment.reset(code)
                self.agent.reset()
                s = [self.environment.get_image(self.network_type, datatype) for datatype in self.data_type]
                s_potfol = np.array(self.agent.get_states())

                episode_step = 1

                while j < self.max_epLength and not d:  #<30 ,三张相同时间图片输入得到结果
                    j += 1

                    # 选择行为作为输入值
                    all_Q_d = np.zeros([self.agent.NUM_ACTIONS])
                    for i, mainQN in enumerate(self.mainQN):  #三张图片分别带入
                        Q_d = sess.run(mainQN.Q_dist, feed_dict={mainQN.inImage: [s[i]],
                                                                 mainQN.portfolio_state: [s_potfol],
                                                                 mainQN.temp: e,
                                                                 mainQN.keep_per: (1 - e) + 0.1,
                                                                 mainQN.phase: True})
                        all_Q_d += Q_d[0]
                    # print("1:{}".format(all_Q_d))  #1:[0.78131707 1.67413716 1.54454569]
                    all_Q_d /= len(self.data_type)
                    all_Q_d /= np.sum(all_Q_d)
                    #print("1:{}".format(all_Q_d)) #[0.25632965 0.49337724 0.25029311]
                    a = np.random.choice(all_Q_d, p=all_Q_d)
                    action = np.argmax(all_Q_d==a)
                    #print("5:{}".format(action))
                    # 向政策传达行动
                    delayed_reward = self.agent.act(action=action, confidence=all_Q_d[action])
                    self.sharpe.append(self.agent.portfolio_value)
                    self.stock_price.append((trend[self.environment.idx]-trend[184])/trend[184])  #price
                    d = self.environment.step()

                    s1 = [self.environment.get_image(self.network_type, datatype) for datatype in self.data_type] #下一个状态
                    s1_potfol = np.array(self.agent.get_states())

                    self.rAll.append(delayed_reward) #+= delayed_reward

                    # rAll = delayed_reward
                    # 改变状态.
                    del s
                    s = s1
                    del s_potfol
                    s_potfol = s1_potfol
                    total_steps += 1
                    episode_step += 1
                mean_stock=np.mean(self.stock_price)  #股价平均比例
                mean_rAll=np.mean(self.rAll)  #比例
                mean_sharpe=np.mean(self.sharpe)  #资金
                self.sharpe1 = pd.DataFrame(self.sharpe,columns=list('A'))
                self.sharpe1['daily return']=self.sharpe1['A'].pct_change()  #日回报率
                SR=self.sharpe1['daily return'].mean()/self.sharpe1['daily return'].std()  #夏普比率：总资产日回报率的均值除以日回报率的标准差
                ASR = np.sqrt(1) * SR

                print('my account{}'.format(self.rAll))
                print('stock{}'.format(self.stock_price))
                print("%d  %s %s %d %d %d" % (
                ii,  self.environment.chart_code, delayed_reward,
                self.agent.portfolio_value, self.agent.minimum_portfolio_value, self.agent.maximum_portfolio_value))
                #print(self.sharpe1)
                print("夏普比率:%s 投资组合平均资金:%d 投资组合平均资金比例:%s 股价平均比例:%s" %(ASR,mean_sharpe,mean_rAll,mean_stock))

    def draw(self, save_name1):
        # 绘制结果
        # states_sell, states_buy, profit_rate_account, profit_rate_stock = self.get_info()
        # invest = profit_rate_account[-1]
        # total_gains = self.total_profit
        # close = self.trend
        # fig = plt.figure(figsize=(15, 5))
        # plt.plot(close, color='r', lw=2.)
        # plt.plot(close, 'v', markersize=8, color='k', label='selling signal', markevery=states_sell)
        # plt.plot(close, '^', markersize=8, color='m', label='buying signal', markevery=states_buy)
        # plt.title('total gains %f, total investment %f%%' % (total_gains, invest))
        # plt.legend()
        # plt.savefig(save_name1)
        # plt.close()

        fig = plt.figure(figsize=(15, 5))
        plt.plot(self.rAll, label='my account',color='red',linestyle=':')
        plt.legend(loc='upper left')
        plt.twinx()  # 将ax1的x轴也分配给ax2使用
        plt.plot(self.stock_price, label='stock')
        plt.legend(loc='upper right')
        plt.savefig(save_name1)
        plt.close()

if __name__ == "__main__":
    obj = PolicyLearner(load_model=False)
    #obj.train()
    obj.test()
    obj.draw('601318-22')

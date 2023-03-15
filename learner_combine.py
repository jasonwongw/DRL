from network_2_combine import updateTargetGraph, experience_buffer, updateTarget,TrendPredictNetwork
import os
import tensorflow as tf
from agent import Agent
import numpy as np
from environment import Environment
import sys
from time import sleep
import json
class PolicyLearner:
    def __init__(self,load_model=True, learning_rate = 0.005, min_trading_unit=0, max_trading_unit=100, delayed_reward_threshold=.01, training = True):
        self.environment = Environment()  #调用环境，类对象
        self.agent = Agent(self.environment,  #调用智能体，类对象
                           min_trading_unit=min_trading_unit,
                           max_trading_unit=max_trading_unit,
                           delayed_reward_threshold=delayed_reward_threshold)

        self.y = .99
        self.startE = 1
        self.endE = 0.1
        self.anneling_steps = 20000.
        self.num_episodes = 200
        self.pre_train_steps = 40
        self.max_epLength = 30
        self.replay_memory = 20
        self.training_step= 5

        self.load_model = load_model
        self.path = './dqn-combine'


        if not os.path.exists(self.path):
            os.makedirs(self.path)  #创建目录

        # self.h_size = 512
        self.tau = 0.001

        #self.datavalue = np.genfromtxt("./chart_data/000060_1.csv", delimiter=',')  # 读取数值数据
        #self.data_value = np.flip(self.datavalue, 0)  #日期正序就不用颠倒

        tf.reset_default_graph()


        self.network_type = 20#天数, 6, 7]
        self.data_type = [Environment.TYPE_BASIC, Environment.TYPE_STO, Environment.TYPE_DMI]

        self.buffer_size = 0
        for image_type in self.data_type:
            image_size = 1
            for shape in self.environment.RANGE_SHAPES[self.network_type][image_type]:
                image_size *= shape
            self.buffer_size+=image_size


        self.buffer_size = ((20*(1024**3)) // (self.buffer_size*2*self.max_epLength))//10 * 10 #10GB / Imagesize
        #print(self.buffer_size)
        self.mainQN = TrendPredictNetwork(learning_rate=learning_rate, model_type=self.network_type, name='main_%s_'%(self.network_type),data_type=self.data_type)  #主网络
        if training:
            self.targetQN = TrendPredictNetwork(learning_rate=learning_rate, model_type=self.network_type, name='target_%s_'%(self.network_type),data_type=self.data_type) #目标网络


    def train(self):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver(max_to_keep=1,reshape=True)
        trainables = tf.trainable_variables()

        targetOps = updateTargetGraph(trainables, self.tau)
        rList = []
        #portfolio_list=[]
        total_steps = 0
        myBuffer = experience_buffer(self.buffer_size)
        episode_buffer = experience_buffer()
        e = self.startE

        stepDrop = (self.startE - self.endE) / self.anneling_steps
        with tf.Session() as sess:
            # 初始化变量.
            sess.run(init)
            if self.load_model == True:
                print('Loading Model...')
                # 读取model
                ckpt = tf.train.get_checkpoint_state(self.path)
                saver.restore(sess, ckpt.model_checkpoint_path)
                e = self.endE


            # 与主要神经网络相同，设置目标神经网络
            updateTarget(targetOps, sess)
            for ii in range(self.num_episodes):
                rAll = 0
                d = False
                j = 0
                episode_buffer.buffer = []

                self.environment.reset() #这里之后便有了self.idx，
                self.agent.reset()

                self.data_value=self.environment.chart_data  #获取数值数据
                datanumber = np.array(self.data_value[self.environment.idx - 20:self.environment.idx])

                s = [self.environment.get_image(self.network_type,datatype) for datatype in self.data_type]  #获取图像数据
                s_potfol = np.array(self.agent.get_states()) #（2，）



                episode_step = 1

                while j < self.max_epLength and not d:

                    j += 1

                    #all_Q_d = np.zeros([self.agent.NUM_ACTIONS])
                    feed_dict = {self.mainQN.indata:[datanumber] ,self.mainQN.portfolio_state:[s_potfol], self.mainQN.temp:e, self.mainQN.keep_per:(1-e)+0.1, self.mainQN.phase:True}
                    for i, _ in enumerate(self.data_type):
                        feed_dict[self.mainQN.inImage[i]] = [s[i]]  #往字典里添加东西
                    #智能体决策动作
                    Q_d = sess.run(self.mainQN.Q_dist, feed_dict=feed_dict)

                    #print(np.sum(all_Q_d))
                    Q_d = Q_d[0]
                    a = np.random.choice(Q_d, p=Q_d)
                    action = np.argmax(Q_d == a)
                    #动作传给环境，环境得出reward和下一个状态
                    delayed_reward = self.agent.act(action=action, confidence=Q_d[action])
                    d = self.environment.step()  #s1、 idx+=1
                    if e > self.endE and total_steps > self.pre_train_steps:
                        e -= stepDrop

                    #下一张图片，获取信息
                    s1 = [self.environment.get_image(self.network_type,datatype) for datatype in self.data_type]
                    s1_potfol = np.array(self.agent.get_states())

                    datanumber1=np.array(self.data_value[self.environment.idx - 20:self.environment.idx])

                    # 保存到缓冲区
                    # 原始缓冲区顺序：状态动作奖励下一个状态是否结束
                    # 修改缓冲区：当前图像动作奖励下一个图像，下一个波状态是否结束前一个状态LSTM，状态LSTM当前波
                    # 重新修改缓冲区：当前图像动作下一个图像，下一个Popole状态是否结束前一个状态LSTM，状态LSTM当前Popole奖励（设置折扣）
                    '''经验池要添加数值数据'''
                    episode_buffer.buffer.append([s, action, s1, s1_potfol, d, s_potfol,datanumber,datanumber1,delayed_reward])
                    if total_steps > self.pre_train_steps and total_steps % self.training_step == 0:
                        try:
                            # 从缓冲区获取数据
                            # 学习模式且存在延迟补偿时更新策略神经网络
                            # 原始缓冲区顺序：状态动作奖励下一个状态是否结束
                            # 修改缓冲区：当前图像动作奖励下一个图像，下一个波状态是否结束前一个状态LSTM，状态LSTM当前波波
                            # 批处理学习数据大小
                            trainBatch, size = myBuffer.sample(self.replay_memory)#(self.batch_size)
                            # print(trainBatch[0][3].shape) #(20,14)  #[array([[   [[
                            # print(trainBatch[0][7].shape)
                            #(trainBatch[0][8])  #0.002599190512171458

                            #必须将奖励乘以折扣因子，以影响前一行为

                            # 下面执行Double-DQN，更新target Q-value
                            # 在主要神经网络中选择行为。
                            # 为了学习LSTM，在随机的故事中，从随机的日期开始选择和使用replay memory。
                            # 重新修改缓冲区：当前图像动作下一个图像，下一个Popole状态是否结束前一个状态LSTM，状态LSTM当前Popole奖励（设置折扣）

                            feed_dict = {self.mainQN.portfolio_state: [data for data in trainBatch[:,3]],  #trainBatch（20，8）
                                         self.mainQN.indata:[value for value in trainBatch[:,7]],   #[value]: 20,1,20,14
                                                     self.mainQN.keep_per: 1.0,
                                                     self.mainQN.phase: True}
                            for i in range(len(self.data_type)):
                                feed_dict[self.mainQN.inImage[i]] = [datas[i] for datas in trainBatch[:,2]]
                            Q1 = sess.run(self.mainQN.predict,  # 从经验池中抽取20条
                                          feed_dict=feed_dict)  #[1 1 1 1 1 1 1 1 1 0 1 1 0 0 0 0 0 0 0 0]
                            del feed_dict  # 减少内存消耗
                            feed_dict_2 = {self.targetQN.portfolio_state: [data for data in trainBatch[:, 3]],
                                           self.targetQN.indata: [value for value in trainBatch[:,7]],  #'''这里错了，两个不同的类（网络），一个是mainQN，一个是targetQN，刚刚将mainQN.indata代入了targetQN了'''
                                           self.targetQN.keep_per: 1.0,
                                           self.targetQN.phase: True}
                            for i in range(len(self.data_type)):
                                feed_dict_2[self.targetQN.inImage[i]]= [datas[i] for datas in trainBatch[:,2]]

                            Q2 = sess.run(self.targetQN.Qout,  # feed_dict
                                          feed_dict=feed_dict_2) #[[  4.043326    -7.2520995  -14.229815  ][ -7.4833527  -27.091757    -4.552919  ][ -3.5521657   -8.900963    -9.040138  ][-19.134283   -25.107372    -5.9249563 ]...]

                            del feed_dict_2

                            # 根据结束与否制作假标签
                            end_multiplier = -(trainBatch[:, 4] - 1)  #[1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
                            # 在目标神经网络的Q值中，从主要神经网络中获取行为第二个Q值。（这部分是doubleQ）
                            doubleQ = Q2[range(size), Q1]  #[-18.887108   12.959032   -6.5908947   7.056612   -2.4452925  -0.84719783.1815534  -9.61303     1.840996  -11.404724   -9.563263    1.5815530.3515861   5.982282   -3.8067884  -4.6535473  -6.0412636   8.104461 -8.457569   -2.7319117]
                            # 对奖励加上doubleQ值。y是折扣因子
                            # targetQ是即时奖励+以下状态的最大奖励（doubleQ）
                            targetQ = trainBatch[:, 8] + (self.y * doubleQ * end_multiplier) #[-18.6982364654541 12.829442024230957 -6.52498579025268557.196045837402344 -2.210839548110962 -1.2487258052825930.8797378349304199 -11.786900062561035 -0.4474139404296875...]
                            # 和我们的目标值一起更新神经网络。
                            # 对于行为，通过与targetQ值的差异来避免和更新损失
                            # 原始缓冲区顺序：状态动作奖励下一个状态是否结束
                            # 修改缓冲区：当前图像动作奖励下一个图像，下一个波波状态是否结束前一个状态LSTM，状态LSTM当前波波

                            feed_dict3 = {self.mainQN.portfolio_state: [data for data in trainBatch[:, 5]],
                                          self.mainQN.indata: [value for value in trainBatch[:, 6]],
                                          self.mainQN.targetQ: targetQ,
                                          self.mainQN.actions: trainBatch[:, 1],
                                          self.mainQN.keep_per: 1.0,
                                          self.mainQN.phase: True}
                            for i in range(len(self.data_type)):
                                feed_dict3[self.mainQN.inImage[i]] =  [datas[i] for datas in trainBatch[:, 0]]
                            _ = sess.run(self.mainQN.updateModel, feed_dict=feed_dict3)
                            del feed_dict3

                            del trainBatch, size
                            updateTarget(targetOps, sess) #更新目标网络
                        except IndexError as eee:
                            print(eee)


                    rAll += delayed_reward

                    # 改变状态.
                    del s
                    s = s1
                    del s_potfol
                    s_potfol = s1_potfol
                    total_steps += 1
                    episode_step += 1

                #portfolio_list.append(self.agent.portfolio_value)
                #将奖励添加到花絮缓冲区

                myBuffer.add(episode_buffer.buffer)
                if len(rList)+1>= self.buffer_size:
                    # self.buffer[0:1] = []
                    del rList[0]
                rList.append(rAll)
                self.environment.chartcode_value[self.environment.chart_code] += 1 if self.agent.portfolio_value > self.agent.initial_balance else -1
                print("%d  %s %s %d %d %d" % (ii,  self.environment.chart_code, delayed_reward, self.agent.portfolio_value, self.agent.minimum_portfolio_value, self.agent.maximum_portfolio_value))

                if total_steps > self.pre_train_steps and ii % 50 == 0:
                    try:
                        saver.save(sess, self.path + '/model-' + str(ii) + '.cptk')
                        with open('./value_chart.txt','w') as f:
                            data = json.dumps(self.environment.chartcode_value)
                            f.write(data)
                        del data
                        #print("Saved Model")
                    except:
                        pass
            # 显示学习结束的平均奖励
            saver.save(sess, self.path + '/model-' + str(ii) + '.cptk')
            print("平均episode奖励值 : " + str(sum(rList) / self.num_episodes) )
    def test(self):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver(max_to_keep=1,reshape=True)

        #portfolio_list=[]
        total_steps = 0

        e = self.endE

        with tf.Session() as sess:
            # 初始化变量.
            sess.run(init)
            print('Loading Model...')
            # 读取model
            ckpt = tf.train.get_checkpoint_state(self.path)
            saver.restore(sess, ckpt.model_checkpoint_path)
            for ii, code in enumerate(self.environment.chartcode_list):
                rAll = 0
                d = False
                j = 0

                print('%d %s episode' % (ii, code))
                if not self.environment.reset(code):
                    continue
                self.agent.reset()

                s = [self.environment.get_image(self.network_type,datatype) for datatype in self.data_type]
                s_potfol = np.array(self.agent.get_states())

                episode_step = 1

                while j < self.max_epLength and not d:

                    j += 1

                    feed_dict = {self.mainQN.portfolio_state:[s_potfol], self.mainQN.temp:e, self.mainQN.keep_per:(1-e)+0.1, self.mainQN.phase:True}
                    for i, _ in enumerate(self.data_type):
                        feed_dict[self.mainQN.inImage[i]] = [s[i]]
                    Q_d = sess.run(self.mainQN.Q_dist, feed_dict=feed_dict)

                    Q_d = Q_d[0]
                    a = np.random.choice(Q_d, p=Q_d)
                    action = np.argmax(Q_d == a)

                    delayed_reward = self.agent.act(action=action, confidence=Q_d[action])
                    d = self.environment.step()

                    s1 = [self.environment.get_image(self.network_type,datatype) for datatype in self.data_type]
                    s1_potfol = np.array(self.agent.get_states())

                    rAll += delayed_reward

                    #改变状态s.
                    del s
                    s = s1
                    del s_potfol
                    s_potfol = s1_potfol
                    total_steps += 1
                    episode_step += 1
                print("%d  %s %s %d %d %d" % (ii,  self.environment.chart_code, delayed_reward, self.agent.portfolio_value, self.agent.minimum_portfolio_value, self.agent.maximum_portfolio_value))

if __name__ == "__main__":
    obj = PolicyLearner(load_model=False)
    obj.train()
    #obj.test()


















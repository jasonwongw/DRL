import tensorflow as tf
import numpy as np
from environment import Environment
import tensorflow.contrib.slim as slim
import random
from agent import Agent
from copy import deepcopy
import PIL.Image as pilimg

def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)

    return _initializer

class TrendPredictNetwork:

    MODEL_PATH = ['./stock_predict_basic','./stock_predict_bb','./stock_predict_macd','./stock_predict_obv','./stock_predict_dmi','./stock_predict_sto']
    INPUT_IMAGE_PATH = ['C:/chart_images_other_basic/','C:/chart_images_other_BB/','C:/chart_images_other_MACD/','C:/chart_images_other_OBV/','C:/chart_images_other_DMI/','C:/chart_images_other_STO/']
    LSTM_LAYERS = 4
    DEVICES=['/cpu:0','/gpu:0']

    TYPE_BASIC = 0
    TYPE_BB = 1
    TYPE_MACD = 2
    TYPE_OBV = 3
    TYPE_DMI = 4
    TYPE_STO = 5

    #RANGE_SHAPE = {5: [630, 130, 4], 20: [630, 245, 4], 60: [630, 550, 4], 6:[630,130,4], 7:[630,130,4], 25: [630, 280, 4]}
    #RANGE_SHAPE = {20: [400, 240, 4], 5: [630, 130, 4],  60: [630, 550, 4], 6: [630, 130, 4], 7: [630, 130, 4],25: [630, 280, 4]}
    RANGE_SHAPE = {20: [ [400, 240, 4], [320,235,4],[320,245,4],[320,250,4],[320,215,4],[320,215,4] ] }

    def __init__(self, model_type, is_train = True, data_type = 0, learning_rate = 0.01,name=None):

        self.model_type = model_type
        self.data_type = data_type
        device = self.DEVICES[0] #if is_train else self.DEVICES[0]
        #self.SEQ_LENGTH = self.SEQ_LENGTH if is_train else 1
        with tf.device(device):
            with tf.variable_scope(name):

                self.inImage = tf.placeholder(tf.float32, shape=[None] + Environment.RANGE_SHAPES[model_type][self.data_type])
                self.phase   = tf.placeholder(tf.bool, name='phase_%s' % self.data_type)
                self.in_batch_size = tf.placeholder(tf.int32)
                self.temp = tf.placeholder(tf.float32, shape=None)
                self.keep_per = tf.placeholder(tf.float32, shape=None)
                self.portfolio_state = tf.placeholder(tf.float32, shape=[None, Agent.STATE_DIM])  #Tensor("Placeholder_1:0", shape=(?, Agent.STATE_DIM), dtype=float32)
                #?,778,243,32
                '''
                关闭CNN
                
                self.conv1 = tf.contrib.layers.convolution2d(inputs=self.inImage, num_outputs=32, kernel_size=[14, 14],
                                                        padding="VALID", biases_initializer=None,trainable=is_train)
                self.batch_norm_1 = tf.contrib.layers.batch_norm(self.conv1, is_training=self.phase, scope='bn_1', center=True, scale=True)
                #?,388,121,32
                self.mp1 = tf.contrib.layers.max_pool2d(self.batch_norm_1, kernel_size=[3, 3],stride=3)
                # ?,386,119,32
                self.conv2 = tf.contrib.layers.convolution2d(inputs=self.mp1, num_outputs=32, kernel_size=[7, 7],
                                                        padding="VALID", biases_initializer=None,trainable=is_train)
                self.batch_norm_2 = tf.contrib.layers.batch_norm(self.conv2, is_training=self.phase, scope='bn_2', center=True,
                                                            scale=True)
                # ?,192,59,32
                self.mp2 = tf.contrib.layers.max_pool2d(self.batch_norm_2, kernel_size=[3, 3])

                self.conv3 = tf.contrib.layers.convolution2d(inputs=self.mp2, num_outputs=32, kernel_size=[3, 3],
                                                        padding="VALID", biases_initializer=None,trainable=is_train)
                self.batch_norm_3 = tf.contrib.layers.batch_norm(self.conv3, is_training=self.phase, scope='bn_3', center=True,
                                                            scale=True)
                self.mp3 = tf.contrib.layers.max_pool2d(self.batch_norm_3, kernel_size=[3, 3])
                self.conv4 = tf.contrib.layers.convolution2d(inputs=self.mp3, num_outputs=32, kernel_size=[3, 3],
                                                        padding="VALID", biases_initializer=None,trainable=is_train)
                self.batch_norm_4 = tf.contrib.layers.batch_norm(self.conv4, is_training=self.phase, scope='bn_4', center=True,
                                                            scale=True)
                self.mp4 = tf.contrib.layers.max_pool2d(self.batch_norm_4, kernel_size=[3, 3])

                self.conv5 = tf.contrib.layers.convolution2d(inputs=self.mp4, num_outputs=64, kernel_size=[3, 3],
                                                        padding="VALID", biases_initializer=None,trainable=is_train)
                self.batch_norm_5 = tf.contrib.layers.batch_norm(self.conv5, is_training=self.phase, scope='bn_5', center=True,
                                                            scale=True)  #（n,11，4，64）->(n,)

                
                # img_result = tf.reshape(mp4, [-1, mp4.get_shape().as_list()[0]])
                #img_result = tf.layers.flatten(batch_norm_5)
                img_result = tf.reshape( tf.transpose(self.batch_norm_5,perm=[0,2,1,3]),shape=[-1,self.batch_norm_5.shape[2],self.batch_norm_5.shape[1]*self.batch_norm_5.shape[3]])
                #（n，4，704）
                '''


                img_result = tf.reshape(tf.transpose(self.inImage, perm=[0, 2, 1, 3]), shape=[-1, self.inImage.shape[2],
                                                                                         self.inImage.shape[1] *
                                                                                         self.inImage.shape[
                                                                                             3]])  # #(n,W,H*通道数)

                lstms=[]
                for i in range(2):
                    lstm_cell = tf.contrib.rnn.LSTMCell(1024, state_is_tuple=True,trainable=is_train,name="LSTM_%d"%i)
                    lstms.append(lstm_cell)
                cell = tf.contrib.rnn.MultiRNNCell(lstms)  #多层LSTM

                lstm_outputs, lstm_state = tf.nn.dynamic_rnn(cell, img_result, dtype=tf.float32)   #inputs:[batch_size, max_time, size]#
                                                                            #output，[batch_size, max_time, num_units]如果time_major=False
                #tf.nn.bidirectional_dynamic_rnn双向
                #rnn_out = tf.reshape(lstm_outputs, [-1, 1024])[-1:]
                rnn_out = lstm_outputs[:,-1]  #(None,1024)

                batch_norm = tf.contrib.layers.batch_norm(rnn_out, is_training=self.phase, scope='bn', center=True,
                                                          scale=True)

                concat = tf.concat([batch_norm, self.portfolio_state], 1) # self.portfolio_state:(None,2)

                fc_layer_1 = slim.fully_connected(concat, 512, activation_fn=tf.nn.leaky_relu,
                                                  weights_initializer=normalized_columns_initializer(),
                                                  biases_initializer=None)

                dropout = slim.dropout(fc_layer_1, self.keep_per)

                self.streamAC, self.streamVC = tf.split(dropout, [256, 256], 1)

                self.streamA = tf.contrib.layers.flatten(self.streamAC)
                self.streamV = tf.contrib.layers.flatten(self.streamVC)

                self.AW = tf.Variable(tf.random_normal([256, Agent.NUM_ACTIONS]))
                self.VW = tf.Variable(tf.random_normal([256, 1]))

                self.Advantage = tf.matmul(self.streamA, self.AW)
                self.Value = tf.matmul(self.streamV, self.VW)
                # Q = V + A<-(A - A平均)
                # self.Qout = self.Value + tf.subtract(self.Advantage, tf.reduce_mean(self.Advantage, reduction_indices=1, keep_dims=True))
                # Q = V + A<-(A - Amax)
                self.Qout = self.Value + tf.subtract(self.Advantage,
                                                     tf.reduce_max(self.Advantage, reduction_indices=1, keep_dims=True))
                self.predict = tf.argmax(self.Qout, 1)
                self.Q_dist = tf.nn.softmax(self.Qout / self.temp)  # (self.Qout/self.temp)

                self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
                self.actions = tf.placeholder(shape=[None], dtype=tf.int32)

                self.actions_onehot = tf.one_hot(self.actions, depth=Agent.NUM_ACTIONS)

                self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), reduction_indices=1)

                self.td_error = tf.losses.huber_loss(labels=self.targetQ, predictions=self.Q)
                # self.td_error = tf.square(self.targetQ - self.Q)

                self.loss = tf.reduce_mean(self.td_error)

                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, name)
                with tf.control_dependencies(update_ops):
                    self.trainer = tf.train.AdamOptimizer(learning_rate=learning_rate)
                    self.updateModel = self.trainer.minimize(self.loss)

    def append_chartdata_in_batch(self, list, x_buffer, y_buffer, idx_buffer, last_idx_buffer, code_buffer, y_list, is_train = True):
        if len(list) > 0:
            chart_code=''
            while len(list) > 0:
                chart_code = list[0]
                del list[0]
                if self.chartcode_value[chart_code] >= 0:
                    try:
                        chart_data_y = np.genfromtxt(self.OUTPUT_DATA_PATH + chart_code + ".csv", delimiter=',')
                    except OSError:
                        chart_code = ''
                        continue
                    break
                else:
                    self.chartcode_list.remove(chart_code)
                    chart_code = ''
            if chart_code == "":
                self.has_data = False
                return
            if is_train:
                last_idx = 180 + int((chart_data_y.shape[0]) * 0.8)
                idx = 180
            else:
                idx = 180 + int((chart_data_y.shape[0]) * 0.8)
                last_idx = 179 + chart_data_y.shape[0]
            print('buffer load, code : %s, idx : %s, last_idx : %s reamining data : %s' % (chart_code, idx, last_idx,len(list)))

            input_x = self.get_data(idx, chart_code)
            if input_x is not None:
                input_x = [input_x]
            else:
                print('无数据')
                return None

            x_buffer.extend(input_x)
            y_buffer.append(chart_data_y)
            idx_buffer.append(idx)
            last_idx_buffer.append(last_idx)
            code_buffer.append(chart_code)
            y_list.append(chart_data_y[np.where(chart_data_y == idx)[0][0], 1])
        else:
            self.has_data = False

    def train(self,isTrain = False):

        #config.gpu_options.allow_growth = True
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            saver = tf.train.Saver(max_to_keep=1,reshape=True)
            if isTrain:
                print('Loading Model')
                ckpt = tf.train.get_checkpoint_state(self.MODEL_PATH[self.data_type])
                saver.restore(sess, ckpt.model_checkpoint_path)
            for epoch in range(1, TrendPredictNetwork.EPOCH_NUM + 1):
                print('EPOCH %s start' % epoch)
                list = deepcopy(self.chartcode_list)
                random.shuffle(list)
                list=list[:len(list)//self.DIVIVE_DATA]
                x_buffer = []
                y_buffer= []

                idx_buffer= []
                last_idx_buffer= []
                code_buffer= []
                self.has_data = True
                cnt = 0
                while True:
                    y_batch = []
                    del_list = []
                    #删除索引到达并添加新的图像数据
                    for i in range(len(idx_buffer)):
                        if idx_buffer[i] < last_idx_buffer[i]:
                            del x_buffer[i*self.SEQ_LENGTH]
                            x_buffer.insert(i*self.SEQ_LENGTH , self.get_data(idx_buffer[i],code_buffer[i]))
                            y_batch.append(y_buffer[i][np.where(y_buffer[i] == idx_buffer[i])[0][0], 1])
                            idx_buffer[i] += 1
                        else:
                            del_list.append(i)
                    if del_list:
                        del_list.reverse()
                        for i in del_list:
                            del x_buffer[i*self.SEQ_LENGTH: (i+1)*self.SEQ_LENGTH],y_buffer[i],idx_buffer[i],last_idx_buffer[i],code_buffer[i]

                    while self.has_data and len(idx_buffer) < self.BATCH_SIZE:
                        self.append_chartdata_in_batch(list,x_buffer,y_buffer,idx_buffer,last_idx_buffer,code_buffer,y_batch)
                    if len(y_batch) <= 0:
                        break

                    cost, _ = sess.run([self.cost, self.optimizer], feed_dict={
                        self.inImage: x_buffer, self.out: y_batch, self.in_batch_size: len(y_batch), self.phase : True })
                    cnt += 1
                    if cnt % 500 == 0 :
                        print('%s epoch %s times mean error : %s' % (epoch, cnt, cost))


                saver.save(sess, self.MODEL_PATH[self.data_type]+'/model_Predict.cptk')
            if epoch % 300 == 0:
                self.validation(sess,epoch)
    def validation(self, sess, epoch):
        print('start %s epoch validation' % epoch)
        cnt = 0
        correct_cnt = 0
        list = deepcopy(self.chartcode_list)
        random.shuffle(list)
        list=list[:len(list)//self.DIVIVE_DATA]
        x_buffer = []
        y_buffer = []

        idx_buffer = []
        last_idx_buffer = []
        code_buffer = []
        self.has_data = True

        while True:
            y_batch = []
            del_list = []

            for i in range(len(idx_buffer)):
                if idx_buffer[i] < last_idx_buffer[i]:
                    del x_buffer[i * self.SEQ_LENGTH]
                    x_buffer.insert(i * self.SEQ_LENGTH, self.get_data(idx_buffer[i], code_buffer[i]))
                    y_batch.append(y_buffer[i][np.where(y_buffer[i] == idx_buffer[i])[0][0], 1])
                    idx_buffer[i] += 1
                else:
                    del_list.append(i)
            if del_list:
                del_list.reverse()
                for i in del_list:
                    del x_buffer[i * self.SEQ_LENGTH: (i + 1) * self.SEQ_LENGTH], y_buffer[i], idx_buffer[i], last_idx_buffer[i], code_buffer[i]

            while self.has_data and len(idx_buffer) < self.BATCH_SIZE:
                self.append_chartdata_in_batch(list, x_buffer, y_buffer, idx_buffer, last_idx_buffer,
                                               code_buffer, y_batch, is_train=False)
            if len(y_batch) <= 0:
                break

            p_y_list = sess.run(self.predict, feed_dict={self.inImage: x_buffer, self.in_batch_size: len(y_batch), self.phase : False})

            for i, p_y in enumerate(p_y_list):
                cnt += 1
                if (y_batch[i] == p_y):
                    correct_cnt += 1

            print('epoch %s 准确度 : %s (%s/%s)' % (epoch, ((correct_cnt / cnt) * 100), correct_cnt, cnt))

    def predict(self):
        pass
    def load_model(self):
        pass

    def get_data(self, idx, code):
        return self.get_image(code, idx, self.model_type)
    def get_image(self, code, idx, days=20):
        filepath = self.INPUT_IMAGE_PATH[self.data_type] + "%s_%s_%s.png" % (code, days, idx)
        try:
            with pilimg.open(filepath) as im_file:
                im = np.asarray(im_file)

                xPixel = Environment.RANGE_SHAPES[days][self.data_type][1]
                yPixel = Environment.RANGE_SHAPES[days][self.data_type][0]
                if im.shape[1] < xPixel:
                    while True:
                        try:
                            xarray = np.array([[[255, 255, 255, 255]] * (xPixel - im.shape[1])] * im.shape[0])
                            im = np.hstack([im, xarray])
                            break
                        except:
                            print('y', xarray.shape, im.shape)
                    del xarray
                elif im.shape[1] > xPixel:
                    im=im[:,:xPixel]

                if im.shape[0] < yPixel:
                    while True:
                        try:
                            yarray = np.array([[[255, 255, 255, 255]] * (im.shape[1])] * (yPixel - im.shape[0]))
                            im = np.vstack([im, yarray])
                            break
                        except:
                            print('y', yarray.shape, im.shape)
                    del yarray
                elif im.shape[0] > yPixel:
                    im = im[:yPixel]

            return im
        except:
            return None
    # 测试代码
    def test(self):
        #config.gpu_options.allow_growth = True
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            saver = tf.train.Saver(max_to_keep=1,reshape=True)

            print('Loading Model')
            ckpt = tf.train.get_checkpoint_state(self.MODEL_PATH[self.data_type])
            saver.restore(sess, ckpt.model_checkpoint_path)
            self.validation(sess,0)

import sys
class experience_buffer():  #存储器 ，140
    def __init__(self, buffer_size=140):
        self.buffer = []
        self.reward_buffer=[]
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) + 1 >= self.buffer_size:
            #self.buffer[0:1] = []
            del self.buffer[0]

        self.buffer.append(experience)

        #print(sys.getsizeof(self.buffer))

    def sample(self, size, reward_list = None):
        if reward_list:
            reward_list = np.array(reward_list,dtype=np.float32)
            reward_list += abs(np.min(reward_list))+1
            sum_r =np.sum(reward_list)
            reward_list /= sum_r
            idx = np.random.choice(len(self.buffer),1,p=reward_list)
            episode = self.buffer[idx[0]]
        else:
        #return np.reshape(np.array(random.sample(self.buffer, size)), [size, 5])
            episode = random.sample(self.buffer, 1)[0]  #从buffer中获取1个元素
            #print("----------{}".format(episode[0][0])) #len(episode):30 因为设置了30为一迭代。 len(episode[0]):7 /s*图片数+s_*图片数+5,其中s是图片的合并[basic,sto,sso,qinggan]
        size = min(len(episode), size) #size=20
        buffer = random.randint(0, len(episode) - size) #0-10随机数
        result = np.array(episode[buffer: buffer + size]), size  #len(result[0]):20   shape=(20,7)
        #print(result[0][:,3])
        return result

def updateTargetGraph(tfVars, tau):
    total_vars = len(tfVars)

    op_holder = []

    for idx, var in enumerate(tfVars[0:int(total_vars / 2)]):
        op_holder.append(tfVars[int(idx) + int(total_vars / 2)].assign(
            (var.value() * tau) + ((1 - tau) * tfVars[int(idx) + int(total_vars / 2)].value())))
    return op_holder

def updateTarget(op_holder, sess):
    for op in op_holder:
        sess.run(op)


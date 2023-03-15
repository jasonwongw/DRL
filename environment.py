import numpy as np
import random
import draw as cd
import PIL.Image as pilimg
import os
import json

class Environment:

    chartcode_list = [
        "000060",
    ]

    chartcode_value={
        "000060":1,
    }

    PRICE_IDX = 1
    #candle stick file path
    #FILE_PATH =
    FILE_PATH = ['./a/chart_images_other_basic/', './a/chart_images_other_BB/', './a/chart_images_other_MACD/', './a/chart_images_other_OBV/', './a/chart_images_other_DMI/', './a/chart_images_other_STO/','./a/chart_images_other/','./a/chart_images_other_KOSPI/','./a/chart_images_other_qinggan/']
    FILE_PATH_TEST = ['./a/chart_images_other_basic_test/', './a/chart_images_other_BB_test/', './a/chart_images_other_MACD_test/',
                 './a/chart_images_other_OBV_test/', './a/chart_images_other_DMI_test/', './a/chart_images_other_STO_test/',
                 './a/chart_images_other_test/','./a/chart_images_other_KOSPI_test/','./a/chart_images_other_qinggan_test/']

    FILE_TYPE = [FILE_PATH, FILE_PATH_TEST]

    TYPE_BASIC = 0
    TYPE_BB = 1
    TYPE_MACD = 2
    TYPE_OBV = 3
    TYPE_DMI = 4
    TYPE_STO = 5
    TYPE_ORIGIN = 6

    TYPE_QINGGAN=8
    #RANGE_SHAPE = {5: [630, 130, 4], 20: [630, 245, 4], 60: [630, 550, 4], 6:[630,130,4], 7:[630,130,4], 25: [630, 280, 4]}
    RANGE_SHAPE = {20: [630, 245, 3], 5: [630, 130, 4],  60: [630, 550, 4], 6: [630, 130, 4], 7: [630, 130, 4],
                   25: [630, 280, 3]}
    RANGE_SHAPES = {20: [ [400, 240, 3], [320,235,3],[320,245,3],[320,250,3],[320,215,3],[320,215,3],[630, 245, 3],[320,230,3],[320,215,3] ]  }
    Range_shapes={20:[[20,14]]}

    #idx = 204  #控制idx不能<=2500
    def __init__(self):
        self.stockcode_len = len(Environment.chartcode_list) #1

        '''
        with open('./value_chart.txt', 'r') as f:
            lines = f.readlines()
            data = ''
            for line in lines:
                data += line
            self.chartcode_value = json.loads(data)
            print('value ')'''
    def reset(self, code=None):
        if code is None:   #code是划分测试与训练集
            #self.idx=179
            while True:
                while True:
                    value_list = np.array(list(self.chartcode_value.values()), dtype=np.float32)   #[1.]
                    value_list += abs(np.min(value_list))+1
                    sum_r = np.sum(value_list)
                    value_list /= sum_r
                    self.chart_code = np.random.choice(self.chartcode_list, 1)[0]#, p=value_list)[0]    000060
                    #测试
                    #self.chart_data = np.genfromtxt("./chart_data_test/" + self.chart_code + "_1.csv", delimiter=',')
                    #训练
                    self.chart_data = np.genfromtxt("./chart_data/" + self.chart_code + "_1.csv", delimiter=',') #（2500.13）
                    if self.chart_data.shape[0] > 200:
                        self.chart_data = np.flip(self.chart_data, 0)  # (2500,13)矩阵翻转，头->尾  尾->头
                        break  #行数大于200就要，少于200就del
                    del self.chart_data
                #print(self.chart_code , self.chart_data.shape[1])  #000060 2500
                # 入口点收盘价数据和状态值。
                #self.idx+=1  #从477开始又跳到180
                self.idx = random.randint(180, 2470)
                #self.idx = random.randint(180, 2432-30)#self.chart_data.shape[0] - 30)   #(180,2470)  这里要改，不能弄成随机的, 为啥是2470，因为还要+30
                #if self.chart_data[self.idx, self.PRICE_IDX] < 10000:  #剔除开盘价少于10000的，即删除了475行左右
                    #continue  #continue指向while True:   continue只要满足if语句，跳过当前while true，不执行下面语句，重新循环
                # self.observation = self.chart_data[self.idx]

                self.idx_end = self.chart_data.shape[0]  #2500
                self.file_path = self.FILE_TYPE[0]  #地址

                break  #只要有一个满足开盘价大于10000的，跳出函数reset
        else:
            self.chart_code = code  #单独的股票代码测试
            try:
                self.chart_data = np.genfromtxt("./chart_data_test/" + self.chart_code + "_1.csv", delimiter=',')

            except:
                return False

            self.chart_data = np.flip(self.chart_data, 0)  #矩阵翻转
            self.file_path = self.FILE_TYPE[1]  #test目录

            self.idx =np.where(self.chart_data[:,0] == 20190401)[0][0]  #204
            #self.idx = np.where(self.chart_data[:, 0] == 20210104)[0][0]  #147

            #print(self.idx)
            #self.idx = np.where(self.chart_data[:, 0] == 20190502)[0][0]  #测试 5月份开始  226
            self.idx_end = self.idx + 242 #21
            #self.idx = 180
            if self.chart_data.shape[0] < 180:
                return False

            return True
    def step(self):
        self.idx += 1

        # 趋势计数

        return False if self.idx + 1 < self.idx_end else True   #


    def get_image(self, days=1,type=0):  #type=0,4,5
        #装入/组合图像

        filepath = self.file_path[type] + "%s_%s_%s.jpg" % (self.chart_code, days, self.idx)

        if not os.path.isfile(filepath): #用于判断某一对象(需提供绝对路径)是否为文件
            cd.data_generate(self.chart_data, range=days, start_idx=self.idx, title=filepath,type = type)
        print(filepath)
        is_changed = False
        with pilimg.open(filepath) as im_file:  #打开图片，channel的通道顺序为RGB
            im = np.asarray(im_file)  #(400, 240, 3)   (320, 215, 3)   (320, 215, 3)  ，图片->数组

            #固定图像大小，
            #添加水平像素
            #xPixel = self.RANGE_SHAPE[days][1]
            #yPixel = self.RANGE_SHAPE[days][0]
            xPixel = self.RANGE_SHAPES[days][type][1]   #240 ,215,215
            yPixel = self.RANGE_SHAPES[days][type][0]   #400,320,320
            if im.shape[1] < xPixel:
                while True:
                    try:
                        #xarray = np.array([[[255, 255, 255, 255]] * (xPixel - im.shape[1])] * im.shape[0],dtype=np.uint8)
                        xarray = np.array([[[255, 255, 255]] * (xPixel - im.shape[1])] * im.shape[0],
                                          dtype=np.uint8)
                        im = np.hstack([im, xarray])
                        break
                    except:
                        print('y无法添加轴图像', xarray.shape, im.shape)
                        sleep(1)
                del xarray
                is_changed=True
            elif im.shape[1] > xPixel:
                im=im[:,:xPixel]
                is_changed = True
            # 添加垂直像素
            if im.shape[0] < yPixel:
                while True:
                    try:
                        #yarray = np.array([[[255, 255, 255, 255]] * (im.shape[1])] * (yPixel - im.shape[0]),dtype=np.uint8)
                        yarray = np.array([[[255, 255, 255]] * (im.shape[1])] * (yPixel - im.shape[0]),
                                          dtype=np.uint8)
                        im = np.vstack([im, yarray])
                        break
                    except:
                        print('无法添加y轴图像', yarray.shape, im.shape)
                        sleep(1)
                del yarray
                is_changed = True
            elif im.shape[0] > yPixel:
                im = im[:yPixel]
                is_changed = True

        if is_changed:
            im_data = pilimg.fromarray(im,mode="RGB")  #array转换成image
            im_data.save(filepath, "JPEG")
            print('%s 完成转换' % filepath)
        
        return im

    def get_price(self, next=0, range=0):
        #return np.mean(self.chart_data[(self.idx + next+1)-range:(self.idx + next + 1), Environment.PRICE_IDX])
        return self.chart_data[self.idx + next, Environment.PRICE_IDX]



if __name__ == "__main__":
    from time import sleep
    obj = Environment()
    network_type = [20]
    chart_type =[Environment.TYPE_BASIC,Environment.TYPE_DMI,Environment.TYPE_STO]   #0,4,5

    while True:
    #code = '000060'#002220 20 2035
        #print("%s 正在创建图像" % code)
        end = False
        e = obj.reset('000060')  #"code='000060:测试数据集'",#None,调用Environment()类的reset函数  ，obj是对象
        #if e:
        while not end:
            a =[obj.get_image(days,type) for type in chart_type   #self.idx每次抽取一个，一次循环
                                     for days in network_type]  #先循环后面的type，在循环前面的type
            end = obj.step() #self.idx+=1,只要idx<2500，就继续while not end循环
            #sleep(0.05)
        #跳出循环
        # if obj.idx>=2499:
        #     break
    print('创建完成')



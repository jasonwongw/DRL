import pyltp
from pyltp import Segmentor
from pyltp import SentenceSplitter
from pyltp import Postagger
import numpy as np
import os

import pandas as pd
from pandas import DataFrame
import time

# 读取文件，文件读取函数
def read_file(filename):
    sentences = pd.read_csv(filename, header=None,encoding="GBK")
    sentences = sentences[0].values
    sentences = list(sentences)
    return sentences

def read_file1(filename):   #分句子
    with  open(filename, 'r', encoding='utf-8') as f:
        text1 = f.read()
        # 返回list类型数据
        text1 = text1.split('\n')
        #print(text1)
    return text1

# 将数据写入文件中
def write_data(filename, data):
    with open(filename, 'a', encoding='utf-8') as f:
        f.write(str(data))

#文本分句
def cut_sentence(text):
    sentences = SentenceSplitter.split(text) #['句子。','句子？','句子！']  遇到 结束的标点符号 就分割
    #print(sentences)
    sentence_list = [w for w in sentences]
    #print(sentence_list)
    return sentence_list


# 文本分词
def tokenize(sentence):  #大句子中的每一小句
    # 加载模型
    LTP_DATA_DIR = 'C:/Users/Administrator/Desktop/bishe/lunwen'  # ltp模型目录的路径
    cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')  # 分词模型

    segmentor = Segmentor(cws_model_path)  # 初始化实例  加载模型
    # 加载模型
    #segmentor.load(r'F:\DeapLearning\androidproject\lunwen\cws.model')
    # 产生分词，segment分词函数
    words = segmentor.segment(sentence)   #对每一小句进行分词 ['','',...'。']
    #print(words)
    words = list(words)
    #print(words)
    # 释放模型
    segmentor.release()
    return words


# 词性标注
def postagger(words):
    # 初始化实例
    postagger = Postagger()
    # 加载模型
    postagger.load(r'C:/Users/Administrator/Desktop/bishe/lunwen/pos.model')
    # 词性标注
    postags = postagger.postag(words)
    # 释放模型
    postagger.release()
    # 返回list
    postags = [i for i in postags]
    return postags


# 分词，词性标注，词和词性构成一个元组
def intergrad_word(words, postags):
    # 拉链算法，两两匹配
    pos_list = zip(words, postags)
    pos_list = [w for w in pos_list]
    return pos_list


# 去停用词函数
def del_stopwords(words):
    # 读取停用词表
    stopwords = read_file1(r"C:/Users/Administrator/Desktop/bishe/lunwen/stopwords.txt")
    #print(stopwords) #['———', '》），', '）÷（１－', '”，', '）、', '＝（', ':', '→', '℃ ', '&', '*',...]
    # 去除停用词后的句子
    new_words = []
    for word in words:
        if word not in stopwords:
            new_words.append(word)
    return new_words    #删除了在停用词词典的词


# 获取六种权值的词，根据要求返回list，这个函数是为了配合Django的views下的函数使用
def weighted_value(request):
    result_dict = []
    if request == "one":
        result_dict = read_file1(r"C:/Users/Administrator/Desktop/bishe/lunwen/most.txt")
    elif request == "two":
        result_dict = read_file1(r"C:/Users/Administrator/Desktop/bishe/lunwen/very.txt")
    elif request == "three":
        result_dict = read_file1(r"C:/Users/Administrator/Desktop/bishe/lunwen/more.txt")
    elif request == "four":
        result_dict = read_file1(r"C:/Users/Administrator/Desktop/bishe/lunwen/ish.txt")
    elif request == "five":
        result_dict = read_file1(r"C:/Users/Administrator/Desktop/bishe/lunwen/insufficiently.txt")
    elif request == "six":
        result_dict = read_file1(r"C:/Users/Administrator/Desktop/bishe/lunwen/inverse.txt")
    elif request == 'posdict':
        result_dict = read_file1(r"C:/Users/Administrator/Desktop/bishe/lunwen/posdict.txt")
    elif request == 'negdict':
        result_dict = read_file1(r"C:/Users/Administrator/Desktop/bishe/lunwen/negdict.txt")
    else:
        pass
    return result_dict


print("reading sentiment dict .......")
# 读取情感词典
posdict = weighted_value('posdict')
print(posdict)
negdict = weighted_value('negdict')
print(negdict)
# 读取程度副词词典
# 权值为2
mostdict = weighted_value('one')
#print(mostdict)
# 权值为1.75
verydict = weighted_value('two')
# 权值为1.50
moredict = weighted_value('three')
# 权值为1.25
ishdict = weighted_value('four')
# 权值为0.25
insufficientdict = weighted_value('five')
# 权值为-1
inversedict = weighted_value('six')


# 程度副词处理，对不同的程度副词给予不同的权重
def match_adverb(word, sentiment_value):
    # 最高级权重为
    if word in mostdict:
        sentiment_value *= 8
    # 比较级权重
    elif word in verydict:
        sentiment_value *= 6
    # 比较级权重
    elif word in moredict:
        sentiment_value *= 4
    # 轻微程度词权重
    elif word in ishdict:
        sentiment_value *= 2
    # 相对程度词权重
    elif word in insufficientdict:
        sentiment_value *= 0.5
    # 否定词权重
    elif word in inversedict:
        sentiment_value *= -1
    else:
        sentiment_value *= 1
    return sentiment_value


# 对每一条微博打分
def single_sentiment_score(text_sent):   #对每一句句子操作 #[' 。 ？  ！']
    sentiment_scores = []
    # 对单条微博分句
    sentences = cut_sentence(text_sent)  #['句子。','句子？','句子！']
    for sent in sentences:
        # 查看分句结果
        # print('分句：',sent)
        # 分词
        words = tokenize(sent)  #['','',...'。']
        seg_words = del_stopwords(words)  #每一小句词典删除了停用词
        # i，s 记录情感词和程度词出现的位置
        i = 0  # 记录扫描到的词位子
        s = 0  # 记录情感词的位置
        poscount = 0  # 记录积极情感词数目
        negcount = 0  # 记录消极情感词数目
        # 逐个查找情感词
        for word in seg_words:
            # 如果为积极词
            if word in posdict:
                poscount += 1  # 情感词数目加1
                # 在情感词前面寻找程度副词
                for w in seg_words[s:i]: #？
                    poscount = match_adverb(w, poscount)  #对程度副词进行操作，
                s = i + 1  # 记录情感词位置
            # 如果是消极情感词
            elif word in negdict:
                negcount += 1
                for w in seg_words[s:i]:
                    negcount = match_adverb(w, negcount)
                s = i + 1
            # 如果结尾为感叹号或者问号，表示句子结束，并且倒序查找感叹号前的情感词，权重+4
            elif word == '!' or word == '！' or word == '?' or word == '？':
                for w2 in seg_words[::-1]: #从后往前循环
                    # 如果为积极词，poscount+2
                    if w2 in posdict:
                        poscount += 4
                        break
                    # 如果是消极词，negcount+2
                    elif w2 in negdict:
                        negcount += 4
                        break
            i += 1  # 定位情感词的位置

        #----------------------------------------------------------------------------------------------------------------------
        # 计算情感值
        sentiment_score = poscount - negcount
        sentiment_scores.append(sentiment_score) #[ 分值, , ,...]
        #print(sentiment_scores)
        # 查看每一句的情感值
        # print('分句分值：',sentiment_score)
    sentiment_sum = 0
    for s in sentiment_scores:
        # 计算出一条微博的总得分
        sentiment_sum += s
    return sentiment_sum


# 分析test_data.txt 中的所有微博，返回一个列表，列表中元素为（分值，微博）元组
def run_score(contents):
    # 待处理数据
    scores_list = []
    #计数
    jishu=0
    print(len(contents))
    #计时
    start = time.time()
    #每10000句保存一次
    juzi=0

    for content in contents:  #一个句子一个句子的来
        #计数,计时
        jishu+=1
        juzi+=1
        if jishu % 300 == 0:
            end = time.time()
            print(end-start)
            print(jishu)
        #非空句子
        if content != '':
            score = single_sentiment_score(content)  # 对每条微博调用函数求得打分
            scores_list.append((score, content))  # 形成（分数，微博）元组  [(分值,‘大句子’),(),...]
            #print(scores_list)

            #每次句子输出结果都保存，追加保存到csv文件
            data1=[(content,
                    score)]
            data2=DataFrame(data1)
            #DataFrame每次都会有列名和索引，因此使用header=False, index=False
            data2.to_csv('C:/Users/Administrator/Desktop/bishe/lunwen/shuju/1.csv', header=False, index=False, mode='a+')

        #每5000次保存txt文件
        # zhi=[]
        # wen=[]
        # for s in scores_list:
        #     fenzhi.append(score[0])
        #     wenben.append(score[1])
        # if juzi % 5000 ==0:
        #     with open("C:/Users/Administrator/Desktop/bishe/lunwen/shuju/{}.txt".format(juzi), "w",encoding = "UTF-8") as f:
        #         f.write(str(scores_list))  # 这句话自带文件关闭功能，不需要再写f.close()

    return scores_list


# 主程序
if __name__ == '__main__':
    print('Processing........')
    # 测试
    # sentence = '要怎么说呢! 我需要的恋爱不是现在的样子, 希望是能互相鼓励的勉励, 你现在的样子让我觉得很困惑。 你到底能不能陪我一直走下去, 你是否有决心?是否你看不惯我?你是可以随意的生活,但是我的未来我耽误不起！'
    # sentence = '转有用吗？这个事本来就是要全社会共同努力的，公交公司有没有培训到位？公交车上地铁站内有没有放足够的宣传标语？我现在转一下微博，没有多大的意义。'

    #sentences = read_file1(r'F:\DeapLearning\androidproject\lunwen\cs.txt') #分句子    格式：['句子','句子']
    sentences = read_file("C:/Users/Administrator/Desktop/1.csv")

    #print(sentences)
    ##要注意代码运行顺序，这里是最后面了。这行代码是运行句子，
    scores = run_score(sentences)  #[(分值,‘大句子’),( ,' '),...]
    # 人工标注情感词典
    #man_sentiment = read_file(r'test_data\人工情感标注.txt')

    #------------------------------------------------------------------------------------------------------------
    al_sentiment = []
    for score in scores:
        #print('情感分值：', score[0])
        if score[0] < 0:
            #print('情感倾向：消极')
            s = '消极'
        elif score[0] == 0:
            #print('情感倾向：中性')
            s = '中性'
        else:
            #print('情感倾向：积极')
            s = '积极'
        al_sentiment.append(s)
        #print('情感分析文本：', score[1])
    i = 0
    # 写入文件文本中
    #filename = r'F:\DeapLearning\androidproject\lunwen\result_data.txt'

    fenzhi=[]
    wenben=[]
    biaozhu=[]


    for score in scores:
        # 写出excel
        fenzhi.append(score[0])
        wenben.append(score[1])
        biaozhu.append(al_sentiment[i])
        i=i+1

        #要注意代码运行顺序，这里是最后面了
        # if cishu % 5 ==0:
        #     data1 = {
        #         '情感分析文本': wenben,
        #         '情感分值': fenzhi,
        #         '机器情感标注': biaozhu
        #     }
        #     df1 = DataFrame(data1)
        #     print('这里')
        #     # print(data)
        #     df1.to_excel('C:/Users/Administrator/Desktop/bishe/lunwen/{}.xlsx'.format(cishu))

        #写出文本
        # write_data(filename, '情感分析文本：{}'.format(str(score[1])) + '\n')  # 写入情感分析文本
        # write_data(filename, '情感分值：{}'.format(str(score[0])) + '\n')  # 写入情感分值
        # #write_data(filename, '人工标注情感：{}'.format(str(man_sentiment[i])) + '\n')  # 写入人工标注情感
        # write_data(filename, '机器情感标注：{}'.format(str(al_sentiment[i])) + '\n')  # 写入机器情感标注
        # write_data(filename, '\n')

    data = {
            '情感分析文本': wenben,
            '情感分值': fenzhi,
            '机器情感标注': biaozhu
        }
    df=DataFrame(data)
    #print(data)
    df.to_excel('C:/Users/Administrator/Desktop/bishe/lunwen/1.xlsx')
    print('succeed.......')


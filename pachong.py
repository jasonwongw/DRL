import requests,os
from bs4 import BeautifulSoup
import pandas as pd
import time
from fake_useragent import UserAgent
#header={'Content-Type':'text/html;charset=utf-8','User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/105.0.0.0 Safari/537.36'}
max_page=1444
#ua = UserAgent()
location = os.getcwd() + '/fake_useragent_0.1.11.json'
ua = UserAgent(path=location) #cache_path
# all_title = []   #爬取的标题存储列表
# all_time  = []   #爬取的发表时间储存列表
for page in range(953,max_page+1):
    print('crawling the page is {}'.format(page))
    url = f'http://guba.eastmoney.com/list,600887_{page}.html'
    headers = {'Content-Type':'text/html;charset=utf-8','User-Agent': ua.random}
    html = requests.get(url, headers=headers)
    soup = BeautifulSoup(html.content, 'lxml')
    # 阅读数
    read_counts = soup.find_all('span', attrs={'class': 'l1 a1'})
    # 评论数
    #comment_counts = soup.find_all('span', attrs={'class': 'l2 a2'})
    # 标题数
    title_counts = soup.find_all('span', attrs={'class': 'l3 a3'})#find_all()返回的是所有匹配结果的列表
    # 作者
    #author_counts = soup.find_all('span', attrs={'class': 'l4 a4'})
    # 时间
    time_counts = soup.find_all('span', attrs={'class': 'l5 a5'})
    for i in range(len(read_counts) - 1):
        data1 = [(read_counts[i + 1].string,
                  #comment_counts[i + 1].string,
                  title_counts[i + 1].find(name='a').get('title'),
                  #author_counts[i + 1].find(name='font').string,
                  time_counts[i + 1].string)]#find()返回的是第一个匹配的标签结果
        data2 = pd.DataFrame(data1)
        data2.to_csv('C://Users//Administrator//Desktop//wenben2600887.csv', header=False, index=False, mode='a+')
        #a+ : 可读可写, 可以不存在, 必不能修改原有内容, 只能在结尾追加写, 文件指针只对读有效 (写操作会将文件指针移动到文件尾)
    time.sleep(1)

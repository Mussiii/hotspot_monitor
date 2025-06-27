import requests, torch, torchtext
from bs4 import BeautifulSoup
import jieba.analyse
import sqlite3
import torch.nn as nn, torch.nn.functional as F
import pandas as pd
import jieba
from torchtext.vocab import build_vocab_from_iterator
import schedule
import time
from flask import Flask, jsonify, send_file, render_template,request
import datetime as dt
from flask_cors import CORS
import random
from collections import defaultdict
import string
import json
from datetime import datetime ,timedelta
from apscheduler.schedulers.background import BackgroundScheduler
import requests
import math



app = Flask(__name__)
CORS(app)




# 这是一些人工智能判断积极消极会用到的方法和类
def pre_text(text):
    text = text.replace('！', '').replace('，', '').replace('。', '')
    return jieba.lcut(text)


def yield_tokens(data):
    for text in data:
        yield text





# 这个参考代码使用双向 LSTM， 注意 self.fc1 定义中 hidden_size 乘以 2 。
class BIRNN_Net(nn.Module):
    def __init__(self, vocab_size, embeding_dim=100, hidden_size=200):
        super(BIRNN_Net, self).__init__()
        self.em = nn.Embedding(vocab_size, embeding_dim)
        self.rnn = nn.LSTM(embeding_dim, hidden_size, bidirectional=True)
        self.fc1 = nn.Linear(hidden_size * 2, 64)
        # self.fc2 = nn.Linear(64, 2)
        self.fc2 = nn.Linear(64, 3)
        
    def forward(self, inputs):
        x = self.em(inputs)
        x = F.dropout(x)
        x, _ = self.rnn(x)
        x = F.dropout(F.relu(self.fc1(x[-1])))
        x = self.fc2(x)
        return x


#爬取的热度数据处理
def normalize_hot(hot_list):
    normalized = []
    for item in hot_list:
        try:
            # 清理空格和非字符串直接跳过
            if not isinstance(item, str):
                item = str(item)

            item = item.strip()

            # 如果是形如 "盛典 289531"，保留数字部分，忽略前面文字
            parts = item.split()
            if len(parts) == 2 and parts[1].isdigit():
                num = int(parts[1])
                normalized.append(f"{num / 10000:.1f}万")

            # 如果是纯数字
            elif item.isdigit():
                num = int(item)
                normalized.append(f"{num / 10000:.1f}万")

            # 如果是已是“xx.x万”格式
            elif '万' in item and any(char.isdigit() for char in item):
                normalized.append(item)

            # 其他情况（如空字符串、异常格式等）
            else:
                normalized.append('0.0万')  # 可按需改为保留原始 item

        except:
            normalized.append('0.0万')  # 异常保护

    return normalized


# 这个方法用来处理hot的数据清洗
def process_data(data):
    processed_data = []
    prev_value = None

    for item in data:
        if item:
            if '万' in item:  # 处理第一种类型的数据
                value = float(item.replace('万', '')) * 10000
            else:  # 处理第二种类型的数据
                value = float(item)

            processed_data.append(value)
            prev_value = value
        else:  # 处理空数据
            if prev_value is not None:
                if len(processed_data) > 1:  # 至少有一个前面有值
                    avg_value = (prev_value + processed_data[-1]) / 2
                    processed_data.append(avg_value)
                else:
                    processed_data.append(prev_value)

    return processed_data
    
def get_numeric_date():
    now = dt.datetime.now()
    return now.strftime("%m%d%H%M")

def get_times():
    now = dt.datetime.now()
    return now.strftime("%m%d")

def get_hourtime():
    now = dt.datetime.now()
    return now.strftime("%m%d%H")



@app.route('/flush_data/')
def get_data_back():
    import pandas as pd
    import jieba

    # 定义字段名
    title_name = 'title'
    title = []
    url_name = 'url'
    url = []
    source_name = 'source'
    source = []

    keyword_name = 'keyword'
    keyword = []

    prediction_name = 'prediction'
    prediction = []

    prediction_value_name = 'predictionn_value'
    prediction_value = []

    predicotion_good_point_name = 'prediction_good_point'
    prediction_good_point = []

    predicotion_bad_point_name = 'prediction_bad_point'
    prediction_bad_point = []

    predicotion_mid_point_name = 'prediction_bad_point'
    prediction_mid_point = []

    hot_name = 'hot'
    hot = []

    # 数据源链接
    url_baidu = 'https://tophub.today/n/Jb0vmloB1G'
    url_weibo = 'https://tophub.today/n/KqndgxeLl9'
    url_base = 'https://tophub.today'

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36'
    }

    response_baidu = requests.get(url_baidu, headers=headers)
    response_weibo = requests.get(url_weibo, headers=headers)

    def extract_data(html, origin):
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')
        table = soup.find("table")
        if not table:
            print(f"[{origin}] 无法找到表格")
            return
        rows = table.find_all("tr")[1:]  # 忽略表头

        for row in rows:
            tds = row.find_all("td")
            if len(tds) < 3:
                continue  # 数据不足则跳过
            a_tag = tds[1].find("a")
            if not a_tag:
                continue  # 无超链接则跳过
            t = a_tag.text.strip()
            u = url_base + a_tag.get("href", "").strip()
            h = tds[2].text.strip()
            title.append(t)
            url.append(u)
            source.append(origin)
            hot.append(h)

    if response_baidu.ok:
        extract_data(response_baidu.text, "baidu")
    if response_weibo.ok:
        extract_data(response_weibo.text, "weibo")

    # 打印前20条检查
    # print("===== 前20个 title =====")
    # print(title[:20])
    # print("===== 前20个 url =====")
    # print(url[:20])
    # print("===== 前20个 source =====")
    # print(source[:20])
    # print("===== 前20个 hot =====")
    # print(hot[:20])
    # print("===== 后20个 title =====")
    # print(title[-20:])
    # print(hot[-20:])

    # ===  清洗 hot 数据 ===
    hot = normalize_hot(hot)
    hot = process_data(hot)

    # ===  创建 DataFrame ===
    dataframe = pd.DataFrame({
        title_name: title,
        url_name: url,
        source_name: source,
        hot_name: hot
    })

    # 删除 title 为 None 的行
    dataframe = dataframe[dataframe[title_name].notna()]

    # === keyword 关键词提取 ===
    ##手动提取标签
    titles = dataframe['title']
    ## 使用中文分词提取关键字，并保存到列表中
    for title in titles:
        ## 使用 jieba 进行分词
        seg_list = jieba.lcut(title)
        ##选择前三个关键字
        keywords = seg_list[:3]
        ## 将分词后的结果添加到关键字列表中
        keyword.append(keywords)
    ## 将关键字列表添加到 DataFrame 中作为新的列
    dataframe['keyword'] = keyword
    ## 将包含三个元素的数组拆分成三列
    dataframe[['keyword1', 'keyword2', 'keyword3']] = dataframe['keyword'].apply(pd.Series)
    ## 删除原始的 "keyword" 列
    dataframe.drop(columns=['keyword'], inplace=True)




    # review的获取
    ##模型的初始化
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = pd.read_csv('data.csv')
    data['review'] = data.review.apply(pre_text)
    vocab = build_vocab_from_iterator(iterator=yield_tokens(data.review), specials=["<pad>", "<unk>"], min_freq=2)
    vocab.set_default_index(vocab["<unk>"])
    vocab_size = len(vocab)  # 1560
    ##print(vocab_size)
    embedding_dim = 100
    hidden_size = 200
    model = BIRNN_Net(vocab_size, embedding_dim, hidden_size).to(device)

    #这行代码后面的map参数是因为没有cuda，如果有可以删除掉，注意和上面的device统一
    model = torch.load('save_path2',map_location=device)
    model.eval()
    ##打标签
    for title in dataframe.iloc[:, 0]:  # 标题位于第一列
        title_copy = title
        processed_text = pre_text(title_copy)
        input_tensor = torch.tensor([vocab[token] for token in processed_text], dtype=torch.int64).unsqueeze(1).to(
            device)

        with torch.no_grad():
            predicted_label = model(input_tensor)
            predicted_probabilities = F.softmax(predicted_label, dim=1)
            predicted_class = predicted_probabilities.argmax(dim=1).item()  # 获取概率最高的类别
            predicted_value = torch.max(predicted_probabilities).item()

            #三种打分的获取
            ##这里打分可能是负数，所以用e的次方二次处理一下，最后的数据形式是tensor，所以用int转换了一次
            good_point =  int(predicted_label[0, 2])
            bad_point  =  int(predicted_label[0, 0])
            mid_point  =  int(predicted_label[0, 1])

            pre_good_point = math.exp(good_point)
            pre_bad_point  = math.exp(bad_point)
            pre_mid_point  = math.exp(mid_point)

        prediction.append(predicted_class)
        prediction_value.append(predicted_value)
        #三个打分的录入
        prediction_good_point.append(pre_good_point)
        prediction_bad_point.append(pre_bad_point)
        prediction_mid_point.append(pre_mid_point)

    dataframe['Predicted_Label'] = prediction
    dataframe['Predicted_value'] = prediction_value

    dataframe['Point_good'] = prediction_good_point
    dataframe['Point_bad'] = prediction_bad_point
    dataframe['Point_mid'] = prediction_mid_point

    dataframe['times'] = get_numeric_date()
    dataframe['dates'] = get_times()
    dataframe['DateAndHour'] = get_hourtime()
    
    conn = sqlite3.connect('data_append.db')
    #追加
    dataframe.to_sql('data_append', conn, if_exists='append')
    #获得数据库有多少行
    cursor = conn.cursor()
    ##查询数据库中的行数
    cursor.execute('SELECT COUNT(*) FROM data_append')
    row_count = cursor.fetchone()[0]
    # print("数据库中的行数为:", row_count)
    conn.close()
    

    conn = sqlite3.connect('data.db')
        #覆盖
    dataframe.to_sql('data', conn, if_exists='replace')
    conn.close()

    # 大于500行清洗一次
    if row_count > 500:
        conn = sqlite3.connect('data_append.db')
        cursor = conn.cursor()
        # 获取当前时间并格式化为日期级别，比如0528
        current_time = (dt.datetime.now() - dt.timedelta(days=14)).strftime('%m%d')
        # 执行删除操作
        cursor.execute("DELETE FROM data_append WHERE dates = ?", (current_time,))
        # 提交更改
        conn.commit()
        # 关闭连接
        conn.close()

        
        print('已完成数据的更新与清洗')

        return 'ok'
    else:
        print('数据更新失败')


def show_time():

    timeshow = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    timenow = datetime.now()  # 获得当前时间
    # print(f"当前时间：{timeshow}")
    return timenow

def ht_pc(str, Number):


    # 所需数据结构
    Page = 30

    mid_list = []
    Cookie = "SINAGLOBAL=4447791141669.853.1654058208679; SCF=AjXjjI7POhpjwpRnMpXr97hrJSZGaaQ2UEtX9dVA7Cj8UVVknBobu2pf4S33KR_QzkXYL8vkvx_X3CDECKmVoRA.; UOR=www.weibotop.cn,s.weibo.com,www.weibotop.cn; PC_TOKEN=0d2f42e957; SUB=_2A25LvQOADeRhGeFJ61YT8yrEwjmIHXVosxlIrDV8PUNbmtANLWfBkW9NfKX81DSsnW_VmpJLsxQKoewla2hZj05o; SUBP=0033WrSXqPxfM725Ws9jqgMF55529P9D9WhQwKL42_9zuokhfC40DShy5NHD95QNS05XeoeX1h.fWs4DqcjMi--NiK.Xi-2Ri--ciKnRi-zNS0M7Shz0Shn4SBtt; ALF=02_1726021840; _s_tentry=weibo.com; Apache=9240845623189.717.1723429864789; ULV=1723429864829:13:3:1:9240845623189.717.1723429864789:1722664711936"
    head = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0",
        "Cookie": Cookie
    }

    for page in range(0, Page + 1):
        url = f'https://s.weibo.com/weibo?q={str}&page={page}'
        response = requests.get(url=url, headers=head)
        response.encoding = 'UTF-8-sig'
        time.sleep(1)
        if (response.ok):

            content = response.text
            # print(content)
            soup = BeautifulSoup(content, "html.parser")
            # print(soup)
            # 获取mid
            all_mid = soup.findAll("div", attrs={"action-type": "feed_list_item", "class": "card-wrap"})
            for mid in all_mid:
                mid_get = mid.get("mid")
                print(f"mid_get:{mid_get}")
                if mid_get != None:
                    mid_list.append(mid_get)
        else:
            print("请求失败")


    return mid_list


def pl_pc(data,title,index,data_bool):
                                 # 所需数据结构
    # user_name = '用户名'
    # user_name_list = []
    # text = '评论'
    # text_list = []
    # time = '时间'
    # time_list = []
    # like_counts = '点赞数'
    # like_counts_list = []
    #
    # id_list = []  # 评论下的回复接口id
    ip_list = []
    Cookie = "SINAGLOBAL=4447791141669.853.1654058208679; SCF=AjXjjI7POhpjwpRnMpXr97hrJSZGaaQ2UEtX9dVA7Cj8UVVknBobu2pf4S33KR_QzkXYL8vkvx_X3CDECKmVoRA.; UOR=www.weibotop.cn,s.weibo.com,www.weibotop.cn; PC_TOKEN=0d2f42e957; SUB=_2A25LvQOADeRhGeFJ61YT8yrEwjmIHXVosxlIrDV8PUNbmtANLWfBkW9NfKX81DSsnW_VmpJLsxQKoewla2hZj05o; SUBP=0033WrSXqPxfM725Ws9jqgMF55529P9D9WhQwKL42_9zuokhfC40DShy5NHD95QNS05XeoeX1h.fWs4DqcjMi--NiK.Xi-2Ri--ciKnRi-zNS0M7Shz0Shn4SBtt; ALF=02_1726021840; _s_tentry=weibo.com; Apache=9240845623189.717.1723429864789; ULV=1723429864829:13:3:1:9240845623189.717.1723429864789:1722664711936"
    head = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0",
        "Cookie": Cookie,
        "Referer": "https://s.weibo.com/weibo?q=%E8%83%96%E7%8C%AB",
        'Content-Type': 'application/x-www-form-urlencoded',
        'X-Requested-With': 'XMLHttpRequest'
    }

    url = 'https://weibo.com/ajax/statuses/buildComments?'
    while (1):
        response = requests.get(url=url, headers=head, params=data,timeout=2)
        response.encoding = 'UTF-8-sig'
        # time.sleep(1)
        if (response.ok):
            content = response.text
            # print(content)
            content = json.loads(content)
            Data = content["data"]
            if (len(Data) == 0):
                break
            data["max_id"] = content["max_id"]
            for data_index in Data:
                # created_at         时间
                # like_counts        点赞数
                # text_raw           评论
                # user下的screen_name 用户

                # user_name_list.append(data_index["user"]["screen_name"])
                # text_list.append(data_index["text_raw"])
                # time_list.append(data_index["created_at"])
                # like_counts_list.append(data_index["like_counts"])
                # print(data_index["source"][2:])
                #
                # id_list.append(data_index["id"])
                if "source" in data_index and "like_counts" in data_index:
                    ip_list.append(data_index["source"][2:])
            if(data["max_id"] == 0):
                break
        else:
            print(F'response.ok = {response.ok}')

    return ip_list

#输入
def main(str):
    ip_name = 'ip地址'
    ip = []

    num = 5
    Number = 500
    data_bool = 1
    # threads = [] #线程列表

    print("请稍等片刻。。。")
    mid_list = ht_pc(str, Number)  # 获取每个话题的mid
    print('话题爬取完成.....\n')
    time.sleep(1)
    # print(len(mid_list))
    # for i in mid_list:
    #     print(i)

    # 为每个min（评论）包装data属性
    data_list = []
    for mid in mid_list:
        data = {
            'id': mid,
            'is_show_bulletin': '2',
            'is_mix': '0',
            'count': '20',
            'max_id': '0'
        }
        data_list.append(data)

    for data_index in range(0, len(data_list)):
        ip_list = pl_pc(data_list[data_index], str, data_index, data_bool)
        ip.extend(ip_list)
        print(f'话题{data_index + 1}评论爬取完成.....\n')
    # print(len(ip))
    # print(f'ip为：{ip}')

    ip_tj = {}
    for ip_data in ip:
        if ip_data in ip_tj:
            ip_tj[ip_data] += 1
        else:
            ip_tj[ip_data] = 1
    # print(f'ip字典为：{ip_tj}')
    # dataframe = pd.DataFrame({ip_name: ip})
    # dataframe.to_csv(f"./数据/关于{str}关键字的ip.csv", index=False, sep=',', encoding='utf_8_sig')


    # 预定义的中国省份列表
    chinese_provinces = [
        '北京', '天津', '河北', '山西', '内蒙古', '辽宁', '吉林', '黑龙江', '上海', 
        '江苏', '浙江', '安徽', '福建', '江西', '山东', '河南', '湖北', '湖南', 
        '广东', '广西', '海南', '重庆', '四川', '贵州', '云南', '西藏', '陕西', 
        '甘肃', '青海', '宁夏', '新疆', '中国台湾', '中国香港', '中国澳门'
    ]

    # 原始数据字典
    data = ip_tj

    # 创建或连接到数据库
    conn = sqlite3.connect('ip.db')

    # 创建游标对象
    cursor = conn.cursor()

    # 创建表
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS province_data (
        province TEXT PRIMARY KEY,
        number INTEGER
    )
    ''')

    # 插入数据
    for province in chinese_provinces:
        number = data.get(province, 0)  # 获取字典中的值，如果没有则为0
        cursor.execute('''
        INSERT OR REPLACE INTO province_data (province, number)
        VALUES (?, ?)
        ''', (province, number))

    # 提交事务
    conn.commit()

    # 关闭连接
    conn.close()
    print("ip刷新已完成")

    return ip_tj





if __name__ == '__main__':
    # 初始化数据
    get_data_back()
    # make_hot_trend(fromtime = (dt.datetime.now()).strftime('%m%d'))
    app.run(debug=True, port=5000)
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



@app.route('/ip_flush/', methods=['GET'])
def ip_flush():
    # str = input("请输入要爬取数据的关键字：")
     # 连接到数据库
    conn = sqlite3.connect('data_append.db')
    cursor = conn.cursor()

    # 查询所有标题，url和标签
    cursor.execute("SELECT keyword1, keyword2, keyword3, hot, times FROM data_append")
    word_flush = cursor.fetchall()

    # 关闭数据库连接
    conn.close()

    # 获取所有的标点符号
    punctuation = set(string.punctuation)
    chinese_punctuation = set('，！？。；：‘’“”（）【】《》')

    
    # 找出所有不同的times值
    times_values = set(item[4] for item in word_flush)

    if len(times_values) >= 2:
        # 找出最大和次大的times值
        max_times = max(times_values)
        second_max_times = max(t for t in times_values if t < max_times)

        # 找出最大和次大times值对应的数据
        latest_data_0 = [item for item in word_flush if item[4] == max_times]
        second_latest_data_0 = [item for item in word_flush if item[4] == second_max_times]

        latest_data = []
        second_latest_data = []

        for item in latest_data_0:
            for i in range(0,3):
                if item[i] not in punctuation and item[i] != ' ' and item[i] != '' and item[i] not in chinese_punctuation:
                    if item[i] is not None:
                        latest_data.append((item[i],item[3]))

        for item in second_latest_data_0:
            for i in range(0,3):
                if item[i] not in punctuation and item[i] != ' ' and item[i] != '' and item[i] not in chinese_punctuation:
                    if item[i] is not None:
                        second_latest_data.append((item[i],item[3]))


        # 计算关键词总热度
        hot_latest = {}
        for item in latest_data:
            keyword = item[0]
            hot = item[1]
            if keyword in hot_latest:
                hot_latest[keyword] += hot
            else:
                hot_latest[keyword] = hot
        
        hot_second_latest = {}
        for item in second_latest_data:
            keyword = item[0]
            hot = item[1]
            if keyword in hot_second_latest:
                hot_second_latest[keyword] += hot
            else:
                hot_second_latest[keyword] = hot

        # 计算热度上升速度
        word_speed = []
        for word, hot in hot_latest.items():
            if word in hot_second_latest:
                hot_second = hot_second_latest[word]
                speed = hot / hot_second - 1
            else:
                speed = 1
            word_speed.append((word, speed))

        # 按速度从大到小排序
        word_speed.sort(key=lambda x: x[1], reverse=True)

    else:
        words =[ i[0] for i in word_flush]
        for word in words:
            word_speed.append((word, 1))
    word_top1 = word_speed[0][0]
    print("ip刷新已执行")
    # t1 = show_time()


    #这个是函数的用法,ip_tj就是返回的字典
    # ip_tj = main(str)
    main(word_top1)

    # t2 = show_time()
    # print(f"总耗时：{t2 - t1}")
    return "ok"



@app.route('/ip_get/', methods=['GET'])
def ip_get():
    # 连接到数据库
    conn = sqlite3.connect('ip.db')
    cursor = conn.cursor()

    # 查询所有标题，url和标签
    cursor.execute("SELECT province, number FROM province_data")
    ip_flush = cursor.fetchall()
    # 关闭数据库连接
    conn.close()

    return jsonify(ip_flush)







# 创建一个接口供前端访问数据
@app.route('/pie_data_imm/', methods=['GET'])
def pie_data():
    # 连接到数据库
    conn = sqlite3.connect('data.db')
    cursor = conn.cursor()

    # 查询所有标题，url和标签
    cursor.execute("SELECT title, url, source, hot, keyword1, keyword2, keyword3, Predicted_Label, times FROM data")
    data_df = pd.DataFrame(cursor.fetchall(), columns=['title', 'url', 'source', 'hot', 'keyword1', 'keyword2', 'keyword3', 'Predicted_Label', 'times'])
    # 关闭数据库连接
    conn.close()

    label_counts = data_df['Predicted_Label'].value_counts()
    label_data = label_counts.to_dict()

    return jsonify(label_data)


@app.route('/hot_flush/', methods=['GET'])
def hot_flush():
    # 连接到数据库
    conn = sqlite3.connect('data.db')
    cursor = conn.cursor()

    # 查询所有标题，url和标签
    cursor.execute("SELECT title, url, source, hot, keyword1, keyword2, keyword3, Predicted_Label, times FROM data")
    filtered_df = pd.DataFrame(cursor.fetchall(), columns=['title', 'url', 'source', 'hot', 'keyword1', 'keyword2', 'keyword3', 'Predicted_Label', 'times'])
    # 关闭数据库连接
    conn.close()


    # 按照hot值降序排列筛选后的数据框
    filtered_df = filtered_df.sort_values(by='hot', ascending=False)

    # 获取hot值最高的前n个元素的title、url和Predicted_Label
    n = 5
    top_elements = filtered_df.head(n)[['title', 'url', 'Predicted_Label']].values.tolist()

    # 将超过17字的标题显示为前十五字加上省略号
    for rank,element in enumerate(top_elements, start=1):
        if len(element[0]) >= 17:
            element[0] = element[0][:15] + '......'
        element.append(rank)
        
    return jsonify(top_elements)



@app.route('/api/hot_word_speed/')   
def hot_word_speed():
        # 连接到数据库
    conn = sqlite3.connect('data_append.db')
    cursor = conn.cursor()

    # 查询所有标题，url和标签
    cursor.execute("SELECT keyword1, keyword2, keyword3, hot, times FROM data_append")
    word_flush = cursor.fetchall()

    # 关闭数据库连接
    conn.close()

    # 获取所有的标点符号
    punctuation = set(string.punctuation)
    chinese_punctuation = set('，！？。；：‘’“”（）【】《》')


    word_find = []
    
    # 找出所有不同的times值
    times_values = set(item[4] for item in word_flush)

    if len(times_values) >= 2:
        # 找出最大和次大的times值
        max_times = max(times_values)
        second_max_times = max(t for t in times_values if t < max_times)

        # 找出最大和次大times值对应的数据
        latest_data_0 = [item for item in word_flush if item[4] == max_times]
        second_latest_data_0 = [item for item in word_flush if item[4] == second_max_times]

        latest_data = []
        second_latest_data = []

        for item in latest_data_0:
            for i in range(0,2):
                if item[i] not in punctuation and item[i] != ' ' and item[i] != '' and item[i] not in chinese_punctuation:
                    if item[i] is not None:
                        latest_data.append((item[i],item[3]))

        for item in second_latest_data_0:
            for i in range(0,2):
                if item[i] not in punctuation and item[i] != ' ' and item[i] != '' and item[i] not in chinese_punctuation:
                    if item[i] is not None:
                        second_latest_data.append((item[i],item[3]))


        # 计算关键词总热度
        hot_latest = {}
        for item in latest_data:
            keyword = item[0]
            hot = item[1]
            if keyword in hot_latest:
                hot_latest[keyword] += hot
            else:
                hot_latest[keyword] = hot
        
        hot_second_latest = {}
        for item in second_latest_data:
            keyword = item[0]
            hot = item[1]
            if keyword in hot_second_latest:
                hot_second_latest[keyword] += hot
            else:
                hot_second_latest[keyword] = hot

        # 计算热度上升速度
        word_speed = []
        for word, hot in hot_latest.items():
            if word in hot_second_latest:
                hot_second = hot_second_latest[word]
                speed = hot / hot_second - 1
            else:
                speed = 1
            word_speed.append((word, speed))

        # 按速度从大到小排序
        word_speed.sort(key=lambda x: x[1], reverse=True)

    else:
        words =[ i[0] for i in word_flush]
        for word in words:
            word_speed.append((word, 1))
    word_speed_top4 = word_speed[:5]
    return jsonify(word_speed_top4)  



@app.route('/api/flush_line_history/')
def flush_line_history():
     # 连接到数据库
    conn = sqlite3.connect('data_append.db')
    cursor = conn.cursor()

    # 查询所有标题，url和标签
    cursor.execute("SELECT keyword1, keyword2, keyword3, hot, times FROM data_append")
    word_flush = cursor.fetchall()

    # 关闭数据库连接
    conn.close()

    # 获取所有的标点符号
    punctuation = set(string.punctuation)
    chinese_punctuation = set('，！？。；：‘’“”（）【】《》')

    
    # 找出所有不同的times值
    times_values = set(item[4] for item in word_flush)

    if len(times_values) >= 2:
        # 找出最大和次大的times值
        max_times = max(times_values)
        second_max_times = max(t for t in times_values if t < max_times)

        # 找出最大和次大times值对应的数据
        latest_data_0 = [item for item in word_flush if item[4] == max_times]
        second_latest_data_0 = [item for item in word_flush if item[4] == second_max_times]

        latest_data = []
        second_latest_data = []

        for item in latest_data_0:
            for i in range(0,3):
                if item[i] not in punctuation and item[i] != ' ' and item[i] != '' and item[i] not in chinese_punctuation:
                    if item[i] is not None:
                        latest_data.append((item[i],item[3]))

        for item in second_latest_data_0:
            for i in range(0,3):
                if item[i] not in punctuation and item[i] != ' ' and item[i] != '' and item[i] not in chinese_punctuation:
                    if item[i] is not None:
                        second_latest_data.append((item[i],item[3]))


        # 计算关键词总热度
        hot_latest = {}
        for item in latest_data:
            keyword = item[0]
            hot = item[1]
            if keyword in hot_latest:
                hot_latest[keyword] += hot
            else:
                hot_latest[keyword] = hot
        
        hot_second_latest = {}
        for item in second_latest_data:
            keyword = item[0]
            hot = item[1]
            if keyword in hot_second_latest:
                hot_second_latest[keyword] += hot
            else:
                hot_second_latest[keyword] = hot

        # 计算热度上升速度
        word_speed = []
        for word, hot in hot_latest.items():
            if word in hot_second_latest:
                hot_second = hot_second_latest[word]
                speed = hot / hot_second - 1
            else:
                speed = 1
            word_speed.append((word, speed))

        # 按速度从大到小排序
        word_speed.sort(key=lambda x: x[1], reverse=True)

    else:
        words =[ i[0] for i in word_flush]
        for word in words:
            word_speed.append((word, 1))
    word_top1 = word_speed[0][0]

    word1_history = []
    for item in word_flush: 
        for i in range(0,3):
            if item[i] == word_top1:
                word1_history.append(item)
    
    word1_hot = {}
    # 找出所有不同的times值
    times_word1 = []
    for item in word1_history:
            time = item[4]
            hot = item[3]
            if time in times_word1:
                word1_hot[time] += hot
            else:
                word1_hot[time] = hot
                times_word1.append(time)
    
    word_find = [[time, hot] for time, hot in word1_hot.items()]


    if len(word_find) <= 6:
        word_find.reverse()
        while len(word_find) <= 6:
            word_find.append(['no-data', 0])
        word_find.reverse()
    word_latest5 = word_find[-7:-1]
    return jsonify(word_latest5)



@app.route('/api/hot_flush_line/')   
def hot_flush_line():
    # 连接到数据库
    conn = sqlite3.connect('data_append.db')
    cursor = conn.cursor()

    # 查询所有标题，url和标签
    cursor.execute("SELECT keyword1, keyword2, keyword3, hot, times FROM data_append")
    word_flush = cursor.fetchall()

    # 关闭数据库连接
    conn.close()

    # 获取所有的标点符号
    punctuation = set(string.punctuation)
    chinese_punctuation = set('，！？。；：‘’“”（）【】《》')

    
    # 找出所有不同的times值
    times_values = set(item[4] for item in word_flush)

    if len(times_values) >= 2:
        # 找出最大和次大的times值
        max_times = max(times_values)
        second_max_times = max(t for t in times_values if t < max_times)

        # 找出最大和次大times值对应的数据
        latest_data_0 = [item for item in word_flush if item[4] == max_times]
        second_latest_data_0 = [item for item in word_flush if item[4] == second_max_times]

        latest_data = []
        second_latest_data = []

        for item in latest_data_0:
            for i in range(0,3):
                if item[i] not in punctuation and item[i] != ' ' and item[i] != '' and item[i] not in chinese_punctuation:
                    if item[i] is not None:
                        latest_data.append((item[i],item[3]))

        for item in second_latest_data_0:
            for i in range(0,3):
                if item[i] not in punctuation and item[i] != ' ' and item[i] != '' and item[i] not in chinese_punctuation:
                    if item[i] is not None:
                        second_latest_data.append((item[i],item[3]))


        # 计算关键词总热度
        hot_latest = {}
        for item in latest_data:
            keyword = item[0]
            hot = item[1]
            if keyword in hot_latest:
                hot_latest[keyword] += hot
            else:
                hot_latest[keyword] = hot
        
        hot_second_latest = {}
        for item in second_latest_data:
            keyword = item[0]
            hot = item[1]
            if keyword in hot_second_latest:
                hot_second_latest[keyword] += hot
            else:
                hot_second_latest[keyword] = hot

        # 计算热度上升速度
        word_speed = []
        for word, hot in hot_latest.items():
            if word in hot_second_latest:
                hot_second = hot_second_latest[word]
                speed = hot / hot_second - 1
            else:
                speed = 1
            word_speed.append((word, speed, hot))

        # 按速度从大到小排序
        word_speed.sort(key=lambda x: x[1], reverse=True)

    else:
        words =[ i[0] for i in word_flush]
        for word in words:
            word_speed.append((word, 1))
    word_top1 = word_speed[0][0]

    word_la_hot = [word_top1, max_times, word_speed[0][2]]
    return jsonify(word_la_hot)


@app.route('/api/hot_topics/')
def get_titles():
    # 连接到数据库
    conn = sqlite3.connect('data_append.db')
    cursor = conn.cursor()

    # 查询所有标题，url和标签
    cursor.execute("SELECT  title, url, Predicted_Label, hot,  keyword1, keyword2, keyword3, source, times FROM data_append")
    time1 = cursor.fetchall()

    # 关闭数据库连接
    conn.close()

    x = len(time1) -1
    y = len(time1) -1
    titles = []
    for i in range(len(time1)-1, 1, -1):
        if time1[i][8] == time1[i-1][8]:
            x -= 1
        else:
            time_same = []
            for i in range(y, x,  -1):
                time_same.append(time1[i])
            random.seed(114)
            random.shuffle(time_same)
            x -= 1
            y = x
            titles += time_same
        
    # 渲染到HTML模板并返回
    return render_template('titles.html', titles=titles)


@app.route('/api/hot_topics_positive/')
def get_titles_po():
    # 连接到数据库
    conn = sqlite3.connect('data_append.db')
    cursor = conn.cursor()

    # 查询所有标题，url和标签
    cursor.execute("SELECT title, url, Predicted_Label, hot,  keyword1, keyword2, keyword3, source, times FROM data_append")
    time1 = cursor.fetchall()

    # 关闭数据库连接
    conn.close()

    x = len(time1) -1
    y = len(time1) -1
    titles = []
    for i in range(len(time1)-1, 1, -1):
        if time1[i][8] == time1[i-1][8]:
            x -= 1
        else:
            time_same = []
            for i in range(y, x,  -1):
                time_same.append(time1[i])
            random.seed(114)
            random.shuffle(time_same)
            x -= 1
            y = x
            titles += time_same

    # 渲染到HTML模板并返回
    return render_template('titles_positive.html', titles=titles)


@app.route('/api/hot_topics_negative/')
def get_titles_ne():
    # 连接到数据库
    conn = sqlite3.connect('data_append.db')
    cursor = conn.cursor()

    # 查询所有标题，url和标签
    cursor.execute("SELECT title, url, Predicted_Label, hot,  keyword1, keyword2, keyword3, source, times FROM data_append")
    time1 = cursor.fetchall()

    # 关闭数据库连接
    conn.close()

    x = len(time1) -1
    y = len(time1) -1
    titles = []
    for i in range(len(time1)-1, 1, -1):
        if time1[i][8] == time1[i-1][8]:
            x -= 1
        else:
            time_same = []
            for i in range(y, x,  -1):
                time_same.append(time1[i])
            random.seed(114)
            random.shuffle(time_same)
            x -= 1
            y = x
            titles += time_same

    # 渲染到HTML模板并返回
    return render_template('titles_negative.html', titles=titles)


# 跳转至对应详情页
def get_news_details(news_id):
    connection = sqlite3.connect('data_append.db')
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM data_append WHERE ROWID=?", (news_id,))
    news = cursor.fetchone()
    connection.close()
    return news

@app.route('/news/<int:news_id>', methods=['GET'])
def news_details(news_id):
    news = get_news_details(news_id)
    point_p = news[12]
    point_n = news[13]
    point_m = news[14]
    point_s = point_p + point_n + point_m
    per_p = (round((point_p/point_s*100), 1))
    per_n = (round((point_n/point_s*100), 1))
    per_m = (round((point_m/point_s*100), 1))
    # 连接到数据库
    conn = sqlite3.connect('data_append.db')
    cursor = conn.cursor()

    # 查询所有标题，url和标签
    cursor.execute("SELECT * FROM hot_trend")
    hot_trend = cursor.fetchall()

    # 查询所有标题，url和标签
    cursor.execute("SELECT * FROM data_append")
    news_data = cursor.fetchall()
    # 关闭数据库连接
    conn.close()
    for item in hot_trend:
        if item[0] == news[1]:
            date = news[11]
             # 将热度部分转换为列表，以便修改
            hotness = list(item[1:])
            
            # 判断当前日期，替换日期之后且为0的热度
            for i in range(len(hotness)):
                if hotness[i] == 0.0:
                    # 假设 news[11] 存储的日期格式与 current_time 格式相同
                    item_date = (dt.datetime.now() - dt.timedelta(days=i)).strftime('%m%d')
                    if int(item_date) > int(date):
                        hotness[i] = 10000.0
            hot_history = hotness


    wordHot1 = 0
    wordHot2 = 0
    wordHot3 = 0
    wordNum1 = 0
    wordNum2 = 0
    wordNum3 = 0
    for item in news_data:
        if item[5]==news[5] or item[6]==news[5] or item[7]==news[5]:
            wordHot1 += item[4]
            wordNum1 += 1
        elif item[5]==news[6] or item[6]==news[6] or item[7]==news[6]:
            wordHot2 += item[4]
            wordNum2 += 1
        elif item[5]==news[7] or item[6]==news[7] or item[7]==news[7]:
            wordHot3 += item[4]
            wordNum3 += 1
    wordHot1 = int(wordHot1 * 0.01)
    wordHot2 = int(wordHot2 * 0.01)
    wordHot3 = int(wordHot3 * 0.01)


    if news:
        news_data = {
            'id': news[0],
            'title': news[1],
            'url': news[2],
            'source': news[3],
            'hot': news[4],
            'keyword1': news[5],
            'keyword2': news[6],
            'keyword3': news[7],
            'wordHot1': wordHot1,
            'wordNum1': wordNum1,
            'wordHot2': wordHot2,
            'wordNum2': wordNum2,
            'wordHot3': wordHot3,
            'wordNum3': wordNum3,
            'PredictedLabel': news[8],
            'times': news[10],
            'date': news[11],
            'perP': per_p,
            'perN': per_n,
            'perM': per_m,
            'hot_trend': hot_history,
            'current_date': (dt.datetime.now()).strftime('%m%d')
        }
        return jsonify(news_data)
    else:
        return jsonify({'error': 'News not found'}), 404




@app.route('/api/emo_per/')
def index():
    # 连接到数据库
    conn = sqlite3.connect('data_append.db')
    cursor = conn.cursor()

    # 查询所有标题，url和标签
    cursor.execute("SELECT Predicted_Label FROM data_append")
    emo_data_ori = cursor.fetchall()

    # 关闭数据库连接
    conn.close()

    emo_data = [item[0] for item in emo_data_ori]
    event_len = len(emo_data)
    mid_count = emo_data.count(1)
    neg_count = emo_data.count(0)
    mid_per = (round((mid_count/event_len*100), 2))
    neg_per = (round((neg_count/event_len*100), 2))
    pos_per = round((100 - mid_per - neg_per), 2)
    emo_per = [pos_per, neg_per, mid_per]

    return send_file('demo.html')


@app.route('/get_list/')
def get_list():
    # 连接到数据库
    conn = sqlite3.connect('data_append.db')
    cursor = conn.cursor()

    # 查询所有标题，url和标签
    cursor.execute("SELECT Predicted_Label FROM data_append")
    emo_data_ori = cursor.fetchall()

    # 关闭数据库连接
    conn.close()


    emo_data = [item[0] for item in emo_data_ori]
    event_len = len(emo_data)
    mid_count = emo_data.count(1)
    neg_count = emo_data.count(0)
    mid_per = (round((mid_count/event_len*100), 2))
    neg_per = (round((neg_count/event_len*100), 2))
    pos_per = round((100 - mid_per - neg_per), 2)
    emo_per = [pos_per, neg_per, mid_per]
    return json.dumps(emo_per)


@app.route('/api/emo_data/')
def index1():
    # 连接到数据库
    conn = sqlite3.connect('data_append.db')
    cursor = conn.cursor()

    # 查询所有标题，url和标签
    cursor.execute("SELECT Predicted_Label FROM data_append")
    emo_data_ori = cursor.fetchall()

    # 关闭数据库连接
    conn.close()

    emo_data = [item[0] for item in emo_data_ori]
    event_len = len(emo_data)
    pos_count = emo_data.count(2)
    neg_count = emo_data.count(0)
    data_num = [event_len, pos_count, neg_count]

    return send_file('demo.html')


@app.route('/get_list_data/')
def get_list_data():
    # 连接到数据库
    conn = sqlite3.connect('data_append.db')
    cursor = conn.cursor()

    # 查询所有标题，url和标签
    cursor.execute("SELECT Predicted_Label FROM data_append")
    emo_data_ori = cursor.fetchall()

    # 关闭数据库连接
    conn.close()


    emo_data = [item[0] for item in emo_data_ori]
    event_len = len(emo_data)
    pos_count = emo_data.count(2) + emo_data.count(1)
    neg_count = emo_data.count(0)
    data_num = [event_len, pos_count, neg_count]
    return json.dumps(data_num)


@app.route('/api/line_data/')
def line_data():
    # 连接到数据库
    conn = sqlite3.connect('data_append.db')
    cursor = conn.cursor()

    # 查询所有标题，url和标签
    cursor.execute("SELECT Predicted_Label, dates FROM data_append")
    line_data = cursor.fetchall()

    # 关闭数据库连接
    conn.close()

    # 创建一个 defaultdict 用于统计每个时间段中 0、1、2 的数量
    time_stats = defaultdict(lambda: {'0': 0, '1': 0, '2':0})

    # 遍历列表 data，统计每个时间段中 0、1、2 的数量
    for i in range(len(line_data)):
        label, time = line_data[i]
        if label == 0:
            time_stats[time]['0'] += 1
        elif label == 1:
            time_stats[time]['1'] += 1
        elif label == 2:
            time_stats[time]['2'] += 1

    # 获取最近的五个时间
    recent_times = [time for time, _ in sorted(time_stats.items(), key=lambda x: x[0], reverse=True)[:5]]

    # 获取每个时间段中 0、1、2 的总数
    zero_counts = [time_stats[time]['0'] for time in recent_times]
    one_counts = [time_stats[time]['1'] for time in recent_times]
    two_counts = [time_stats[time]['2'] for time in recent_times]

    # 填充时间不足五个时的情况
    while len(recent_times) < 5:
        recent_times.append(0)
        zero_counts.append(0)
        one_counts.append(0)
        two_counts.append(0)
    recent_times = recent_times[::-1]
    zero_counts = zero_counts[::-1]
    one_counts = one_counts[::-1]
    two_counts = two_counts[::-1]

    
    data = {
        'recent_times': recent_times,
        'zero_counts': zero_counts,
        'one_counts': one_counts,
        'two_counts': two_counts
    }
    return jsonify(data)



@app.route('/api/get_hot_word/')
def get_all_word():
    # 连接到数据库
    conn = sqlite3.connect('data_append.db')
    cursor = conn.cursor()

    # 查询所有标题，url和标签
    cursor.execute("SELECT hot, keyword1, keyword2, keyword3 FROM data_append")
    word_all_0 = cursor.fetchall()

    # 关闭数据库连接
    conn.close()


    word_all_ori = [(row[0], row[1], row[2], row[3]) for row in word_all_0]

    # 获取所有的标点符号
    punctuation = set(string.punctuation)

    word_all = []
    chinese_punctuation = set('，！？。；：‘’“”（）【】《》')

    for row in word_all_ori:
        count = row[0]
        words = [word for word in row[1:] if word not in punctuation and word not in chinese_punctuation and word is not None and word !='' and word != ' ']
        
        if len(words) > 0:
            word_all.append((count, *words))

    keyword_dict = {}

    for row in word_all:
        hot = row[0]
        keywords = row[1:]
        
        for keyword in keywords:
            if keyword in keyword_dict:
                keyword_dict[keyword] = (keyword_dict[keyword][0] + 1, keyword_dict[keyword][1] + hot)
            else:
                keyword_dict[keyword] = (1, hot)

    sorted_keywords = sorted(keyword_dict.items(), key=lambda x: x[1][1], reverse=True)[:5]

   # 计算这五个关键词的出现总数
    total_top5_count = sum(count for keyword, (count, _) in sorted_keywords)

    output = []
    for keyword, (count, hot_sum) in sorted_keywords:
        percentage = (count / total_top5_count) * 100
        output.append((keyword, count, hot_sum, percentage))

    return render_template('hot_word.html', output=output)



@app.route('/api/get_ciyun/')
def get_ciyun():
    # 连接到数据库
    conn = sqlite3.connect('data_append.db')
    cursor = conn.cursor()

    # 查询所有标题，url和标签
    cursor.execute("SELECT hot, keyword1, keyword2, keyword3 FROM data_append")
    word_all_0 = cursor.fetchall()

    # 关闭数据库连接
    conn.close()


    word_all_ori = [(row[0], row[1], row[2], row[3]) for row in word_all_0]

    # 获取所有的标点符号
    punctuation = set(string.punctuation)

    word_all = []
    chinese_punctuation = '，！？。；：‘’“”（）【】《》'

    for row in word_all_ori:
        count = row[0]
        words = [word for word in row[1:] if (word not in punctuation and word not in [' ','']) or word in chinese_punctuation]
        
        if len(words) > 0:
            word_all.append((count, *words))

    keyword_dict = {}

    for row in word_all:
        hot = row[0]
        keywords = row[1:]
        
        for keyword in keywords:
            if keyword in keyword_dict:
                keyword_dict[keyword] = (keyword_dict[keyword][0] + 1, keyword_dict[keyword][1] + hot)
            else:
                keyword_dict[keyword] = (1, hot)

    sorted_keywords = sorted(keyword_dict.items(), key=lambda x: x[1][0], reverse=True)
    cleaned_sorted_keywords = [item for item in sorted_keywords if item[0] is not None]
    keyword_top_58 = cleaned_sorted_keywords[:58]


    # 初始化总和为0
    total_count = 0
    # 遍历sorted_keywords并累加每个关键词出现次数所在列的值
    for keyword, (count, value) in keyword_top_58:
        total_count += count


    output = []
    for keyword, (count, hot_sum) in keyword_top_58:
        percentage = round(((count / total_count) * 100), 1)
        output.append((keyword, count, hot_sum, percentage))

        
    return jsonify(output)


def query_database(query):
    # 连接 SQLite 数据库
    conn = sqlite3.connect('data_append.db')
    cursor = conn.cursor()

    # 执行 SQL 查询以查找包含输入查询的标题
    cursor.execute("SELECT title, url, source, times, Predicted_Label  FROM data_append WHERE title LIKE ?", ('%' + query + '%',))
    
    # 获取所有匹配的结果
    results = cursor.fetchall()
    results.reverse()
    
    # 关闭数据库连接
    conn.close()
    
    # 返回标题列表
    return [{'title': result[0], 'url': result[1], 'source': result[2], 'times': result[3], 'Predicted_Label': result[4]} for result in results]

@app.route('/search/', methods=['GET'])
def search():
    # 从请求中获取搜索查询参数
    query = request.args.get('query', '')

    if query:
        # 查询数据库以获取匹配的标题
        results = query_database(query)
    else:
        results = []

    # 以 JSON 格式返回结果
    return jsonify(results)


def get_date():
    now = dt.datetime.now()
    return now.strftime("%m%d")

def get_date_table():
    now = dt.datetime.now()
    return str('table'+ str(now.strftime("%m%d")))

def get_past_five_days(input_date):
    # 获取当前日期的年份
    current_year = datetime.now().year
    # 将输入日期字符串转换为 datetime 对象
    input_date_obj = datetime.strptime(input_date, "%m%d").replace(year=current_year)
    
    # 获取今天的日期
    today = datetime.now().date()
    # 如果输入日期已经过去，将其年份加一年
    if input_date_obj.date() < today:
        input_date_obj = input_date_obj.replace(year=current_year + 1)
    
    # 创建一个空列表来存储结果
    result = []
    
    # 循环生成前五天的日期
    for i in range(5):
        # 计算当前循环中的日期
        date_to_add = input_date_obj - timedelta(days=i)
        # 将日期格式化为 %m%d 形式的字符串
        formatted_date = date_to_add.strftime("%m%d")
        # 将格式化后的日期添加到结果列表中
        result.append(formatted_date)
    
    # 返回结果列表
    return result

def remove_duplicates(input_list):
    # 使用集合来去除重复元素，并保持原来的顺序
    seen = set()
    output_list = []
    for item in input_list:
        if item not in seen:
            seen.add(item)
            output_list.append(item)
    return output_list

#生成数据库标题列表用的
def fetch_titles_from_db(db_path='data_append.db', table_name='data_append'):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    
    try:
        # Create a cursor object to execute SQL queries
        cursor = conn.cursor()
        
        # Query to fetch all titles from the table
        query = f"SELECT title FROM {table_name}"
        cursor.execute(query)
        
        # Fetch all rows (titles)
        rows = cursor.fetchall()
        
        # Extract titles from the fetched rows
        titles_list = [row[0] for row in rows]
        
        return titles_list
        
    except sqlite3.Error as e:
        print(f"Error querying database: {e}")
        return None
    
    finally:
        # Close the database connection
        conn.close()

#指定数据库，指定表，指定title和dates查找hot
def find_hot_by_title_and_dates(db_path='data_append.db', title='指定标题', dates='指定时间'):
    conn = sqlite3.connect(db_path)  # 连接到 SQLite 数据库
    c = conn.cursor()
    
    # 使用 SQL 查询语句
    c.execute("SELECT hot FROM data_append WHERE title=? AND dates=?", (title, dates))
    
    # 获取查询结果
    result = c.fetchall()
    #print(result)
    conn.close()  # 关闭数据库连接
    
    if result:
        #只要第一个，都是一天的差不多
        return result[0][0]# 返回查询到的 hot 值
    else:
        return int(0)  # 如果未查询到，返回 None 或者其他适当的值
    
@app.route('/hot_trend_flush/', methods=['GET']) 
def make_hot_trend(fromtime = (dt.datetime.now()).strftime('%m%d')):
    #生成时间的列表
    #生成timelist保存过去的五天时间
    time_list = get_past_five_days(input_date=fromtime)

    #生成标题的列表
    title_list = fetch_titles_from_db()
    title_list = remove_duplicates(title_list)
    
    #用来存放对应的热毒值
    hot_0day = []
    hot_1day = []
    hot_2day = []
    hot_3day = []
    hot_4day = []
    
    #获取五天的热度
    for title_one in title_list:
        #第一天的热度值
        hot = find_hot_by_title_and_dates(db_path='data_append.db', title=title_one, dates=time_list[0])
        hot_0day.append(hot)
        #第二天的热度值
        hot = find_hot_by_title_and_dates(db_path='data_append.db', title=title_one, dates=time_list[1])
        hot_1day.append(hot)
        #第三天的热度值
        hot = find_hot_by_title_and_dates(db_path='data_append.db', title=title_one, dates=time_list[2])
        hot_2day.append(hot)
        #第四天的热度值
        hot = find_hot_by_title_and_dates(db_path='data_append.db', title=title_one, dates=time_list[3])
        hot_3day.append(hot)
        #第五天的热度值
        hot = find_hot_by_title_and_dates(db_path='data_append.db', title=title_one, dates=time_list[4])
        hot_4day.append(hot)
    #保存一次dataframe
    dataframe = pd.DataFrame({'title': title_list})
    dataframe['hot0day'] = hot_0day
    dataframe['hot1day'] = hot_1day
    dataframe['hot2day'] = hot_2day
    dataframe['hot3day'] = hot_3day
    dataframe['hot4day'] = hot_4day
     #保存dataframe到数据库中
    ##创建连接
    conn = sqlite3.connect('data_append.db')
    dataframe.to_sql('hot_trend', con=conn, if_exists='replace', index=False)
    conn.close()
    print('从',fromtime,'开始，','已经保存了热度趋势')










# 定义更新数据的任务函数
def update_data():
    try:
        response = requests.get('http://127.0.0.1:5000/flush_data/')
        response.raise_for_status()
        print('Data updated successfully')
    except requests.RequestException as e:
        print('Data update failed:', e)


# 定义新任务的函数
def fetch_ip_flush():
    try:
        response = requests.get('http://127.0.0.1:5000/ip_flush/')
        response.raise_for_status()
        print('IP flush data fetched successfully')
    except requests.RequestException as e:
        print('IP flush data fetch failed:', e)

# 定义新任务的函数
def hot_trend_flush():
    try:
        response = requests.get('http://127.0.0.1:5000/hot_trend_flush/')
        response.raise_for_status()
        print('IP flush data fetched successfully')
    except requests.RequestException as e:
        print('hot trend flush failed:', e)


# 创建调度器实例
scheduler = BackgroundScheduler()

# 添加定时任务，每10分钟执行一次
scheduler.add_job(update_data, 'interval', minutes=10)

# 添加新任务，每60分钟执行一次
scheduler.add_job(fetch_ip_flush, 'cron', minute=55)

# 添加新任务，每24小时执行一次
scheduler.add_job(hot_trend_flush, 'interval', days=1)

# 启动调度器
scheduler.start()

###备注：启动时间应卡在10分钟的倍数启动，避免ip和其他数据的更新混乱###




if __name__ == '__main__':
    # 初始化数据
    # get_data_back()
    # make_hot_trend(fromtime = (dt.datetime.now()).strftime('%m%d'))
    app.run(debug=True, port=5000)
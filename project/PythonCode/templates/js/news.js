// 从URL参数中获取newsId
const urlParams = new URLSearchParams(window.location.search);
const newsId = urlParams.get('newsId');

// 使用fetch API从后端获取新闻详情
fetch(`http://localhost:5000/news/${newsId}`)
    .then(response => response.json())
    .then(data => {
        document.getElementById('newsTitle').innerText = data.title;
        document.getElementById('newsUrl').href = data.url;
        document.getElementById('newsSource').innerText = `来源: ${data.source}`;
        document.getElementById('newsTime').innerText = `时间: ${data.times}`;
        const perP = data.perP;
        const perN = data.perN;
        const perM = data.perM;
        var chartDom1 = document.getElementById('emo_point');
        var pieChart = echarts.init(chartDom1);
        var option;

        option = {
            title: {
                text: '情感分析',  // 添加标题
                left: 'center',
                top: 'top',
                textStyle: {
                    fontSize: 24,
                    fontWeight: 'bold',
                    color: '#3e5569'  // 可以根据需要调整标题颜色
                }
            },
            tooltip: {
                trigger: 'item',
                formatter: '{a} <br/>{b}: {c}%'
            },
            legend: {
                orient: 'vertical',
                left: 10,
                data: ['正面', '负面', '中性']
            },
            series: [
                {
                    name: '情感占比',
                    type: 'pie',
                    radius: ['50%', '70%'],
                    avoidLabelOverlap: false,
                    label: {
                        show: false,
                        position: 'center'
                    },
                    emphasis: {
                        label: {
                            show: true,
                            fontSize: '30',
                            fontWeight: 'bold',
                            formatter: function (params) {
                                // 返回带有动态颜色的文本
                                return `{text|${params.name}}`;
                            },
                            rich: {
                                text: {
                                    color: function (params) {
                                        // 根据不同数据项返回不同的颜色
                                        const colorMap = {
                                            '正面': '#FF6CAF',
                                            '负面': '#2D5FDA',
                                            '中性': '#7CC1F4'
                                        };
                                        return colorMap[params.name] || '#3e5569';  // 默认为灰色
                                    }
                                }
                            }
                        }
                    },
                    labelLine: {
                        show: false
                    },
                    data: [
                        {value: perP, name: '正面', itemStyle: { color: '#FF6CAF' } },
                        {value: perN, name: '负面', itemStyle: { color: '#2D5FDA' }},
                        {value: perM, name: '中性', itemStyle: { color: '#7CC1F4' }}
                    ]
                }
            ]
        };
        pieChart.setOption(option);

        var chartDom2 = document.getElementById('hot_trend');
        var lineChart = echarts.init(chartDom2);

        var dates = [
            data.current_date,
            (new Date(Date.now() - 86400000 * 1)).toISOString().slice(5, 10).replace('-', ''),
            (new Date(Date.now() - 86400000 * 2)).toISOString().slice(5, 10).replace('-', ''),
            (new Date(Date.now() - 86400000 * 3)).toISOString().slice(5, 10).replace('-', ''),
            (new Date(Date.now() - 86400000 * 4)).toISOString().slice(5, 10).replace('-', '')
        ];

        var hotTrend = data.hot_trend.slice(); // 复制一份热度数据

        // 根据 news_data.date 处理早于该日期的数据
        for (var i = 0; i < dates.length; i++) {
            if (parseInt(dates[i]) < parseInt(data.date)) {
                hotTrend[i] = null; // 设置为空以避免图上显示点
            }
        }

        var maxValue = Math.max(...hotTrend); // 获取热度数据中的最大值
        var yAxisMax = Math.max(maxValue, 60000); // 确保纵轴最大值至少为60000

        var option2 = {
            title: {
                text: '热度趋势',
                left: 'center',
                top: 'top',
                textStyle: {
                    fontSize: 24,
                    fontWeight: 'bold',
                    color: '#3e5569'  // 可以根据需要调整标题颜色
                }
            },
            tooltip: {
                trigger: 'axis',
                formatter: function (params) {
                    var value = params[0].data;
                    if (value === null) {
                        return '';  // 如果数据为 null，不显示提示
                    } else if (value <= 10000) {
                        return '热度：<=10000';
                    }
                    return '热度：' + value;
                }
            },
            xAxis: {
                type: 'category',
                data: dates
            },
            yAxis: {
                type: 'value',
                min: 0,
                max: yAxisMax,  // 设置 y 轴的最大值
                axisLabel: {
                    formatter: function (value) {
                        if (value === 0) {
                            return '0';
                        } else if (value <= 10000) {
                            return '<=10000';
                        }
                        return value;
                    }
                },
                splitNumber: 5,  // 设置合适的分割段数
                // minInterval: 10000,  // 保证最小间隔为10000
            },
            series: [{
                name: '热度',
                type: 'line',
                data: hotTrend
            }]
        };

        lineChart.setOption(option2);


        var chartDom3 = document.getElementById('word_data');
        var barChart = echarts.init(chartDom3);
        var option3 = {
            title: {
                text: '关键词分析',
                left: 'center',
                top: 'top',
                textStyle: {
                    fontSize: 24,
                    fontWeight: 'bold',
                    color: '#3e5569'  // 根据需要调整标题颜色
                }
            },
            tooltip: {
                trigger: 'axis',
                axisPointer: {
                    type: 'shadow'  // 使用阴影指示器
                }
            },
            legend: {
                data: ['关键词热度', '出现次数'],
                bottom: 10
            },
            xAxis: {
                type: 'category',
                data: [data.keyword1, data.keyword2, data.keyword3],
                axisLabel: {
                    interval: 0,  // 强制显示所有类别
                    rotate: 30    // 如果类别名过长，可以旋转标签
                }
            },
            yAxis: [
                {
                    type: 'value',
                    name: '热度',
                    position: 'left',
                },
                {
                    type: 'value',
                    name: '出现次数',
                    position: 'right',
                    offset: 0,
                }
            ],
            series: [
                {
                    name: '关键词热度',
                    type: 'bar',
                    data: [data.wordHot1, data.wordHot2, data.wordHot3],
                    itemStyle: {
                        color: '#73C0DE'  // 选择一个合适的颜色
                    },
                    yAxisIndex: 0  // 使用第一个y轴（左侧）
                },
                {
                    name: '出现次数',
                    type: 'bar',
                    data: [data.wordNum1, data.wordNum2, data.wordNum3],
                    itemStyle: {
                        color: '#5470C6'  // 选择另一个合适的颜色
                    },
                    yAxisIndex: 1  // 使用第二个y轴（右侧）
                }
            ]
        };

        barChart.setOption(option3);
        
        


    })
    .catch(error => {
        document.getElementById('newsTitle').innerText = '加载出现错误！';
    });



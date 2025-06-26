var chartDom = document.getElementById('wordFlushLine');
var myChart = echarts.init(chartDom);
var option;

var dataHistory = []; // 存储历史数据
var previousHotTopName = ''; // 保存前一个 hotTopName

function fetchDataHistory(){
    fetch('http://127.0.0.1:5000/api/flush_line_history/')
    .then(response => response.json())
    .then(word_latest5 => {
        dataHistory = word_latest5.map(entry => ({ name: entry[0], value: entry[1] }));
    });
}

function fetchDataAndDrawChart() {
    fetch('http://127.0.0.1:5000/api/hot_flush_line/')
        .then(response => response.json())
        .then(word_la_hot => {
            var hotData = word_la_hot;
            var hotTopName = hotData[0]; // 获取关键字
            var hotValue = hotData[2]; // 获取热度
            var hotTime = hotData[1]; // 获取时间

            // 更新关键字
            document.getElementById('hotTopName').innerText = hotTopName;

            // 检查当前 hotTopName 是否与之前的相同
            if (hotTopName !== previousHotTopName) {
                previousHotTopName = hotTopName;
                fetchDataHistory(); // 获取新的历史数据
            }

            // 添加新数据到历史数据中
            dataHistory.push({ name: hotTime, value: hotValue });

            // 限制历史数据最多为 7 个数据点
            if (dataHistory.length > 7) {
                dataHistory.shift(); // 移除最旧的数据
            }

            // 绘制折线图
            option = {
                // title: {
                //     text: 'Hot Flush Line Chart'
                // },
                xAxis: {
                    type: 'category',
                    data: dataHistory.map(entry => entry.name)
                },
                yAxis: {
                    type: 'value'
                },
                tooltip: {
                    trigger: 'axis',
                    formatter: function (params) {
                        return '时间: ' + params[0].name + '<br>' +
                               '热度: ' + params[0].value;
                    }
                },
                series: [{
                    name: 'Hot Value',
                    type: 'line',
                    data: dataHistory.map(entry => entry.value)
                }]
            };

            myChart.setOption(option);
        });
}

// 刷新标签内容
function refreshChart() {
    fetchDataAndDrawChart();
}

// 初始化时获取数据并绘制图表
fetchDataHistory();

fetch('http://127.0.0.1:5000/api/hot_flush_line/')
.then(response => response.json())
.then(word_la_hot => {
    var hotData = word_la_hot;
    previousHotTopName = hotData[0]; // 获取关键字
});

fetchDataAndDrawChart();

// 每 5 秒刷新图表
// setInterval(refreshChart, 600000);

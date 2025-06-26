// 初始化ECharts实例
var myChart = echarts.init(document.getElementById('mapFlush'));

// 动态创建缩放比例显示元素
var zoomLevelDiv = document.createElement('div');
zoomLevelDiv.id = 'zoomLevel';
zoomLevelDiv.innerHTML = '缩放比例: 125%';
document.getElementById('mapFlush').appendChild(zoomLevelDiv);

 // 加载中国地图的 GeoJSON 数据
 fetch('../data/china.json')  // 替换为你保存 GeoJSON 文件的实际路径
 .then(response => response.json())
 .then(chinaJson => {
     // 注册地图数据
     echarts.registerMap('china', chinaJson);

    
    // ECharts配置项
    var option = {
        // title: {
        //     text: '中国人口统计',
        //     subtext: '数据来源：网络',
        //     left: 'center'
        // },
        tooltip: {
            trigger: 'item',
            formatter: '{b}<br/>数量: {c} 人'
        },
        visualMap: {
            min: 0,
            max: 300,
            left: '5%',
            top: '55%',
            text: ['高', '低'],
            calculable: true,
            color: ['#ffb05f', '#d2e850', '#59bfcd']
        },
        series: [
            {
                name: '地区人数统计',
                type: 'map',
                map: 'china',
                roam: true,  // 不允许缩放和平移
                zoom: 1.25,
                // center:[105,55],
                label: {
                    show: true
                },
                data: []
            }
        ]
    };


    // 使用配置项
    myChart.setOption(option);

    // 监听地图缩放事件
    myChart.on('georoam', function(params) {
        var option = myChart.getOption();
        var zoom = option.series[0].zoom * 100;  // 获取缩放比例
        document.getElementById('zoomLevel').innerHTML = `缩放比例: ${zoom.toFixed(0)}%`;  // 更新显示的缩放比例
    });


    // 从服务器获取数据
    fetch('http://127.0.0.1:5000/ip_get/')
                .then(response => response.json())  // 解析JSON数据
                .then(data => {
                    // 映射名称
                    const nameMapping = {
                        '中国台湾': '台湾',
                        '中国香港': '香港',
                        '中国澳门': '澳门'
                    };

                    // 将数据转换为 ECharts 所需格式，并应用名称映射
                    const formattedData = data.map(item => ({
                        name: nameMapping[item[0]] || item[0],  // 如果没有映射，使用原名称
                        value: item[1]
                    }));

                    // 添加南海诸岛数据
                    const southChinaSeaData = { name: '南海诸岛', value: 0 };
                    formattedData.push(southChinaSeaData);

                    // 计算数据的最大值和最小值
                    const maxValue = Math.max(...formattedData.map(item => item.value));
                    const minValue = Math.min(...formattedData.map(item => item.value));

                    // 更新ECharts的数据
                    myChart.setOption({
                        visualMap: {
                            min: minValue,
                            max: maxValue,
                            left: '5%',
                            top: '55%',
                            text: ['高', '低'],
                            calculable: true,
                            color: ['#ffb05f', '#d2e850', '#59bfcd']
                        },
                        series: [
                            {
                                name: '地区人数统计',
                                data: formattedData  // 更新数据
                            }
                        ]
                    });
                })
    .catch(error => {
        console.error('获取数据失败:', error);
    });
})
.catch(error => {
    console.error('加载地图数据失败:', error);
});

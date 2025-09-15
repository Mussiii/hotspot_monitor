var previousData = [];
var myPieChart;

function pie_flush() {
    fetch('http://127.0.0.1:5000/pie_data_imm/')
        .then(response => response.json())
        .then(data => {
            var labels = Object.keys(data);
            var values = Object.values(data);

            var ctx = document.getElementById('pieChart').getContext('2d');
            myPieChart = new Chart(ctx, {
                type: 'pie',
                data: {
                    labels: ['负面', '中性', '正面'],
                    datasets: [{
                        data: values,
                        backgroundColor: ['#6F4CDF', '#67D8C6', '#F1A16C']
                    }]
                },
                options: {
                    responsive: false,
                    maintainAspectRatio: false,
                    tooltips: {
                        callbacks: {
                            label: function(tooltipItem, data) {
                                var label = data.labels[tooltipItem.index];
                                var value = data.datasets[0].data[tooltipItem.index];
                                return label + ': ' + value;
                            }
                        }
                    }
                }
            });

            previousData = values.slice(); // 存储上一次的数据
        });
}

// 初次加载页面时请求数据并更新饼图
pie_flush();

function updateChart() {
    fetch('http://127.0.0.1:5000/pie_data_imm/')
        .then(response => response.json())
        .then(data => {
            var values = Object.values(data);

            // 实现数据平滑过渡效果
            var duration = 500; // 过渡动画时长
            var startTime = Date.now();
            var endTime = startTime + duration;
            var startValues = previousData.slice();

            function animate() {
                var now = Date.now();
                var progress = Math.min((now - startTime) / duration, 1);
                var interpolatedValues = startValues.map((startValue, index) => {
                    return startValue + (values[index] - startValue) * progress;
                });

                myPieChart.data.datasets[0].data = interpolatedValues;
                myPieChart.update();

                if (now < endTime) {
                    requestAnimationFrame(animate);
                } else {
                    myPieChart.data.datasets[0].data = values;
                    myPieChart.update();
                    previousData = values.slice(); // 更新上一次的数据
                    updateLabels(data);
                }
            }

            animate();
        });
};

function updateLabels(data) {
    // 更新标签内所有内容
    var labels = document.querySelectorAll('#pieChart');
    labels.forEach((label, index) => {
        label.textContent = data[index];
    });
}


// updateChart()
// 每5秒调用一次 updateChart() 函数
// setInterval(updateChart, 600000);

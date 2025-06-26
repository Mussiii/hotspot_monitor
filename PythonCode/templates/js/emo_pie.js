const data = {
    labels: ['正面', '负面', '中性'],
    datasets: [{
        data: [/*导入到这里*/],
        backgroundColor: ['#FF6CAF', '#2D5FDA', '#7CC1F4']
    }]
};

fetch('http://127.0.0.1:5000/get_list/')
    .then(response => response.json())
    .then(emo_per => {
        data.datasets[0].data = emo_per;
        // 在这里可以使用 data 对象来创建图表或其他操作
        const ctx = document.getElementById('emoPie');
        new Chart(ctx, {
            type: 'pie',
            data: data,
            options: {
                responsive: true,
                maintainAspectRatio: false,
 
            }
        });

    });



    fetch('http://127.0.0.1:5000/get_list/')
    .then(response => response.json())
    .then(emo_per => {
        // 获取文档中的三个标签元素
        const tag1 = document.getElementById('emo_per_pos');
        const tag2 = document.getElementById('emo_per_neg');
        const tag3 = document.getElementById('emo_per_mid');

        // 将列表中的三个数据分别插入到三个标签中
        tag1.textContent = emo_per[0];
        tag2.textContent = emo_per[1];
        tag3.textContent = emo_per[2];
    });

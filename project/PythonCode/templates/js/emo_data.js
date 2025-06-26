fetch('http://127.0.0.1:5000/get_list_data/')
.then(response => response.json())
    .then(data_num => {
        // 获取文档中的三个标签元素
        const tag1 = document.getElementById('emo_data_all');
        const tag2 = document.getElementById('emo_data_pos');
        const tag3 = document.getElementById('emo_data_neg');

        // 将列表中的三个数据分别插入到三个标签中
        tag1.textContent = data_num[0];
        tag2.textContent = data_num[1];
        tag3.textContent = data_num[2];
});
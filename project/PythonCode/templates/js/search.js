document.getElementById('search-form').addEventListener('submit', function (event) {
    event.preventDefault(); // 阻止表单的默认提交行为

    const query = document.getElementById('search-input').value;

    if (query.trim() === '') {
        alert('请输入搜索关键词。');
        return;
    }

    // 使用 Fetch API 向后端发送异步请求
    fetch(`http://localhost:5000/search?query=${encodeURIComponent(query)}`)
        .then(response => response.json()) // 解析 JSON 响应
        .then(data => displayResults(data)) // 显示结果
        .catch(error => console.error('Error:', error));
});

// 显示搜索结果
function displayResults(results) {
    const resultsList = document.getElementById('results-list');
    resultsList.innerHTML = ''; // 清空当前的结果列表

    if (results.length === 0) {
        // 如果没有结果，显示提示信息
        resultsList.innerHTML = '<li>没有找到相关结果。</li>';
        return;
    }

    // 将每个结果添加到结果列表中
    results.forEach(item => {
        // 创建结果行容器
        const rowContainer1 = document.createElement('div');
        rowContainer1.className = 'row-container';

        // 创建标题链接
        const titleLink = document.createElement('a');
        titleLink.className = 'result-title';
        // titleLink.href = `./news.html?newsId=${item.rowId}`;
        titleLink.href = item.url;
        titleLink.target = '_blank'; // 在新标签页中打开链接
        titleLink.style.fontSize = '22px'; // 设置字体大小
        titleLink.textContent = item.title;

        // 创建情感标签
        const emotion = document.createElement('span');
        emotion.className = 'emo';

        // 根据 Predicted_Label 设置情感文字和样式
        switch (item.Predicted_Label) {
            case 0:
                emotion.textContent = '负面';
                emotion.classList.add('tag-negative');
                break;
            case 1:
                emotion.textContent = '中性';
                emotion.classList.add('tag-neutral');
                break;
            case 2:
                emotion.textContent = '正面';
                emotion.classList.add('tag-positive');
                break;
            default:
                emotion.textContent = '未知';
                emotion.classList.add('tag-unknown');
        }

        // 添加标题和情感到行容器1
        rowContainer1.appendChild(titleLink);
        rowContainer1.appendChild(emotion);

        // 创建第二个结果行容器
        const rowContainer2 = document.createElement('div');
        rowContainer2.className = 'row-container1';

        // 创建时间和来源信息
        const time = document.createElement('span');
        time.textContent = `时间：${item.times}`;
        time.className = 'number_data';
        time.id = 'time';

        const source = document.createElement('span');
        source.textContent = `来源：${item.source}`;
        source.className = 'number_data';
        source.id = 'source';

        // 添加时间和来源到行容器2
        rowContainer2.appendChild(time);
        rowContainer2.appendChild(source);

        // 将行容器添加到结果列表
        resultsList.appendChild(rowContainer1);
        resultsList.appendChild(rowContainer2);
    });
}
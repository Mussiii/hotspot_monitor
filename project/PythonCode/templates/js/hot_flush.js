fetch('http://127.0.0.1:5000/hot_flush/')
    .then(response => response.json())
    .then(output => {
        const wrap = document.getElementById("hot_flush"); // 获取包含所有热搜项的容器
        const tagElements = wrap.getElementsByClassName("redirect_link"); // 获取所有热搜项的 <a> 标签
        const tagElements1 = wrap.getElementsByClassName("eventTag");

        // 确保数据长度与标签数量匹配
        if (tagElements.length === output.length) {
            output.forEach((item, index) => {
                const title = item[0]; // 获取标题
                const url = item[1];   // 获取 URL
                const sentiment = item[2]; // 获取情感值

                const tag = tagElements[index]; // 获取对应的 <a> 标签
                tag.href = url;                 // 设置 <a> 标签的 href 属性
                tag.textContent = title;        // 设置 <a> 标签的文本内容
                

                // 根据情感值决定情感文本
                switch (sentiment) {
                    case 0:
                        sentimentText = '负面';
                        sentimentClass = 'tag-negative';
                        break;
                    case 1:
                        sentimentText = '中性';
                        sentimentClass = 'tag-neutral';
                        break;
                    case 2:
                        sentimentText = '正面';
                        sentimentClass = 'tag-positive';
                        break;
                    default:
                        sentimentText = '未知';
                        sentimentClass = ''; // 没有对应的类
                }
                const tag1 = tagElements1[index];
                tag1.textContent = sentimentText;
                if (sentimentClass) {
                    tag1.classList.add(sentimentClass);
                }
            });
        } else {
            console.error('数据长度与标签数量不匹配');
        }
    })
    .catch(error => console.error('发生错误:', error));
fetch('http://127.0.0.1:5000/api/get_ciyun/')
    .then(response => response.json())
    .then(output => {
        const wrap = document.getElementById("wrap");
        const tagElements = wrap.getElementsByClassName("tag");

        if (tagElements.length === output.length) {
            output.forEach((item, index) => {
                const keyword = item[0];
                const count = item[1];
                const percentage = item[3];

                const a = tagElements[index];
                a.title = "词频:" + count + "\n占比:" + percentage + "%";
                a.textContent = keyword;
                a.href = '#'
            });
        } else {
            console.error('数据长度与标签数量不匹配');
        }
    })
    .catch(error => console.error('发生错误:', error));
function updateData() {
    fetch('http://127.0.0.1:5000/flush_data/')
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            // 处理成功的响应，可以在这里更新数据
            console.log('1');
        })
        .catch(error => {
            // 处理请求失败的情况
            console.error('Fetch error:', error);
        });
}

function saveLastUpdateTime() {
    localStorage.setItem('lastUpdateTime', new Date().getTime().toString());
}

function loadLastUpdateTime() {
    return localStorage.getItem('lastUpdateTime');
}

document.addEventListener('DOMContentLoaded', function() {
    let lastUpdateTime = loadLastUpdateTime();
    if (lastUpdateTime) {
        let currentTime = new Date().getTime();
        let elapsedTime = currentTime - parseInt(lastUpdateTime, 10);
        if (elapsedTime >= 600000) {
            updateData();
        }
    }

    // 每10分钟更新一次数据
    setInterval(() => {
        updateData();
        saveLastUpdateTime();
    }, 600000);

    // 在页面刷新或关闭时保存最后更新时间
    window.addEventListener('beforeunload', saveLastUpdateTime);
});
function fetchWordSpeed() {
    fetch('http://127.0.0.1:5000/api/hot_word_speed/')
    .then(response => response.json())
    .then(word_speed_top4 => {
        const tag1 = document.getElementById('word_flush1');
        const tag2 = document.getElementById('word_flush2');
        const tag3 = document.getElementById('word_flush3');
        const tag4 = document.getElementById('word_flush4');
        const tag5 = document.getElementById('word_flush5');

        tag1.textContent = word_speed_top4[0][0];
        tag2.textContent = word_speed_top4[1][0];
        tag3.textContent = word_speed_top4[2][0];
        
        tag4.textContent = word_speed_top4[3][0];
        tag5.textContent = word_speed_top4[4][0];
    });
}

// 初始调用一次
fetchWordSpeed();

// 每五秒刷新一次
// setInterval(fetchWordSpeed, 600000);  // 1000 毫秒等于 1 秒
fetch('http://127.0.0.1:5000/api/get_hot_word/')
    .then(response => response.text())
    .then(data => {
        const parser = new DOMParser();
        const doc = parser.parseFromString(data, 'text/html');
        const hotWordsDiv = doc.getElementById('hot_words');

        const targetDiv = document.getElementById('hotWords');
        targetDiv.appendChild(hotWordsDiv);
    })
    .catch(error => console.error('Error fetching content:', error));
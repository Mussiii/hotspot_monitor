const renderPagination = (currentPage, totalLiCount, totalPages) => {
    const findingElement = document.getElementById('finding');
    findingElement.innerHTML = '';
    
    const start = (currentPage - 1) * 30;
    const end = Math.min(currentPage * 30, totalLiCount);
    const liElements = document.querySelectorAll('#hotEventRanking ul li');
    liElements.forEach((li, index) => {
        if (index >= start && index < end) {
            li.style.display = 'block';
        } else {
            li.style.display = 'none';
        }
    });
    
    if (currentPage < totalPages-5) {
        const startPage = Math.max(1, currentPage - 2);
        const endPage = Math.min(startPage + 4, totalPages);
    
        for (let i = startPage; i <= endPage; i++) {
            const pageLink = document.createElement('a');
            pageLink.textContent = i;
            pageLink.id = 'mark';
            if (i === currentPage) {
                pageLink.id = 'markC'; 
                // 添加当前页样式
            }
            pageLink.addEventListener('click', () => {
                renderPagination(i, totalLiCount, totalPages);
            });
            findingElement.appendChild(pageLink);
        }
    
        if (endPage < totalPages) {
            const ellipsis = document.createElement('div');
            ellipsis.textContent = '...';
            ellipsis.id = 'mark1';
            findingElement.appendChild(ellipsis);
        }
    
        const prevPage = document.createElement('a');
        prevPage.textContent = '上一页';
        prevPage.id = 'mark2';
        prevPage.addEventListener('click', () => {
            const prev = Math.max(1, currentPage - 1);
            renderPagination(prev, totalLiCount, totalPages);
        });
        findingElement.insertBefore(prevPage, findingElement.firstChild);
    
        const nextPage = document.createElement('a');
        nextPage.textContent = '下一页';
        nextPage.id = 'mark2';
        nextPage.addEventListener('click', () => {
            const next = Math.min(totalPages, currentPage + 1);
            renderPagination(next, totalLiCount, totalPages);
        });
        findingElement.appendChild(nextPage);
    
        const searchInput = document.createElement('input');
        searchInput.setAttribute('type', 'text');
        searchInput.setAttribute('placeholder', '跳转');
        searchInput.addEventListener('change', () => {
            const pageNumber = parseInt(searchInput.value);
            if (pageNumber >= 1 && pageNumber <= totalPages) {
                renderPagination(pageNumber, totalLiCount, totalPages);
            } else {
                alert('请输入有效页数！');
            }
        });
        findingElement.appendChild(searchInput);
    
        const lastThreePages = totalPages - 2;
        for (let i = Math.max(startPage, lastThreePages); i <= totalPages; i++) {
            const pageLink = document.createElement('a');
            pageLink.textContent = i;
            pageLink.id = 'mark';
            pageLink.addEventListener('click', () => {
                renderPagination(i, totalLiCount, totalPages);
            });
            findingElement.insertBefore(pageLink, nextPage);
        } 
    }
    else {
        const startPage = Math.max(1, totalPages-6);
        const endPage = Math.min(totalPages);
    
        for (let i = startPage; i <= endPage; i++) {
            const pageLink = document.createElement('a');
            pageLink.textContent = i;
            pageLink.id = 'mark';
            if (i === currentPage) {
                pageLink.id = 'markC'; 
                // 添加当前页样式
            }
            pageLink.addEventListener('click', () => {
                renderPagination(i, totalLiCount, totalPages);
            });
            findingElement.appendChild(pageLink);
        }
        
        const prevPage = document.createElement('a');
        prevPage.textContent = '上一页';
        prevPage.id = 'mark2';
        prevPage.addEventListener('click', () => {
            const prev = Math.max(1, currentPage - 1);
            renderPagination(prev, totalLiCount, totalPages);
        });
        findingElement.insertBefore(prevPage, findingElement.firstChild);
    
        const nextPage = document.createElement('a');
        nextPage.textContent = '下一页';
        nextPage.id = 'mark2';
        nextPage.addEventListener('click', () => {
            const next = Math.min(totalPages, currentPage + 1);
            renderPagination(next, totalLiCount, totalPages);
        });
        findingElement.appendChild(nextPage);
    
        const searchInput = document.createElement('input');
        searchInput.setAttribute('type', 'text');
        searchInput.setAttribute('placeholder', '跳转');
        searchInput.addEventListener('change', () => {
            const pageNumber = parseInt(searchInput.value);
            if (pageNumber >= 1 && pageNumber <= totalPages) {
                renderPagination(pageNumber, totalLiCount, totalPages);
            } else {
                alert('请输入有效页数！');
            }
        });
        findingElement.appendChild(searchInput);
    }
};

fetch('http://127.0.0.1:5000/api/hot_topics/')
    .then(response => response.text())
    .then(data => {
        const parser = new DOMParser();
        const doc = parser.parseFromString(data, 'text/html');
        const ulElement = doc.querySelector('ul');
        
        const targetElement = document.getElementById('hotEventRanking');
        targetElement.appendChild(ulElement);
        
        const totalLiCount = document.querySelectorAll('#hotEventRanking ul li').length;
        const totalPages = Math.ceil(totalLiCount / 30);
        
        renderPagination(1, totalLiCount, totalPages);
    })
    .catch(error => {
        console.error('Error fetching and importing list:', error);
    });

document.addEventListener("DOMContentLoaded", function() {
    var tabItems = document.querySelectorAll("#search-tab li");
    
    tabItems.forEach(function(item) {
        item.addEventListener("click", function() {
            var dataType = this.getAttribute("data-type");
            var url = "http://127.0.0.1:5000/api/hot_topics/";
    
            if (dataType === "0") {
                url = "http://127.0.0.1:5000/api/hot_topics/";
            } else if (dataType === "1") {
                url = "http://127.0.0.1:5000/api/hot_topics_positive/";
            } else if (dataType === "3") {
                url = "http://127.0.0.1:5000/api/hot_topics_negative/";
            }
    
            var xhr = new XMLHttpRequest();
            xhr.open("GET", url, true);
            xhr.onreadystatechange = function() {
                if (xhr.readyState == 4 && xhr.status == 200) {
                    var responseData = xhr.responseText;
                    document.getElementById("hotEventRanking").innerHTML = responseData;
                    const totalLiCount = document.querySelectorAll('#hotEventRanking ul li').length;
                    const totalPages = Math.ceil(totalLiCount / 30);
                    renderPagination(1, totalLiCount, totalPages);
                }
            };
    
            xhr.send();
        });
    });
});






// document.addEventListener('DOMContentLoaded', function() {
//     // 获取当前页面的 URL 地址
//     const currentUrl = window.location.href;

//     // 获取传递过来的 URL 参数
//     const urlParams = new URLSearchParams(window.location.search);
//     const apiUrl = urlParams.get('http://127.0.0.1:5000/api/hot_topics/');

//     // 获取要隐藏的 a 标签
//     const aTag = document.querySelector('#hotpoint a');

//     // 检查当前页面的 URL 是否包含传递的 API URL
//     if (currentUrl.includes(apiUrl)) {
//         // 隐藏指定的 a 标签
//         aTag.style.display = 'none';
//     }
// });








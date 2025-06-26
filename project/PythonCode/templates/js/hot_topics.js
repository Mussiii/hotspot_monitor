document.addEventListener("DOMContentLoaded", function() {
    // 添加事件监听，点击标题文字自动跳转
    var titleElements = document.querySelectorAll("#hotpoint li");
    console.log("00");
    titleElements.forEach(function(titleElement, index) {
        console.log("Looping... Element index: " + index)
        titleElement.addEventListener("click", function() {
            console.log("11");
            var linkElement = this.nextElementSibling;
            if (linkElement) {
                linkElement.click();
            } else {
                console.log("No next sibling element found");
            }
            // 触发链接点击事件
        });
    });
    
    //  隐藏“跳转”链接
    var linkElements = document.querySelectorAll(".redirect_link");
    linkElements.forEach(function(linkElement) {
        linkElement.style.display = "none";
    });


    //         document.getElementById("hotEventRanking").addEventListener("click", function(event) {
    //             if (event.target.tagName === "LI") {
    //                 console.log("1"); 
    //             }
    //         });


    // function simulateClick(element) {
    //     var event = new MouseEvent('click', {
    //         view: window,
    //         bubbles: true,
    //         cancelable: true
    //     });
    //     element.dispatchEvent(event);
    // }
    // simulateClick(titleElements);


});


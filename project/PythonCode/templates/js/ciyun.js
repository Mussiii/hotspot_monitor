// 检查格式的
'use strict';
const _baseAngle = Math.PI/ 360,
       R = 140;
let speed = 0.003,
    // 速度因子
    angleX = speed * _baseAngle,
    angleY = -speed * _baseAngle,
    _focalLength = R * 1.5;//150*1.5=225

function calculateAngles(i, len, R) {
  var angleA = Math.acos((2 * (i + 1) - 1) / len - 1);
  var angleB = angleA * Math.sqrt(len * Math.PI);
  var z = R * Math.cos(angleA);
  var y = R * Math.sin(angleA) * Math.sin(angleB);
  var x = R * Math.sin(angleA) * Math.cos(angleB);

  return { angleA, angleB, z, y, x };
}

function Tag(data, x, y, z, options){
    this.options = options;
    this.dataArr = options.data;
    this.data = data;
    this.x = x;
    this.y = y;
    this.z = z;
}
// 定义一个名为 Initialization 的函数，接受一个参数 options
function Initialization(options) {
    // 将传入的 options 对象赋值给函数对象的 options 属性
    this.options = options;
    // 将 options 对象中的 container 属性赋值给函数对象的 container 属性
    this.container = options.container;
    // 将 options 对象中的 data 属性赋值给函数对象的 dataArr 属性
    this.dataArr = options.data;
    // 调用初始化方法
    this.init();
}
// 初始化方法
Initialization.prototype.init = function(){
    // 获取数据数组的长度
    let len = this.dataArr.length;
    // 创建一个空数组，用于存储新创建的标签对象
    let newTags = [];

    // 遍历数据数组
    for(let i = 0; i < len; i++){
        // 计算标签在三维空间中的位置坐标
        // i = 1 len = 58为例
        var { angleA, angleB, z, y, x } = calculateAngles(i, len, R);
        // 生成随机颜色，并将其应用到标签样式中
        var color = "#" + Math.floor(Math.random()*0xffffff).toString(16);
        this.dataArr[i].style.color = color;
        // 创建标签对象，并传入位置坐标和选项参数
        var newtag = new Tag(this.dataArr[i], x, y, z, this.options);
        // 移动标签到预定位置
        newtag.move();
        // 将新创建的标签对象添加到数组中
        newTags.push(newtag);
        // 执行可能的动画效果
        this.animate();
    }
    // 将新创建的标签对象数组赋值给函数对象的 newTags 属性
    this.newTags = newTags;
}


Initialization.prototype.rotateX = function(){
    // 计算旋转的三角函数值
    let cos = Math.cos(angleX),
        sin = Math.sin(angleX);

    // 遍历标签数组    
    this.newTags.forEach((tag) => {
        // 执行旋转
        let y = tag.y * cos - tag.z *sin,
            z = tag.z * cos + tag.y * sin;
            
        // 更新标签的坐标
        tag.y = y;
        tag.z = z;
    })

}

Initialization.prototype.rotateY = function(){
    let cos = Math.cos(angleY),
        sin = Math.sin(angleY);
    this.newTags.forEach((tag) => {
        let 
            x = tag.x * cos - tag.z *sin,
            z = tag.z * cos + tag.x * sin;
            // 将X轴的变换暂停，实现椭球的不变
        tag.x = x;
        tag.z = z;
    })
}
Initialization.prototype.animate = function(){
    var that = this;
    setInterval(function(){
        that.rotateX();
        that.rotateY();
        that.newTags.forEach((tag)=> {
            tag.move();

        })
    },20);


}


// 旋转时远近字体变化（透视）
Tag.prototype.move = function(){
    // 获取数组长度
    var len  = this.dataArr.length;//58
    // 计算缩放比例
    var scale = _focalLength /(_focalLength - this.z);
    // 计算透明度（Alpha）
    var alpha = (this.z + R)/(2 * R);
    // 更新DOM元素的样式
    this.data.style.left = this.x + this.x + 150 + 'px';
    this.data.style.top = this.y + 'px';
    this.data.style.fontSize = 16 * scale + 'px';
    this.data.style.opacity = alpha + 0.5;
}

window.onload = function(){
    let tags = document.querySelectorAll('a.tag');
    let wrap = document.getElementById('wrap');

    let options = {
        data:tags,
        container:wrap
    }
    let tagCloud = new Initialization(options);
    document.addEventListener('mousemove', function(e){
        angleY = 3* (e.clientX/ document.body.getBoundingClientRect().width - 0.5) * speed * _baseAngle ;
        angleX = 2* (e.clientY/ document.body.getBoundingClientRect().height - 0.5) * speed * _baseAngle;
    
    
    
    })
}

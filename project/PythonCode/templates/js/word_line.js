fetch('http://127.0.0.1:5000/api/line_data/')
.then(response => response.json())
.then(data => {
    var ctx = document.getElementById('word_line').getContext('2d');

    var lineChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.recent_times,
            datasets: [
                {
                    label: '负面',
                    data: data.zero_counts,
                    borderColor: '#2748C2',
                    fill: false
                },
                // {
                //     label: '中性',
                //     data: data.one_counts,
                //     borderColor: '#EB8F3A',
                //     fill: false
                // },
                {
                    label: '正面',
                    data: data.two_counts,
                    borderColor: '#DE4BAD',
                    fill: false
                }
            ]
        },
        options: {
            responsive: false,
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
});

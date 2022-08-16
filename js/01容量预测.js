//柱状图模块1
$.get('/jiance_echarts').done
(function (data)
 {
    var myChart = echarts.init(document.querySelector('.bar .chart'))

    var option = {
        color: ["#2f89cf"],

        tooltip: {
            trigger: 'axis',
            axisPointer: {
                type: 'shadow'
            }
        },
        grid: {
            left: '0%',
            right: '2%',
            top: '10px',
            bottom: '2%',
            containLabel: true
        },
        xAxis: [
            {
                type: 'category',
                data: data.categories[:6],
                axisTick: {
                    alignWithLabel: true
                },
                axisLabel: {
                    color: "rgba(255,255,255,.6) ",
                    fontSize: "12"
                },
                axisLine: {
                    show: false
                }
            }

        ],

        yAxis: [
            {
                type: 'value',
                max: 2000,

                axisLabel: {
                    color: "rgba(255,255,255,.6) ",
                    fontSize: "12"
                },
                axisLine: {
                    lineStyle: {
                        color: "rgba(255,255,255,.1)",
                        width: 2
                    }
                },
                // y轴分割线的颜色
                splitLine: {
                    lineStyle: {
                        color: "rgba(255,255,255,.1)"
                    }
                }
            }
        ],

        series: [
            {
                name: '容量',
                type: 'bar',
                barWidth: '40%',
                data: data.data_jiance_echarts[:6],
                itemStyle: {
                    // 修改柱子圆角
                    barBorderRadius: 5
                }
            }
        ]
    };

    myChart.setOption(option);
    // 4. 让图表跟随屏幕自动的去适应
    window.addEventListener("resize", function (data) {
        myChart.resize();
    });
})();


//柱状图模块2
$.get('/jiance_echarts').done
(function (data) {
    var myChart = echarts.init(document.querySelector('.bar2 .chart'))

    var option = {
        color: ["#2f89cf"],

        tooltip: {
            trigger: 'axis',
            axisPointer: {
                type: 'shadow'
            }
        },
        grid: {
            left: '0%',
            right: '2%',
            top: '10px',
            bottom: '2%',
            containLabel: true
        },
        xAxis: [
            {
                type: 'category',
                data: data.categories[6:12],
                axisTick: {
                    alignWithLabel: true
                },
                axisLabel: {
                    color: "rgba(255,255,255,.6) ",
                    fontSize: "12"
                },
                axisLine: {
                    show: false
                }
            }

        ],

        yAxis: [
            {
                type: 'value',
                max: 2000,
                axisLabel: {
                    color: "rgba(255,255,255,.6) ",
                    fontSize: "12"
                },
                axisLine: {
                    lineStyle: {
                        color: "rgba(255,255,255,.1)",
                        width: 2
                    }
                },
                // y轴分割线的颜色
                splitLine: {
                    lineStyle: {
                        color: "rgba(255,255,255,.1)"
                    }
                }
            }
        ],

        series: [
            {
                name: '容量',
                type: 'bar',
                barWidth: '40%',
                data: data.data_jiance_echarts[6:12],
                itemStyle: {
                    // 修改柱子圆角
                    barBorderRadius: 5
                }
            }
        ]
    };

    myChart.setOption(option);
    // 4. 让图表跟随屏幕自动的去适应
    window.addEventListener("resize", function (data) {
        myChart.resize();
    });
})();

//柱状图模块3
$.get('/jiance_echarts').done
(function (data) {
    var myChart = echarts.init(document.querySelector('.bar3 .chart'))

    var option = {
        color: ["#2f89cf"],

        tooltip: {
            trigger: 'axis',
            axisPointer: {
                type: 'shadow'
            }
        },
        grid: {
            left: '0%',
            right: '2%',
            top: '10px',
            bottom: '2%',
            containLabel: true
        },
        xAxis: [
            {
                type: 'category',
                data: data.categories[13:19],
                axisTick: {
                    alignWithLabel: true
                },
                axisLabel: {
                    color: "rgba(255,255,255,.6) ",
                    fontSize: "12"
                },
                axisLine: {
                    show: false
                }
            }

        ],

        yAxis: [
            {
                type: 'value',
                max: 2000,
                axisLabel: {
                    color: "rgba(255,255,255,.6) ",
                    fontSize: "12"
                },
                axisLine: {
                    lineStyle: {
                        color: "rgba(255,255,255,.1)",
                        width: 2
                    }
                },
                // y轴分割线的颜色
                splitLine: {
                    lineStyle: {
                        color: "rgba(255,255,255,.1)"
                    }
                }
            }
        ],

        series: [
            {
                name: '容量',
                type: 'bar',
                barWidth: '40%',
                data: data.data_jiance_echarts[13:19],
                itemStyle: {
                    // 修改柱子圆角
                    barBorderRadius: 5
                }
            }
        ]
    };

    myChart.setOption(option);
    // 4. 让图表跟随屏幕自动的去适应
    window.addEventListener("resize", function (data) {
        myChart.resize();
    });
})();

//柱状图模块4
$.get('/jiance_echarts').done
(function () {
    var myChart = echarts.init(document.querySelector('.bar4 .chart'))

    var option = {
        color: ["#2f89cf"],

        tooltip: {
            trigger: 'axis',
            axisPointer: {
                type: 'shadow'
            }
        },
        grid: {
            left: '0%',
            right: '2%',
            top: '10px',
            bottom: '2%',
            containLabel: true
        },
        xAxis: [
            {
                type: 'category',
                data: data.categories[19:],
                axisTick: {
                    alignWithLabel: true
                },
                axisLabel: {
                    color: "rgba(255,255,255,.6) ",
                    fontSize: "12"
                },
                axisLine: {
                    show: false
                }
            }

        ],

        yAxis: [
            {
                type: 'value',
                max: 2000,
                axisLabel: {
                    color: "rgba(255,255,255,.6) ",
                    fontSize: "12"
                },
                axisLine: {
                    lineStyle: {
                        color: "rgba(255,255,255,.1)",
                        width: 2
                    }
                },
                // y轴分割线的颜色
                splitLine: {
                    lineStyle: {
                        color: "rgba(255,255,255,.1)"
                    }
                }
            }
        ],

        series: [
            {
                name: '容量',
                type: 'bar',
                barWidth: '40%',
                data: data.data_jiance_echarts[19:],
                itemStyle: {
                    // 修改柱子圆角
                    barBorderRadius: 5
                }
            }
        ]
    };

    myChart.setOption(option);
    // 4. 让图表跟随屏幕自动的去适应
    window.addEventListener("resize", function (data) {
        myChart.resize();
    });
})();


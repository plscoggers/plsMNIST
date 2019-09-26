
var mousePressed = false;
var lastX, lastY;
var drawStarted = false;

jQuery(document).ready(function() {
    var canvas = document.getElementById('paint');
    var ctx = canvas.getContext('2d');
    ctx.lineWidth = 5;
    var offset = jQuery('#paint').offset()

    var intervalID = window.setInterval(predictImage, 5000);

    $('#paint').mousedown(function (e){
        mousePressed = true;
        Draw(e.pageX - offset.left, e.pageY - offset.top, false);
    });

    $('#paint').mousemove(function (e){
        if(mousePressed)
        {
            Draw(e.pageX - offset.left, e.pageY - offset.top, true);
        }
    });

    $('#paint').mouseup(function(e){
        mousePressed = false;
    });

    $('#paint').mouseleave(function (e){
        mousePressed = false;
    });

    function Draw(x, y, isDown) 
    {
        if(isDown)
        {
            drawStarted = true;
            ctx.beginPath();
            ctx.moveTo(lastX, lastY);
            ctx.lineTo(x,y);
            ctx.stroke();
        }
        lastX = x;
        lastY = y;
    };

    function predictImage()
    {
        console.log('Hello world');
        console.log(drawStarted);
        var dataURL = canvas.toDataURL('jpg');
        console.log(dataURL);
        var jqxhr = $.post('/predict',{file: dataURL}).done(function(result){
            document.getElementById('prediction').value = result['Prediction'];
            document.getElementById('confidence').value = result['Confidence'];
        })
        .fail(function(){
            console.log("failed");
        })
    }

});

function clearArea(){
    drawStarted = false;
    var canvas = document.getElementById('paint'),
        ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
};
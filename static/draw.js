jQuery(document).ready(function() {

    var intervalID = window.setInterval(predictImage, 2500);

    var canvas = new fabric.Canvas('paint');
    canvas.isDrawingMode = true;
    canvas.freeDrawingBrush.width = 10;
    canvas.freeDrawingBrush.color = "#000000";

    $("#clearbutton").click(function() {
        canvas.clear();
    });

    function predictImage()
    {
        var dataURL = canvas.toDataURL();
        var jqxhr = $.post('/predict',{file: dataURL}).done(function(result){
            document.getElementById('prediction').value = result['Prediction'];
        })
        .fail(function(){
            console.log("failed");
        })
    }

});


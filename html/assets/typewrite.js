function type(message, id, speed) {
    let output = document.getElementById(id), i = 0;
    message.split("").forEach(function (letter) {
        setTimeout(function () {
            output.innerHTML += letter;
        }, i++ * speed);
    });
}
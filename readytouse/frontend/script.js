function uploadImage() {
    let input = document.getElementById("imageInput");
    if (input.files.length === 0) {
        alert("Please select an image first!");
        return;
    }

    let formData = new FormData();
    formData.append("image", input.files[0]);

    fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("predictionResult").innerText = "Predicted: " + data.prediction;
    })
    .catch(error => {
        console.error("Error:", error);
    });
}

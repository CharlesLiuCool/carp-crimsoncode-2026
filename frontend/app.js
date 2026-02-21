// Upload hospital CSV to server
function trainHospital() {
    const fileInput = document.getElementById("fileInput");
    if (!fileInput.files.length) {
        alert("Please select a CSV file first");
        return;
    }

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    fetch("http://127.0.0.1:8000/upload", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("status").innerText = `Upload status: ${JSON.stringify(data)}`;
    })
    .catch(err => console.error(err));
}

// Call the public model to get a prediction
function testPrediction() {
    fetch("http://127.0.0.1:8000/predict")
    .then(response => response.json())
    .then(data => {
        document.getElementById("prediction").innerText = `Test prediction: ${data.prediction.toFixed(4)}`;
    })
    .catch(err => console.error(err));
}
async function trainHospital() {
    document.getElementById("status").innerText = "Training and uploading weights...";
    const formData = new FormData();
    
    // simulate uploading a weight file
    const response = await fetch("/upload", {
        method: "POST",
        body: formData
    });
    
    const data = await response.json();
    document.getElementById("status").innerText = `Status: ${data.status}`;
}
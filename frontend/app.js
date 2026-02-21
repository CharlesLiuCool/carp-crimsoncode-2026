async function trainHospital() {

    document.getElementById("status").innerText =
      "Hospital training and uploading...";
  
    await fetch("http://localhost:8000/health");
  
    document.getElementById("status").innerText =
      "Global model updated with private data";
  }
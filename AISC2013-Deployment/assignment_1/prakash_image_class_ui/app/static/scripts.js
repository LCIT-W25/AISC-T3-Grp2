document.addEventListener("DOMContentLoaded", function () {
  const modelSelect = document.getElementById("model-select");
  const uploadForm = document.querySelector("form");
  const submitButton = uploadForm.querySelector("button[type='submit']");
  const imageUpload = document.getElementById("image-upload");
  const previewImage = document.getElementById("preview-image");
  const fileNameDisplay = document.getElementById("file-name");

  modelSelect.addEventListener("change", function () {
    const selectedModel = modelSelect.value;

    if (selectedModel === "dnn") {
      uploadForm.action = "/predict/dnn"; // Change to your Flask endpoint for DNN
      submitButton.textContent = "Predict DNN"; // Change button text
    } else {
      uploadForm.action = "/predict/knn"; // Change to your Flask endpoint for kNN
      submitButton.textContent = "Predict kNN";
    }
  });

  // Initialize form action and button text on page load
  if (modelSelect.value === "dnn") {
    uploadForm.action = "/predict/dnn";
    submitButton.textContent = "Predict DNN";
  } else {
    uploadForm.action = "/predict/knn";
    submitButton.textContent = "Predict kNN";
  }

  // Handle image preview
  imageUpload.addEventListener("change", function (event) {
    const file = event.target.files[0];
    console.log(file);
    if (file) {
      const reader = new FileReader();
      reader.onload = function (e) {
        previewImage.src = e.target.result;
        previewImage.classList.remove("hidden");
        fileNameDisplay.textContent = `Selected File: ${file.name}`;
        fileNameDisplay.classList.remove("hidden");
      };
      reader.readAsDataURL(file);
    } else {
      previewImage.src = "/placeholder.svg"; // Reset preview if no file is selected
      previewImage.classList.add("hidden");
      fileNameDisplay.textContent = "";
      fileNameDisplay.classList.add("hidden");
    }
  });
});

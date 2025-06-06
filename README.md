# 🌿 Plant Disease Detection

This project uses deep learning to detect plant diseases from leaf images. It includes training and testing notebooks, a sample UI image, and model training history.
🌿 Welcome to **AgriScan** — Your Smart Plant Health Companion!

Plant health is critical to food security, crop yield, and farmer livelihood. **AgriScan** is a cutting-edge mobile and web-based solution designed to **detect plant diseases quickly and accurately** using your smartphone’s camera and advanced machine learning models.


## 📁 Project Structure
Plant_Disease_Dataset/
│
├── Train_plant_disease.ipynb # Notebook to train the CNN model
├── Test_Plant_Disease.ipynb # Notebook to test and evaluate the model
├── main.py # Python script for model inference
├── home_page.jpeg # UI screenshot (optional preview)
├── requirement.txt # Python dependencies
├── training_hist.json # Training history saved as JSON


---

## 🚀 Getting Started

### 1. Clone the repository


git clone https://github.com/achaltaki/plant-disease-detection.git
cd plant-disease-detection/Plant_Disease_Dataset
pip install -r requirement.txt

### 📊 Content Breakdown

- 📁 `train/` - **70,295 images**  
   Images used for training deep learning models.

- 📁 `validation/` - **17,572 images**  
   Used to fine-tune and validate the model during training.

- 📁 `test/` - **33 images**  
   A small set of unseen images meant for final prediction testing and demonstration.

---


import streamlit as st
import tensorflow as tf
import numpy as np


#Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max element

#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["🏠 Home", "ℹ️ About", "🧪 Disease Recognition"])

#Main Page
if(app_mode=="🏠 Home"):
    st.header(" 🌿PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "home_page.jpeg"
    st.image(image_path,use_column_width=True)
    st.markdown("""
# 🌿 Welcome to **AgriScan** — Your Smart Plant Health Companion!

Plant health is critical to food security, crop yield, and farmer livelihood. **AgriScan** is a cutting-edge mobile and web-based solution designed to **detect plant diseases quickly and accurately** using your smartphone’s camera and advanced machine learning models.

---

## 🔍 What is AgriScan?

AgriScan empowers farmers, agronomists, and agricultural officers by transforming a simple plant image into a **diagnostic tool**. Leveraging **deep learning models like MobileNetV2 and ResNet18**, it ensures **fast and precise** detection — even in low-connectivity rural areas, thanks to on-device AI.

---

## 🚀 Key Features

### 📷 Disease Detection via Camera  
Capture or upload plant images for **instant, AI-powered diagnosis** using a trained CNN model.

### 💡 Actionable Treatment Recommendations  
Get **natural and chemical remedies** for the detected disease, along with **preventive care tips**.

### 📚 Scan History Tracker  
Automatically store and review previous scans, predictions, and suggested treatments — enabling informed follow-up.

### 🌐 Optional Enhancements  
- **Multilingual Interface** for local farmer accessibility  
- **Community Forums** to share insights and experiences  
- **Cloud Sync & Admin Dashboard** for agricultural officers  
- **Mapping Support** using Google Maps API for geolocation tagging of disease outbreaks  

---

## 📈 Under the Hood – Technical Highlights

- **Machine Learning**: TensorFlow-based models, optimized as `.tflite` for mobile use  
- **Model Types**: MobileNetV2, ResNet18, EfficientNet  
- **Metrics Tracked**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix  
- **Model Training**: Includes data augmentation, class balancing for robustness  
- **Deployment**: Flutter (Android & iOS), Firebase (Realtime DB, Auth, Storage), Hugging Face datasets  
- **Performance**: Lightweight, fast inference for low-end mobile devices  

---

## 🌟 Why Choose AgriScan?

- ✅ **Accurate**: Powered by advanced deep learning models  
- 🚀 **Fast**: Results in seconds, even offline  
- 🎯 **Targeted**: Remedies based on actual disease type  
- 🌍 **Inclusive**: Designed for farmers of all regions and languages  
- 📱 **Mobile-First**: Lightweight and optimized for rural connectivity  
- 🧠 **Empowering**: Learn, act, and share — all from one platform  

---

## 🧭 How to Get Started

1. 📥 **Upload or Capture an Image**  
   Visit the **Disease Recognition** page in the sidebar and upload a plant image.

2. 🧠 **AI Analysis**  
   Our deep learning model instantly scans and identifies potential diseases.

3. 📋 **Review Results**  
   View disease prediction, suggested remedies, and log it for future reference.

---

## 👥 About Us

We are a team of passionate technologists and agricultural enthusiasts committed to leveraging AI for social impact. With AgriScan, we aim to equip every farmer with a **pocket-sized plant doctor** — promoting sustainable agriculture, reducing misdiagnosis, and boosting yields.

---

🧪 *Early detection. Accurate treatment. Healthier harvests.*  
Welcome to **AgriScan** — where AI meets agriculture 🌱
 """)

#About Project
elif(app_mode=="ℹ️ About"):
    st.header("ℹ️ About")
    st.markdown("""
## 📂 About the Dataset

This dataset has been carefully **curated using offline data augmentation techniques** from the original publicly available crop disease dataset. The original source is available on the referenced GitHub repository.

It contains a rich collection of **~87,000 RGB images** of crop leaves, both **healthy and diseased**, spanning **38 distinct categories**.

---

### 🧬 Dataset Structure

The data is split in an **80/20 ratio** for training and validation while maintaining the directory structure for compatibility with most deep learning frameworks. Additionally, a separate test directory was created for model evaluation and prediction testing.

---

### 📊 Content Breakdown

- 📁 `train/` - **70,295 images**  
   Images used for training deep learning models.

- 📁 `validation/` - **17,572 images**  
   Used to fine-tune and validate the model during training.

- 📁 `test/` - **33 images**  
   A small set of unseen images meant for final prediction testing and demonstration.

---

This structured and well-balanced dataset supports robust training of models for **plant disease classification**, making it ideal for both research and practical deployment.
  """)

#Prediction Page
elif(app_mode=="🧪 Disease Recognition"):
    st.header("🧪 Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image,width=4,use_column_width=True)
    #Predict button
    if(st.button("Predict")):
        st.snow()
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        #Reading Labels
        class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                      'Tomato___healthy']
        st.success("Model is Predicting it's a {}".format(class_name[result_index]))
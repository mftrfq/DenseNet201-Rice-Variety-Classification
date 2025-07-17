import streamlit as st  
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import warnings
import cv2
from rembg import remove
from io import BytesIO
from collections import Counter
import os
import gdown

warnings.filterwarnings("ignore")

# Set page config
st.set_page_config(
    page_title="Rice Variety Classification",
    page_icon="🌾",
    initial_sidebar_state='auto'
)

# Hide footer & main menu
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Load model from Google Drive
@st.cache_resource
def load_model_from_drive(drive_id, filename):
    if not os.path.exists(filename):
        with st.spinner(f"Downloading {filename} from Google Drive..."):
            url = f"https://drive.google.com/uc?id={drive_id}"
            gdown.download(url, filename, quiet=False)
    model = tf.keras.models.load_model(filename)
    return model

# Model list with Drive IDs
model_options = {
    "Transfer Learning E10": ("1AbcTL10xxxxxxxxxxxxxxxxxxxxxxxx", "TL_model_10epoch.keras"),
    "Transfer Learning E20": ("1AbcTL20xxxxxxxxxxxxxxxxxxxxxxxx", "TL_model_20epoch.keras"),
    "Transfer Learning E30": ("1AbcTL30xxxxxxxxxxxxxxxxxxxxxxxx", "TL_model_30epoch.keras"),
    "Non-Transfer Learning E10": ("1AbcNonTL10xxxxxxxxxxxxxxxxxxxx", "nonTL_model_10epoch.keras"),
    "Non-Transfer Learning E20": ("1AbcNonTL20xxxxxxxxxxxxxxxxxxxx", "nonTL_model_20epoch.keras"),
    "Non-Transfer Learning E30": ("1AbcNonTL30xxxxxxxxxxxxxxxxxxxx", "nonTL_model_30epoch.keras"),
}
# Ganti ID di atas dengan ID asli file di Google Drive kamu

# Sidebar
with st.sidebar:
    st.title("RICE VARIETY CLASSIFICATION")
    st.subheader("DenseNet-201")
    st.text("Accurate Rice Variety Classifier. It helps users to easily classify rice based on images.")

    selected_model = st.selectbox("Select Classification Model", list(model_options.keys()))
    drive_id, filename = model_options[selected_model]

    try:
        model = load_model_from_drive(drive_id, filename)
        st.success(f"{selected_model} loaded successfully!")
    except Exception as e:
        st.error(f"Failed to load {selected_model}")
        st.error(str(e))

    img_source = st.radio("Choose image source", ("Upload image", "Sample image"))

# Headline
st.header("🌾RICE VARIETY CLASSIFICATION")
st.write(
    "Tahukah anda? biji padi yang kita kenal sebagai beras merupakan sumber karbohidrat utama bagi sebagian besar penduduk dunia. "
    "Beras tidak hanya menjadi makanan pokok yang menyediakan energi, tetapi juga memiliki peran penting dalam budaya, "
    "ekonomi, dan ketahanan pangan banyak negara, terutama di Asia."
)

# Classes and Info
class_names = ['ciherang', 'ir64', 'mentik']
rice_info = {
    "ciherang": "Ciherang adalah varietas unggul yang banyak ditanam di Indonesia.🍚",
    "ir64": "IR64 adalah varietas hasil pemuliaan yang memiliki produktivitas tinggi.🍚",
    "mentik": "Mentik adalah varietas lokal dengan aroma wangi dan tekstur pulen.🍚"
}
label_colors = {
    'ciherang': (255, 0, 0),
    'ir64': (0, 0, 255),
    'mentik': (0, 255, 0),
}

# Prediction function
def import_and_predict(image_data, model):
    size = (224, 224)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img = np.asarray(image) / 255.0
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape, verbose=0)
    return prediction

# Show variety info
def display_info(predicted_class):
    st.warning(f"{predicted_class.upper()} VARIETY")
    st.write(rice_info[predicted_class])

# Sample images
sample_images = {
    "Ciherang": [
        r'Images/sampel ciherang_1.png',
        r'Images/sampel ciherang_2.png',
        r'Images/sampel ciherang_3.png'
    ],
    "IR64": [
        r'Images/sampel ir64_1.png',
        r'Images/sampel ir64_2.png',
        r'Images/sampel ir64_3.png'
    ],
    "Mentik": [
        r'Images/sampel mentik_1.png',
        r'Images/sampel mentik_2.png',
        r'Images/sampel mentik_3.png'
    ]
}

# Image Selection Logic
if img_source == "Sample image":
    st.sidebar.header("Select a class")
    selected_class = st.sidebar.selectbox("Rice Variety", list(sample_images.keys()))
    st.header(f"Sample of {selected_class} images")
    columns = st.columns(3)
    selected_image = None
    for i, image_path in enumerate(sample_images[selected_class]):
        with columns[i % 3]:
            image = Image.open(image_path)
            st.image(image, caption=f"Sample {i + 1}", use_container_width=True)
            if st.button(f"Select Sample {i + 1}", key=image_path):
                selected_image = image_path

    if selected_image:
        image = Image.open(selected_image).convert('RGB')
        st.image(image, caption=selected_image, use_container_width=True)
        predictions = import_and_predict(image, model)
        confidence = np.max(predictions) * 100
        pred_class = class_names[np.argmax(predictions)]
        st.sidebar.header("🔎RESULT")
        st.sidebar.warning(f"Identified variety : {pred_class.upper()}")
        st.sidebar.info(f"Confidence score : {confidence:.2f}%")
        st.markdown("### 💡Information")
        display_info(pred_class)
    else:
        st.info("Select an image for prediction")

else:
    file = st.file_uploader("Upload an image file...", type=["jpg", "png"])
    if file is None:
        st.text("Please upload an image file")
    else:
        try:
            file_bytes = file.read()
            image_buffer = BytesIO(file_bytes)
            image = Image.open(image_buffer).convert('RGB')
            st.image(image, caption="Uploaded Image", use_container_width=True)

            rembg_buffer = BytesIO(file_bytes)
            output_bytes = remove(rembg_buffer.read())
            img_no_bg = Image.open(BytesIO(output_bytes)).convert("RGB")
            img_np = np.array(img_no_bg)

            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

            object_count = 0
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area >= 300:
                    object_count += 1

            if object_count <= 1:
                predictions = import_and_predict(image, model)
                confidence = np.max(predictions) * 100
                pred_class = class_names[np.argmax(predictions)]
                st.sidebar.header("🔎RESULT")
                st.sidebar.warning(f"Identified variety : {pred_class.upper()}")
                st.sidebar.info(f"Confidence score : {confidence:.2f}%")
                st.markdown("### 💡Information")
                display_info(pred_class)
            else:
                st.info(f"Multiple grains detected: {object_count} objects")
                draw_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                variety_counter = Counter()

                for i in range(1, num_labels):
                    area = stats[i, cv2.CC_STAT_AREA]
                    if area < 300:
                        continue

                    x = stats[i, cv2.CC_STAT_LEFT]
                    y = stats[i, cv2.CC_STAT_TOP]
                    w = stats[i, cv2.CC_STAT_WIDTH]
                    h = stats[i, cv2.CC_STAT_HEIGHT]
                    cx, cy = centroids[i]

                    side = int(max(w, h) * 1.5)
                    cx_int = int(cx)
                    cy_int = int(cy)
                    x1 = max(0, cx_int - side // 2)
                    y1 = max(0, cy_int - side // 2)
                    side = min(side, min(img_np.shape[1] - x1, img_np.shape[0] - y1))

                    crop = img_np[y1:y1 + side, x1:x1 + side]
                    resized = cv2.resize(crop, (224, 224))
                    x_input = tf.expand_dims(resized / 255.0, axis=0)

                    pred = model.predict(x_input, verbose=0)
                    score = tf.nn.softmax(pred[0])
                    label = class_names[np.argmax(score)]
                    color = label_colors.get(label, (0, 255, 255))

                    cv2.rectangle(draw_img, (x1, y1), (x1 + side, y1 + side), color, 2)
                    cv2.putText(draw_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.8, color=color, thickness=2)
                    variety_counter[label] += 1

                st.image(cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB), caption="Classification Result", use_container_width=True)
                st.sidebar.header("🔎SUMMARY")
                for variety, total in variety_counter.items():
                    st.sidebar.write(f"{variety.upper()}: {total} grain(s)")
                st.sidebar.success(f"Total classified: {sum(variety_counter.values())} grain(s)")

        except Exception as e:
            st.error("Error processing the image. Please try again with a valid image file.")
            st.error(str(e))

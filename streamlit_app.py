import streamlit as st  
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import cv2
from rembg import remove
from io import BytesIO
from collections import Counter
import os
import gdown
import warnings

warnings.filterwarnings("ignore")

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

# Load fixed model
@st.cache_resource
def load_model():
    file_id = "14T6m4berh-Z_WjMFaQ07sQDthquWjkyk"
    filename = "TL_model_30epoch.keras"
    if not os.path.exists(filename):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, filename, quiet=False)
    return tf.keras.models.load_model(filename)

model = load_model()

with st.sidebar:
    st.title("RICE VARIETY CLASSIFICATION")
    st.subheader("DenseNet-201 Model")
    st.markdown("Accurate rice variety classifier based on image input.")

st.header("🌾 RICE VARIETY CLASSIFICATION")
st.write(
    "Tahukah anda? biji padi yang kita kenal sebagai beras merupakan sumber karbohidrat utama bagi sebagian besar penduduk dunia. "
    "Beras tidak hanya menjadi makanan pokok yang menyediakan energi, tetapi juga memiliki peran penting dalam budaya, "
    "ekonomi, dan ketahanan pangan banyak negara, terutama di Asia."
)

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

def import_and_predict(image_data, model):
    size = (224, 224)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img = np.asarray(image) / 255.0
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape, verbose=0)
    return prediction

def display_info(predicted_class):
    st.markdown(f"**{predicted_class.upper()} VARIETY**")
    st.write(rice_info[predicted_class])

file = st.file_uploader("Upload an image file...", type=["jpg", "png"])
if file is None:
    st.text("Please upload an image file.")
else:
    try:
        file_bytes = file.read()
        image = Image.open(BytesIO(file_bytes)).convert('RGB')
        st.image(image, caption="Uploaded Image", use_container_width=True)

        rembg_buffer = BytesIO(file_bytes)
        output_bytes = remove(rembg_buffer.read())
        img_no_bg = Image.open(BytesIO(output_bytes)).convert("RGB")
        img_np = np.array(img_no_bg)

        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
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

            st.sidebar.header("🔎 RESULT")
            st.sidebar.markdown(f"**Variety:** {pred_class.upper()}")
            st.sidebar.markdown(f"**Confidence:** {confidence:.2f}%")
            st.markdown("### 💡 Information")
            display_info(pred_class)

        else:
            st.info(f"Multiple grains detected: {object_count} object(s)")
            draw_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            variety_counter = Counter()

            for i in range(1, num_labels):
                x, y, w, h, area = stats[i]
                cx, cy = centroids[i]
                if area < 300:
                    continue

                side = int(max(w, h) * 1.5)
                cx_int, cy_int = int(cx), int(cy)
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
                            fontScale=1.2, color=color, thickness=2)
                variety_counter[label] += 1

            st.image(cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB), caption="Classification Result", use_container_width=True)

            st.sidebar.header("🔎 SUMMARY")
            st.sidebar.markdown(f"**Total Classified:** {sum(variety_counter.values())} grain(s)")
            for variety, count in variety_counter.items():
                st.sidebar.markdown(f"- {variety.upper()}: {count} grain(s)")

    except Exception as e:
        st.error("Error processing the image. Please try again with a valid image file.")
        st.error(str(e))

import streamlit as st
import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim

# ---------------------------
# Background Removal using GrabCut
# ---------------------------
def remove_background(image):
    mask = np.zeros(image.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    height, width = image.shape[:2]
    rect = (10, 10, width - 20, height - 20)  # Margin
    cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
    result = image * mask2[:, :, np.newaxis]
    return result

# ---------------------------
# Histogram Calculation
# ---------------------------
def calculate_histogram(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist

# ---------------------------
# SSIM Calculation
# ---------------------------
def calculate_ssim_score(image1, image2):
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(gray1, gray2, full=True)
    return score

# ---------------------------
# Load Database
# ---------------------------
def load_database(folder, standard_size):
    db = []
    for filename in os.listdir(folder):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(folder, filename)
            img = cv2.imread(path)
            img = remove_background(img)
            img = cv2.resize(img, standard_size)
            db.append((filename, img))
    return db

# ---------------------------
# Streamlit App
# ---------------------------
def main():
    st.title("🖼️ Product Image Similarity Checker")
    st.markdown("Upload an image to check if it matches a product in the database.")
    
    uploaded_file = st.file_uploader("Upload Product Image", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        uploaded_bytes = uploaded_file.read()
        np_arr = np.frombuffer(uploaded_bytes, np.uint8)
        uploaded_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Preprocessing
        st.subheader("🎯 Uploaded Image")
        st.image(uploaded_image, channels="BGR")

        uploaded_image = remove_background(uploaded_image)
        standard_size = (300, 300)
        uploaded_image = cv2.resize(uploaded_image, standard_size)

        # Load DB
        database = load_database("product_images", standard_size)

        best_match = None
        best_score = -1

        for name, db_image in database:
            db_hist = calculate_histogram(db_image)
            uploaded_hist = calculate_histogram(uploaded_image)

            hist_score = cv2.compareHist(uploaded_hist, db_hist, cv2.HISTCMP_CORREL)
            ssim_score = calculate_ssim_score(uploaded_image, db_image)

            total_score = (hist_score * 0.5) + (ssim_score * 0.5)

            if total_score > best_score:
                best_score = total_score
                best_match = (name, db_image, hist_score, ssim_score)

        if best_score < 0.3:
            st.warning("❌ No good match found. Try uploading a clearer image or a different angle.")
        else:
            name, match_img, hist_score, ssim_score = best_match
            st.subheader("✅ Most Similar Image")
            st.image(match_img, channels="BGR", caption=f"Match: {name}")
            st.write(f"📊 Histogram Similarity: `{hist_score:.2f}`")
            st.write(f"🧠 SSIM Similarity: `{ssim_score:.2f}`")
            st.success("🟢 Match found successfully!")

if __name__ == "__main__":
    main()

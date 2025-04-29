import streamlit as st

# MUST be the first Streamlit command
st.set_page_config(
    page_title="Product Verification System",
    page_icon="🛍️",
    layout="wide"
)

# Then import other libraries
import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim
from pyzbar.pyzbar import decode
import pytesseract
from PIL import Image
import re
import torch
from pathlib import Path
import shutil
import glob
import warnings

# ---------------------------
# Configuration
# ---------------------------
# Suppress warnings
warnings.filterwarnings("ignore")

# Set Tesseract path (change this according to your system)
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'  # Windows example

# Define boycott brands for logo detection
BOYCOTT_BRANDS = ['Ferrero Rocher', 'Nescafe', 'Pampers', 'pepsi', 'Milka', 'Kit Kat', 'NIVEA', 'TIDE']

# ---------------------------
# YOLO Model Loading
# ---------------------------
@st.cache_resource
def load_yolo_model():
    """Load YOLO model for logo detection"""
    model = torch.hub.load('ultralytics/yolov5', 'custom', path="C:\\Users\\LENOVO\\Desktop\\finalcvproject\\cvvv\\yolov5\\runs\\train\\exp2\\weights\\best.pt", force_reload=True)
    return model

# ---------------------------
# Image Processing Functions
# ---------------------------
def remove_background(image):
    """Remove image background using GrabCut algorithm"""
    mask = np.zeros(image.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    height, width = image.shape[:2]
    rect = (10, 10, width - 20, height - 20)
    cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    result = image * mask2[:, :, np.newaxis]
    return result

def calculate_histogram(image):
    """Calculate normalized HSV histogram"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    hist = cv2.calcHist([lab], [0, 1, 2], None, [30, 30, 30], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist

def calculate_color_similarity(image1, image2):
    """Calculate color similarity metric"""
    lab1 = cv2.cvtColor(image1, cv2.COLOR_BGR2LAB)
    lab2 = cv2.cvtColor(image2, cv2.COLOR_BGR2LAB)
    mean1 = cv2.mean(lab1)
    mean2 = cv2.mean(lab2)
    color_distance = np.sqrt(sum((a - b) ** 2 for a, b in zip(mean1[:3], mean2[:3])))
    max_possible_distance = 100
    return 1 - (color_distance / max_possible_distance)

def calculate_ssim_score(image1, image2):
    """Calculate structural similarity index"""
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(gray1, gray2, full=True)
    return score

# ---------------------------
# Barcode Functions
# ---------------------------
def preprocess_image(image):
    """Convert image to grayscale and apply thresholding"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def detect_barcode(image):
    """Detect barcode location with pyzbar"""
    try:
        barcodes = decode(image)
        if barcodes:
            barcode = barcodes[0]
            return barcode.rect, barcode.data.decode('utf-8')
    except Exception as e:
        st.warning(f"Barcode detection error: {e}")
    return None, None

def extract_text_near_barcode(image, barcode_rect, barcode_data):
    """Extract text below barcode with fallback OCR"""
    if barcode_data and barcode_data.isdigit():
        return barcode_data
    
    x, y, w, h = barcode_rect
    roi = image[y+h:y+h+100, x:x+w+100]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh_roi = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789'
    text = pytesseract.image_to_string(thresh_roi, config=custom_config)
    numbers = re.findall(r'\d+', text)
    return numbers[0] if numbers else None

# ---------------------------
# Database Functions
# ---------------------------
def load_database(folder, standard_size):
    """Load product images from database folder"""
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
# Logo Detection Functions
# ---------------------------
def detect_logos(image_path, model):
    """Detect logos using YOLO model"""
    results = model(image_path)
    results.save()  # saves in runs/detect/exp/
    
    # Get latest detection folder
    output_dir = sorted(glob.glob('runs/detect/exp*'), key=os.path.getctime)[-1]
    result_path = os.path.join(output_dir, os.path.basename(image_path))
    
    if not os.path.exists(result_path):
        return None, []
    
    # Get detected classes
    detected = results.names
    detected_ids = results.xyxy[0][:, -1].tolist()
    detected_labels = [detected[int(i)] for i in detected_ids]
    
    return result_path, detected_labels

# ---------------------------
# Main Application
# ---------------------------
def main():
    # Load YOLO model once
    yolo_model = load_yolo_model()
    
 
    
    # Custom CSS
    st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
        }
        .header {
            color: #2c3e50;
            text-align: center;
            padding: 1rem;
        }
        .match-oui {
            color: #e74c3c;
            font-weight: bold;
            font-size: 18px;
        }
        .match-non {
            color: #27ae60;
            font-weight: bold;
            font-size: 18px;
        }
        .metric-box {
            border-radius: 5px;
            padding: 10px;
            background: #ffffff;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 10px;
        }
        .tab-content {
            padding: 20px;
            background: #ffffff;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-top: 10px;
        }
        .sidebar .sidebar-content {
            background-color: #2c3e50;
            color: white;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center;'>
        <h1> T_CATCH </h1>
        <p>🕊️ <i>"Boycotter, c'est aimer autrement.. C'est aimer la paix, aimer les innocents qu'on ne voit pas mais qu'on entend, dans le silence des bombes et l'oubli des médias."</i></p>
        <p>Un simple refus d'achat, un choix différent dans un rayon, c'est peut-être rien aux yeux du monde...</p>
        <p>Mais ensemble, ces riens deviennent des voix, des vagues, des actes qui dérangent, qui réveillent et qui transforment.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    **Pourriez-vous télécharger une image du produit afin de :**
    - Vérifier si elle correspond à nos produits boycottés
    - Identifier les logos de marque
    - Valider le numéro de code-barres
    
    Nous vous remercions d'avance pour votre aide.
    """)
    
    # Sidebar settings
    with st.sidebar:
        st.header("⚙️ Settings")
        similarity_threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.9, 0.05)
        show_debug = st.checkbox("Show debug information")
    
    # File uploader
    uploaded_file = st.file_uploader("📤 Télécharger une image de produit", type=["png", "jpg", "jpeg"])
    
    if uploaded_file:
        # Read and process uploaded image
        uploaded_bytes = uploaded_file.read()
        np_arr = np.frombuffer(uploaded_bytes, np.uint8)
        uploaded_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        # Create tabs for different functionalities
        tab1, tab2, tab3 = st.tabs(["🔍 Correspondance Produit", "🖼️ Détection Logo", "📊 Validation Code-barres"])
        
        with tab1:
            st.header("Correspondance Produit")
            
            with st.spinner("Traitement de l'image..."):
                processed_image = remove_background(uploaded_image.copy())
                standard_size = (200, 200)
                resized_image = cv2.resize(processed_image, standard_size)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Image originale")
                    st.image(uploaded_image, channels="BGR", use_column_width=True)
                with col2:
                    st.subheader("Image traitée")
                    st.image(processed_image, channels="BGR", use_column_width=True)
            
            with st.spinner("Recherche dans la base de données..."):
                database = load_database("product_images", standard_size)
                
                meilleure_correspondance = None
                meilleur_score = -1
                details_correspondance = {}
                
                for nom, db_image in database:
                    hist_score = cv2.compareHist(
                        calculate_histogram(resized_image),
                        calculate_histogram(db_image),
                        cv2.HISTCMP_CORREL
                    )
                    color_score = calculate_color_similarity(resized_image, db_image)
                    ssim_score = calculate_ssim_score(resized_image, db_image)
                    
                    score_total = (hist_score * 0.4) + (ssim_score * 0.3) + (color_score * 0.3)
                    
                    if score_total > meilleur_score:
                        meilleur_score = score_total
                        meilleure_correspondance = db_image
                        details_correspondance = {
                            "nom": nom,
                            "image": db_image,
                            "hist_score": hist_score,
                            "ssim_score": ssim_score,
                            "color_score": color_score,
                            "score_total": score_total
                        }
            
            if meilleure_correspondance is not None:
                if meilleur_score >= 0.8:  # You can make this threshold configurable
                    st.markdown('<p class="match-oui">OUI - Malheureusement, ce produit fait partie des articles boycottés. Nous vous encourageons à explorer d\'autres options disponibles. Merci beaucoup pour votre compréhension et votre soutien dans cette démarche.</p>', unsafe_allow_html=True)
                    st.success(f"Confiance de la correspondance : {meilleur_score:.2f}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Votre produit")
                        st.image(resized_image, channels="BGR", use_column_width=True)
                    with col2:
                        st.subheader("Correspondance en base")
                        st.image(details_correspondance['image'], channels="BGR", 
                                 caption=f"Produit: {details_correspondance['nom']}", 
                                 use_column_width=True)
                    
                    st.markdown("### Métriques de similarité")
                    cols = st.columns(4)
                    with cols[0]:
                        st.metric("Similarité histogramme", f"{details_correspondance['hist_score']:.2f}")
                    with cols[1]:
                        st.metric("Similarité SSIM", f"{details_correspondance['ssim_score']:.2f}")
                    with cols[2]:
                        st.metric("Similarité couleur", f"{details_correspondance['color_score']:.2f}")
                    with cols[3]:
                        st.metric("Score total", f"{details_correspondance['score_total']:.2f}")
                else:
                    st.markdown('<p class="match-non">NON - Ce produit n\'est pas boycotté et est disponible à l\'achat. Vous pouvez l\'ajouter à votre panier dès maintenant.</p>', unsafe_allow_html=True)
                    st.warning(f"Le meilleur score était {meilleur_score:.2f} (en dessous du seuil de 0.8)")
        
        with tab2:
            st.header("Détection de Logo")
            
            # Save uploaded image temporarily
            temp_path = "temp_upload.jpg"
            with open(temp_path, "wb") as f:
                f.write(uploaded_bytes)
            
            with st.spinner("Détection des logos en cours..."):
                result_path, detected_labels = detect_logos(temp_path, yolo_model)
                
                if result_path:
                    st.image(result_path, caption="Résultat de détection de logo", use_column_width=True)
                    
                    st.markdown("### Marques détectées:")
                    for brand in detected_labels:
                        st.write(f"🛍️ {brand}")
                    
                    is_boycott = any(brand in BOYCOTT_BRANDS for brand in detected_labels)
                    if is_boycott:
                        st.error("🚫 Ce produit contient une marque **BOYCOTTÉE**!")
                    else:
                        st.success("✅ Aucune marque boycottée détectée.")
                else:
                    st.warning("Aucun logo détecté dans l'image.")
            
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        with tab3:
            st.header("Validation Code-barres")
            
            opencv_image = cv2.cvtColor(uploaded_image, cv2.COLOR_BGR2RGB)
            
            with st.spinner("Détection du code-barres..."):
                barcode_rect, barcode_data = detect_barcode(opencv_image)
                
                if not barcode_rect:
                    st.error("Aucun code-barres détecté. Veuillez essayer avec une image plus nette.")
                else:
                    x, y, w, h = barcode_rect
                    barcode_display = opencv_image.copy()
                    cv2.rectangle(barcode_display, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Code-barres détecté")
                        st.image(barcode_display, caption='Rectangle vert montre le code-barres détecté', 
                                 use_column_width=True)
                    
                    with st.spinner("Extraction du numéro de code-barres..."):
                        extracted_number = extract_text_near_barcode(opencv_image, barcode_rect, barcode_data)
                        
                        with col2:
                            st.subheader("Résultat de validation")
                            if extracted_number:
                                st.write(f"Numéro de code-barres : `{extracted_number}`")
                                
                                if extracted_number.startswith('619'):
                                    st.success("✅ Valide - Ce produit est fièrement fabriqué en Tunisie ! Vous pouvez l'acheter en toute confiance. Nous vous encourageons vivement à soutenir et à privilégier les produits locaux.")
                                elif extracted_number.startswith('729'):
                                    st.error("❌ Invalide - Malheureusement, ce produit fait partie des articles boycottés. Nous vous encourageons à explorer d'autres options disponibles.")
                                else:
                                    st.success("✅ Valide - Bonne nouvelle ! Ce produit n'est pas boycotté et est disponible à l'achat.")
                            else:
                                st.warning("⚠️ Impossible d'extraire le numéro de code-barres")

if __name__ == "__main__":
    # Create directories if needed
    os.makedirs("product_images", exist_ok=True)
    os.makedirs("runs/detect", exist_ok=True)
    
    # Run the app
    main()
import streamlit as st
import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim
from pyzbar.pyzbar import decode
import pytesseract
from PIL import Image
import re

# ---------------------------
# Configuration
# ---------------------------
# Set Tesseract path (change this according to your system)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows example
# For Linux/Mac: usually '/usr/bin/tesseract'

# ---------------------------
# Traitement d'image
# ---------------------------
def remove_background(image):
    """Suppression de l'arrière-plan avec l'algorithme GrabCut"""
    mask = np.zeros(image.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    height, width = image.shape[:2]
    rect = (10, 10, width - 20, height - 20)  # Marge
    cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
    
    # Noyau fixe utilisé ici
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    
    result = image * mask2[:, :, np.newaxis]
    return result

def calculate_histogram(image):
    """Calculer l'histogramme HSV normalisé"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    hist = cv2.calcHist([lab], [0, 1, 2], None, [30, 30, 30], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist

def calculate_color_similarity(image1, image2):
    """Métrique supplémentaire de similarité de couleur"""
    # Convertir en espace colorimétrique LAB
    lab1 = cv2.cvtColor(image1, cv2.COLOR_BGR2LAB)
    lab2 = cv2.cvtColor(image2, cv2.COLOR_BGR2LAB)
    
    # Calculer les valeurs de couleur moyennes
    mean1 = cv2.mean(lab1)
    mean2 = cv2.mean(lab2)
    
    # Calculer la distance euclidienne entre les couleurs moyennes
    color_distance = np.sqrt(sum((a - b) ** 2 for a, b in zip(mean1[:3], mean2[:3])))
    # Convertir la distance en score de similarité (0-1)
    max_possible_distance = 100 # Distance LAB maximale
    return 1 - (color_distance / max_possible_distance)

def calculate_ssim_score(image1, image2):
    """Calculer l'indice de similarité structurelle"""
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(gray1, gray2, full=True)
    return score

# ---------------------------
# Fonctions code-barres
# ---------------------------
def preprocess_image(image):
    """Convertir l'image en niveaux de gris et appliquer un seuillage"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def detect_barcode(image):
    """Détecter l'emplacement du code-barres avec pyzbar"""
    try:
        barcodes = decode(image)
        if barcodes:
            # Prendre le premier code-barres (peut être modifié pour gérer plusieurs codes)
            barcode = barcodes[0]
            return barcode.rect, barcode.data.decode('utf-8')
    except Exception as e:
        st.warning(f"Erreur de détection de code-barres : {e}")
    return None, None

def extract_text_near_barcode(image, barcode_rect, barcode_data):
    """Extraire le texte sous le code-barres avec OCR de secours"""
    x, y, w, h = barcode_rect
    
    # PREMIÈRE PRIORITÉ : Utiliser les données décodées du code-barres si valides
    if barcode_data and barcode_data.isdigit():
        return barcode_data
    
    # SECONDE PRIORITÉ : OCR comme solution de secours
    roi = image[y+h:y+h+100, x:x+w+100]  # 100 pixels sous le code-barres
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh_roi = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # OCR avec concentration stricte sur les chiffres
    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789'
    text = pytesseract.image_to_string(thresh_roi, config=custom_config)
    
    numbers = re.findall(r'\d+', text)
    return numbers[0] if numbers else None

def extract_all_text(image):
    """Extraire tout le texte de l'image pour vérification"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return pytesseract.image_to_string(thresh)

# ---------------------------
# Fonctions base de données
# ---------------------------
def load_database(folder, standard_size):
    """Charger les images produits depuis le dossier de la base de données"""
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
# Application Streamlit
# ---------------------------
def main():
    # Configuration de la page
    st.set_page_config(
        page_title="Système de Vérification de Produits",
        page_icon="🛍️",
        layout="wide"
    )
    
    # CSS personnalisé pour une meilleure apparence
    st.markdown("""
    <style>
        .main {
        background-color: #d9f9f3;  /* Light gray background */
        }       
        .match-oui { color: red; font-weight: bold; font-size: 18px; }
        .match-non { color: green; font-weight: bold; font-size: 18px; }
        .metric-box { border-radius: 5px; padding: 10px; background: #f0f2f6; }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("Les Yeux 👀 de la Justice : Reconnaître & Résister")
    st.markdown("""
    🕊️ Boycotter, c est aimer autrement..
    C est aimer la paix, aimer les innocents qu on ne voit pas mais qu on entend, dans le silence des bombes et l oubli des médias.  
    Un simple refus d achat, un choix différent dans un rayon, c est peut-être rien aux yeux du monde…
    Mais ensemble, ces riens deviennent des voix, des vagues, des actes qui dérangent, qui réveillent et qui transforment. 
                                     
    Pourriez-vous télécharger une image du produit afin de :

    -Vérifier si elle correspond à nos produits boycottés.

    -Valider le numéro de code-barres.
                
     Nous vous remercions d'avance pour votre aide.
    """)
    
    # Paramètres
    with st.sidebar:
        st.header("Paramètres")
        seuil_similarite = st.slider("Seuil de similarité", 0.0, 1.0, 0.9, 0.05)
        afficher_debug = st.checkbox("Afficher les informations de débogage")
    
    # Téléchargement de fichier
    uploaded_file = st.file_uploader("Télécharger une image de produit", type=["png", "jpg", "jpeg"])
    
    if uploaded_file:
        # Lire et afficher l'image téléchargée
        uploaded_bytes = uploaded_file.read()
        np_arr = np.frombuffer(uploaded_bytes, np.uint8)
        uploaded_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        # Créer des onglets pour différentes fonctionnalités
        tab1, tab2 = st.tabs(["🔍 Correspondance Produit", "📊 Validation Code-barres"])
        
        with tab1:
            st.header("Correspondance Produit")
            
            # Prétraitement
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
            
            # Charger la base de données et trouver les correspondances
            with st.spinner("Recherche dans la base de données..."):
                database = load_database("product_images", standard_size)
                
                meilleure_correspondance = None
                meilleur_score = -1
                details_correspondance = {}
                
                for nom, db_image in database:
                    # Calculer les trois métriques de similarité
                    hist_score = cv2.compareHist(
                        calculate_histogram(resized_image),
                        calculate_histogram(db_image),
                        cv2.HISTCMP_CORREL
                    )
                    color_score = calculate_color_similarity(resized_image, db_image)
                    ssim_score = calculate_ssim_score(resized_image, db_image)
                    
                    # Score combiné avec pondération équilibrée
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
            
            # Afficher les résultats
            if meilleur_score >= 0.8:
                st.markdown('<p class="match-oui">OUI - Malheureusement, ce produit fait partie des articles boycottés. Nous vous encourageons à explorer dautres options disponibles. Merci beaucoup pour votre compréhension et votre soutien dans cette démarche. !</p>', unsafe_allow_html=True)
                st.success(f"Confiance de la correspondance à un article boycotté : {meilleur_score:.2f}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Votre produit")
                    st.image(resized_image, channels="BGR", use_column_width=True)
                with col2:
                    st.subheader("Correspondance en base")
                    st.image(details_correspondance['image'], channels="BGR", 
                             caption=f"Produit: {details_correspondance['nom']}", 
                             use_column_width=True)
                
                # Métriques
                st.markdown("### Métriques de similarité")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Similarité histogramme", f"{details_correspondance['hist_score']:.2f}")
                with col2:
                    st.metric("Similarité SSIM", f"{details_correspondance['ssim_score']:.2f}")
                with col3:
                    st.metric("Similarité couleur", f"{details_correspondance['color_score']:.2f}")
                with col4:
                    st.metric("Score total", f"{details_correspondance['score_total']:.2f}")
                
                if afficher_debug:
                    with st.expander("Informations de débogage"):
                        st.write(details_correspondance)
            else:
                st.markdown('<p class="match-non">NON - Ce produit nest pas boycotté et est disponible à lachat. Vous pouvez lajouter à votre panier dès maintenant. </p>', unsafe_allow_html=True)
                st.warning(f"Le meilleur score était {meilleur_score:.2f} (en dessous du seuil de 0.8)")
        
        with tab2:
            st.header("Validation Code-barres")
            
            # Convertir au format OpenCV
            opencv_image = cv2.cvtColor(uploaded_image, cv2.COLOR_BGR2RGB)
            
            # Détecter le code-barres
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
                    
                    # Extraire le numéro du code-barres
                    with st.spinner("Extraction du numéro de code-barres..."):
                        numero_extrait = extract_text_near_barcode(opencv_image, barcode_rect, barcode_data)
                        
                        with col2:
                            st.subheader("Résultat de validation")
                            if numero_extrait:
                                st.write(f"Numéro de code-barres extrait : `{numero_extrait}`")
                                
                                if numero_extrait.startswith('619'):
                                    st.success("✅ Valide - Ce produit est fièrement fabriqué en Tunisie ! Vous pouvez l'acheter en toute confiance. Nous vous encourageons vivement à soutenir et à privilégier les produits locaux, pour contribuer au développement de notre économie. ")
                                elif  numero_extrait.startswith('729'):
                                    st.error("❌ Invalide - Malheureusement, ce produit fait partie des articles boycottés. Nous vous encourageons à explorer d'autres options disponibles. Merci beaucoup pour votre compréhension et votre soutien . ")

                                else:
                                    st.success("✅ Valide - Bonne nouvelle ! Ce produit n'est pas boycotté et est disponible à l'achat. Vous pouvez l'ajouter à votre panier dès maintenant. ")
                            else:
                                st.warning("⚠️ Impossible d'extraire le numéro de code-barres")
                    
                    if afficher_debug:
                        with st.expander("Informations de débogage code-barres"):
                            st.write(f"Données brutes du code-barres : {barcode_data}")
                            st.text_area("Texte extrait:", extract_all_text(opencv_image), height=200)

if __name__ == "__main__":
    # Créer le répertoire product_images s'il n'existe pas
    if not os.path.exists("product_images"):
        os.makedirs("product_images")
        st.warning("Répertoire 'product_images' créé. Veuillez y ajouter vos images produits.")
    
    main()
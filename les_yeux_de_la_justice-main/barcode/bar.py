import streamlit as st
import cv2
import numpy as np
from pyzbar.pyzbar import decode
import pytesseract
from PIL import Image
import re

# Set Tesseract path (change this according to your system)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows example
# For Linux/Mac: usually '/usr/bin/tesseract'

def preprocess_image(image):
    """Convert image to grayscale and apply thresholding"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def detect_barcode(image):
    """Detect barcode location using pyzbar"""
    try:
        barcodes = decode(image)
        if barcodes:
            # Get the first barcode (you can modify to handle multiple barcodes)
            barcode = barcodes[0]
            return barcode.rect, barcode.data.decode('utf-8')
    except Exception as e:
        st.warning(f"Barcode detection error: {e}")
    return None, None

def extract_text_near_barcode(image, barcode_rect, barcode_data):
    """Extract text below the barcode with OCR fallback"""
    x, y, w, h = barcode_rect
    
    # FIRST PRIORITY: Use the barcode's own decoded data if valid
    if barcode_data and barcode_data.isdigit():
        return barcode_data
    
    # SECOND PRIORITY: OCR as fallback
    roi = image[y+h:y+h+100, x:x+w+100]  # 100 pixels below barcode
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh_roi = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # OCR with strict digit focus
    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789'
    text = pytesseract.image_to_string(thresh_roi, config=custom_config)
    
    numbers = re.findall(r'\d+', text)
    return numbers[0] if numbers else None

def extract_all_text(image):
    """Extract all text from the image for verification"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return pytesseract.image_to_string(thresh)

def main():
    st.title("Barcode Number Validator")
    st.write("Upload an image of a product with a barcode to check if it starts with '619'")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Convert to OpenCV format
        opencv_image = np.array(image.convert('RGB'))
        
        # Detect barcode
        barcode_rect, barcode_data = detect_barcode(opencv_image)
        
        if not barcode_rect:
            st.error("No barcode detected. Please try with a clearer image.")
            return
            
        # Display processing tabs
        tab1, tab2, tab3 = st.tabs(["Barcode Detection", "Number Extraction", "All Text"])
        
        with tab1:
            st.subheader("Barcode Detection")
            x, y, w, h = barcode_rect
            barcode_display = opencv_image.copy()
            cv2.rectangle(barcode_display, (x, y), (x + w, y + h), (0, 255, 0), 2)
            st.image(barcode_display, caption='Detected Barcode', channels="BGR")
            st.success(f"Barcode scanner data: {barcode_data}")
        
        with tab2:
            st.subheader("Number Extraction")
            extracted_number = extract_text_near_barcode(opencv_image, barcode_rect, barcode_data)
            
            if extracted_number:
                st.write(f"Final extracted number: {extracted_number}")
                
                if extracted_number.startswith('619'):
                    st.success("✅ YES - The code starts with 619")
                else:
                    st.error("❌ NO - The code doesn't start with 619")
            else:
                st.warning("Could not extract numbers")
        
        with tab3:
            st.subheader("All Extracted Text")
            st.text_area("Debug Info:", extract_all_text(opencv_image), height=300)

if __name__ == "__main__":
    main()
Système de Vérification de Produits : les yeux de la justice
Cette application web, développée avec Streamlit, permet de vérifier si un produit donné appartient à une base de données de produits boycottés. Elle effectue également une vérification du code-barres afin d’identifier l’origine du produit.

Fonctionnalités
Correspondance d’image de produit
Compare l’image du produit importé à une base de données à l’aide de :

Indice de similarité structurelle (SSIM)

Comparaison d’histogrammes de couleurs

Distance moyenne de couleur en espace LAB

Reconnaissance et validation de code-barres
Détection et décodage automatique des codes-barres à l’aide de :

pyzbar pour la détection et le décodage

pytesseract comme solution de secours via OCR si le code-barres est incomplet

Vérification de l’origine via les préfixes GS1

Interface web Streamlit

Importation et analyse d’images depuis le navigateur

Seuil de similarité ajustable

Affichage visuel des résultats de comparaison et du statut du code-barres

Affichage optionnel des informations de débogage

Prérequis
Python 3.7 ou version supérieure

Streamlit

OpenCV

NumPy

Pillow

scikit-image

pytesseract

pyzbar




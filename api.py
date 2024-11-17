from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware 
import numpy as np
import joblib
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Initialisation de l'application FastAPI
app = FastAPI()
# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permet toutes les origines
    allow_credentials=True,
    allow_methods=["*"],  # Permet toutes les méthodes
    allow_headers=["*"],  # Permet tous les headers
)
# Charger le modèle SVM
svm_model = joblib.load('model/svm_model.pkl')

# Charger MobileNetV2 pour l'extraction des caractéristiques
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = GlobalAveragePooling2D()(base_model.output)
feature_extractor = Model(inputs=base_model.input, outputs=x)

@app.get('/')
def index():
    return {"message":"Bienvenue sur mon api"}
@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    try:
        # Lire et prétraiter l'image
        image_data = await image.read()
        with open("temp_image.jpg", "wb") as f:
            f.write(image_data)

        img = load_img("temp_image.jpg", target_size=(224, 224))
        #converite l'image en tableau Numpy representant les pixels de l'image (224(l), 224(L), 3(RGB))
        img_array = img_to_array(img) / 255.0  # Normalisation des pixel entre 0 et 1
        #ajout d'une dimension supplement au debut du tableau car le modele keras attend toujour
        #un lot d'image meme si nous avons une seule image 
        img_array = np.expand_dims(img_array, axis=0)

        # Extraire les caractéristiques
        features = feature_extractor.predict(img_array)

        # Prédire la classe avec SVM
        print(features)
        prediction = svm_model.predict(features)
        class_name = "Class 1" if prediction[0] == 1 else "Class 0"

        return {"prediction": class_name}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

from PIL import Image

import streamlit as st

import base64
import numpy as np
import io
from PIL import Image

import tensorflow as tf
from tensorflow import keras

from keras.preprocessing import image
from tqdm import tqdm

IMG_SHAPE = (224, 224, 3)

dog_names = ['Affenpinscher', ' Afghan_hound', ' Airedale_terrier', ' Akita', ' Alaskan_malamute', ' American_eskimo_dog', ' American_foxhound', ' American_staffordshire_terrier', ' American_water_spaniel', ' Anatolian_shepherd_dog', ' Australian_cattle_dog', ' Australian_shepherd', ' Australian_terrier', ' Basenji', ' Basset_hound', ' Beagle', ' Bearded_collie', ' Beauceron', ' Bedlington_terrier', ' Belgian_malinois', ' Belgian_sheepdog', ' Belgian_tervuren', ' Bernese_mountain_dog', ' Bichon_frise', ' Black_and_tan_coonhound', ' Black_russian_terrier', ' Bloodhound', ' Bluetick_coonhound', ' Border_collie', ' Border_terrier', ' Borzoi', ' Boston_terrier', ' Bouvier_des_flandres', ' Boxer', ' Boykin_spaniel', ' Briard', ' Brittany', ' Brussels_griffon', ' Bull_terrier', ' Bulldog', ' Bullmastiff', ' Cairn_terrier', ' Canaan_dog', ' Cane_corso', ' Cardigan_welsh_corgi', ' Cavalier_king_charles_spaniel', ' Chesapeake_bay_retriever', ' Chihuahua', ' Chinese_crested', ' Chinese_shar-pei', ' Chow_chow', ' Clumber_spaniel', ' Cocker_spaniel', ' Collie', ' Curly-coated_retriever', ' Dachshund', ' Dalmatian', ' Dandie_dinmont_terrier', ' Doberman_pinscher', ' Dogue_de_bordeaux', ' English_cocker_spaniel', ' English_setter', ' English_springer_spaniel', ' English_toy_spaniel', ' Entlebucher_mountain_dog', ' Field_spaniel', ' Finnish_spitz', ' Flat-coated_retriever', ' French_bulldog', ' German_pinscher', ' German_shepherd_dog', ' German_shorthaired_pointer', ' German_wirehaired_pointer', ' Giant_schnauzer', ' Glen_of_imaal_terrier', ' Golden_retriever', ' Gordon_setter', ' Great_dane', ' Great_pyrenees', ' Greater_swiss_mountain_dog', ' Greyhound', ' Havanese', ' Ibizan_hound', ' Icelandic_sheepdog', ' Irish_red_and_white_setter', ' Irish_setter', ' Irish_terrier', ' Irish_water_spaniel', ' Irish_wolfhound', ' Italian_greyhound', ' Japanese_chin', ' Keeshond', ' Kerry_blue_terrier', ' Komondor', ' Kuvasz', ' Labrador_retriever', ' Lakeland_terrier', ' Leonberger', ' Lhasa_apso', ' Lowchen', ' Maltese', ' Manchester_terrier', ' Mastiff', ' Miniature_schnauzer', ' Neapolitan_mastiff', ' Newfoundland', ' Norfolk_terrier', ' Norwegian_buhund', ' Norwegian_elkhound', ' Norwegian_lundehund', ' Norwich_terrier', ' Nova_scotia_duck_tolling_retriever', ' Old_english_sheepdog', ' Otterhound', ' Papillon', ' Parson_russell_terrier', ' Pekingese', ' Pembroke_welsh_corgi', ' Petit_basset_griffon_vendeen', ' Pharaoh_hound', ' Plott', ' Pointer', ' Pomeranian', ' Poodle', ' Portuguese_water_dog', ' Saint_bernard', ' Silky_terrier', ' Smooth_fox_terrier', ' Tibetan_mastiff', ' Welsh_springer_spaniel', ' Wirehaired_pointing_griffon', ' Xoloitzcuintli', ' Yorkshire_terrier']

# create our model
def get_model():
    model = keras.models.load_model('models/InceptionV3_final_model.hdf5')
    return model

# load model so it's in memory and not loaded each time there is a request
model = get_model()

def path_to_tensor(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

def decode_request(req):
    encoded = req["image"]
    decoded = base64.b64decode(encoded)
    return decoded

def preprocess(decoded):
    pil_image = Image.open(io.BytesIO(decoded)).resize((224,224), Image.LANCZOS).convert("RGB") 
    image = np.asarray(pil_image)
    batch = np.expand_dims(image, axis=0)
    return batch

def extract_InceptionV3(tensor):
    from keras.applications.inception_v3 import InceptionV3, preprocess_input
    return InceptionV3(weights='imagenet', include_top=False).predict(preprocess_input(tensor))

def InceptionV3_predict_breed(img):
    bottleneck_feature = extract_InceptionV3(preprocess(img))
    predicted_vector = model.predict(bottleneck_feature)
    return dog_names[np.argmax(predicted_vector)], np.max(predicted_vector) * 100

if __name__ == '__main__':

    st.title('Dog breed classifier')
    instructions = """
        Upload an image of a dog.
        The image will be fed through the Deep Neural Network 
        and the output will be displayed to the screen.
        """
    st.write(instructions)

    file = st.file_uploader('Upload An Image')

    if file:
        data = file.read()
        prediction = InceptionV3_predict_breed(data)
        top_prediction = prediction[0]
        top_prediction = ' '.join(top_prediction.split('_'))
        acc = prediction[1]
        img = Image.open(file)
        st.title("Here is the image you have selected")
        st.image(img)
        st.title("Predicted dog breed")
        st.text(top_prediction)
        st.title("Here is the accuracy")
        st.text("{:.2f}".format(acc) + '%')

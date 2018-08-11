from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
from scipy import spatial

vgg_model = VGG16(weights='imagenet', include_top=False)

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    features = vgg_model.predict(x)
    feature_vector = features.reshape(7*7*512)
    return feature_vector



import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


def pre_process_image(image_path: str) -> np.array:
    img = image.load_img(image_path, target_size=(128, 128))

    img_array = image.img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)

    img_array = img_array / 255

    return img_array


def get_model():
    model = load_model("model/duck_model.h5")

    return model


def predict(image_path: str) -> str:
    img_array = pre_process_image(image_path)
    model = get_model()
    prediction = model.predict(img_array)

    if prediction[0][0] >= 0.5:
        print("duck!")
    else:
        print("no duck :(")


def main(image_path: str) -> None:
    predict(image_path=image_path)
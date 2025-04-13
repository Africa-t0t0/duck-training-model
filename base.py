from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Parameters
IMG_SIZE = (128, 128)
BATCH_SIZE = 32


def _get_datasets_path_dict() -> dict:
    dd = {
        "train": "dataset/train",
        "validation": "dataset/validation"
    }
    return dd


def _get_data_generators(element: str) -> dict | ImageDataGenerator:
    # Pre-process
    # In this pre-process, we divide each pixel by 255 in order to get values between 0.0 to 1.0 to
    # normalize. This is necessary because it allows simple calculations and the model can learn faster.
    if element == "train":
        train_data_generator = ImageDataGenerator(rescale=1. / 255)
        return train_data_generator

    elif element == "validation":
        validation_data_generator = ImageDataGenerator(rescale=1. / 255)
        return validation_data_generator
    else:
        train_data_generator = ImageDataGenerator(rescale=1. / 255)
        validation_data_generator = ImageDataGenerator(rescale=1. / 255)

        dd = {
            "train": train_data_generator,
            "validation": validation_data_generator
        }

        return dd


def train_generator() -> ImageDataGenerator:
    datasets_paths_dd = _get_datasets_path_dict()
    train_path = datasets_paths_dd.get("train")

    train_image_generator = _get_data_generators(element="train")

    train_data = train_image_generator.flow_from_directory(
        directory=train_path,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary"
    )

    return train_data


def validation_generator() -> ImageDataGenerator:
    datasets_paths_dd = _get_datasets_path_dict()
    validation_path = datasets_paths_dd.get("validation")

    validation_image_generator = _get_data_generators(element="validation")

    train_data = validation_image_generator.flow_from_directory(
        directory=validation_path,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary"
    )

    return train_data


def get_model() -> Sequential:
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(128, 128, 3)),
        MaxPooling2D(2, 2),

        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),

        Flatten(),
        Dense(units=128, activation="relu"),
        Dropout(0.5),
        Dense(units=1, activation="sigmoid")
    ])

    return model


def compilation() -> None:
    model = get_model()

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    train_data = train_generator()
    validation_data = validation_generator()

    model.fit(
        train_data,
        epochs=10,
        validation_data=validation_data
    )

    model.save("model/duck_model.h5")


def main() -> None:
    compilation()

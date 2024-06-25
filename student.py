from tensorflow.keras import layers
import tensorflow as tf

def create_student_model():
    student = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(224, 224, 3)),
            layers.Conv2D(32, (3, 3), strides=(2, 2), padding="same"),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"),
            
            layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same"),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"),

            layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same"),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"),

            layers.GlobalAveragePooling2D(),
            layers.Dense(10),
        ],
        name="student",
    )
    return student
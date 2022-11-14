import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout, LeakyReLU
from tensorflow.keras.layers import Input, Dense, Flatten, GlobalAveragePooling2D, MaxPooling2D
# from keras.applications.resnet import ResNet50


def make_model():
    ModelDenseNet201 = tf.keras.models.Sequential([
        tf.keras.applications.DenseNet201(input_shape=(150, 150, 3),
                                          include_top=False,
                                          pooling='max',
                                          weights='imagenet'),
        # tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(256, activation='ReLU'),
        tf.keras.layers.Dense(128, activation='ReLU'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='ReLU'),
        # 512 neuron hidden layer
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    return ModelDenseNet201


# def make_model():
#     model = Sequential()
#     model.add(Conv2D(16, (3, 3), input_shape=(32, 32, 3), padding='same'))
#     model.add(LeakyReLU(0.1))
#     model.add(Conv2D(32, (3, 3), padding='same'))
#     model.add(LeakyReLU(0.1))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.25))
#     model.add(Conv2D(32, (3, 3), padding='same'))
#     model.add(LeakyReLU(0.1))
#     model.add(Conv2D(64, (3, 3), padding='same'))
#     model.add(LeakyReLU(0.1))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.25))
#     model.add(Flatten())
#     model.add(Dense(256))
#     model.add(LeakyReLU(0.1))
#     model.add(Dropout(0.5))
#     model.add(Dense(10, activation='softmax'))

#     return model

from keras.layers import SeparableConv2D, BatchNormalization, MaxPool2D, Dropout, Input, Flatten, Dense
import keras.backend as K
from keras.models import Model

class CancerNet:
    @staticmethod
    def create(height, width, depth, num_class):
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanel = 1
        else:
            inputShape = (height, width, depth)
            chanel = -1
        input_ = Input(inputShape)
        x = SeparableConv2D(32, (3, 3), padding="same", activation="relu")(input_)
        x = MaxPool2D((2, 2))(x)
        x = Dropout(0.25)(x)

        x = SeparableConv2D(64, (3, 3), padding="same", activation="relu")(x)
        x = BatchNormalization(chanel)(x)
        x = SeparableConv2D(64, (3, 3), padding="same", activation="relu")(x)
        x = BatchNormalization(chanel)(x)
        x = MaxPool2D((2, 2))(x)
        x = Dropout(0.25)(x)

        x = SeparableConv2D(128, (3, 3), padding="same", activation="relu")(x)
        x = BatchNormalization(chanel)(x)
        x = SeparableConv2D(128, (3, 3), padding="same", activation="relu")(x)
        x = BatchNormalization(chanel)(x)
        x = SeparableConv2D(128, (3, 3), padding="same", activation="relu")(x)
        x = BatchNormalization(chanel)(x)
        x = MaxPool2D((2, 2))(x)
        x = Dropout(0.25)(x)

        x = Flatten()(x)
        x = Dense(256, activation="relu")(x)
        x = BatchNormalization(chanel)(x)
        x = Dropout(0.5)(x)
        x = Dense(num_class, activation="sigmoid", bias_initializer="zeros")(x)

        return Model(inputs=input_, outputs=x)
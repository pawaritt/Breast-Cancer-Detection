from keras.layers import SeparableConv2D, BatchNormalization, MaxPool2D, Dropout, Input, Flatten, Dense
import keras.backend as K
from keras.callbacks import EarlyStopping
from keras.models import Model
from keras.optimizers.legacy import Adagrad
import keras_tuner as kt
from config import *
class CancerNet:
    @staticmethod
    def build(hp:kt.HyperParameters):
        chanel = -1
        input_ = Input(inputShape)
        x = SeparableConv2D(
            hp.Int("conv_1", min_value=32, max_value=96, step=32), (3, 3), padding="same", activation="relu")(input_)
        x = MaxPool2D((2, 2))(x)
        x = Dropout(0.25)(x)

        x = SeparableConv2D(
            hp.Int("conv_2", min_value=64, max_value=128, step=32), (3, 3), padding="same", activation="relu")(x)
        x = BatchNormalization(chanel)(x)
        x = SeparableConv2D(
            hp.Int("conv_3", min_value=64, max_value=128, step=32), (3, 3), padding="same", activation="relu")(x)
        x = BatchNormalization(chanel)(x)
        x = MaxPool2D((2, 2))(x)
        x = Dropout(0.25)(x)

        x = SeparableConv2D(
            hp.Int("conv_4", min_value=128, max_value=384, step=128), (3, 3), padding="same", activation="relu")(x)
        x = BatchNormalization(chanel)(x)
        x = SeparableConv2D(
            hp.Int("conv_5", min_value=128, max_value=384, step=128), (3, 3), padding="same", activation="relu")(x)
        x = BatchNormalization(chanel)(x)
        x = SeparableConv2D(
            hp.Int("conv_6", min_value=128, max_value=384, step=128), (3, 3), padding="same", activation="relu")(x)
        x = BatchNormalization(chanel)(x)
        x = MaxPool2D((2, 2))(x)
        x = Dropout(0.25)(x)

        x = Flatten()(x)
        x = Dense(
            hp.Int("dense_6", min_value=256, max_value=768, step=256), activation="relu")(x)
        x = BatchNormalization(chanel)(x)
        x = Dropout(0.5)(x)
        x = Dense(NUM_CLASSES, activation="sigmoid", bias_initializer="zeros")(x)
        lr = hp.Choice("learning_rate", values=[1e-1, 1e-2, 1e-3])
        opt = Adagrad(learning_rate=lr, decay=lr / NUM_EPOCHS)
        model = Model(inputs=input_, outputs=x)
        model.compile(
            optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"]
        )
        return model

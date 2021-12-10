import tensorflow as tf
tf.config.run_functions_eagerly(True)
import numpy as np

IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3
TARGET_FILE_NAME = 'notQuantizedUnet.h5'


def mean_iou(y_true, y_pred):
    th = 0.5
    y_pred_ = tf.cast(y_pred > th, tf.int32)
    metric = tf.keras.metrics.MeanIoU(num_classes=2)
    score = metric(y_true, y_pred_)
    return score
from keras.layers import UpSampling2D, Reshape, Add, Conv2DTranspose, Dropout, Conv2D,Input,MaxPooling2D,concatenate

# Build U-NET model
inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
#s = Lambda(lambda x: x / 255) (inputs)

conv_1_1 = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
conv_1_1 = Dropout(0.1) (conv_1_1)
conv_1_2 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv_1_1)
pool_1 = MaxPooling2D(2)(conv_1_2)

conv_2_1 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool_1)
conv_2_1 = Dropout(0.1) (conv_2_1)
conv_2_2 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv_2_1)
pool_2 = MaxPooling2D(2)(conv_2_2)

conv_3_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool_2)
conv_3_1 = Dropout(0.2) (conv_3_1)
conv_3_2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv_3_1)
pool_3 = MaxPooling2D(2)(conv_3_2)

conv_4_1 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool_3)
conv_4_1 = Dropout(0.2) (conv_4_1)
conv_4_2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv_4_1)
pool_4 = MaxPooling2D(2)(conv_4_2)

conv_5_1 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool_4)
conv_5_1 = Dropout(0.3) (conv_5_1)
conv_5_2 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv_5_1)

up_1 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (conv_5_2) 
conc_1 = concatenate([conv_4_2, up_1])
conv_up_1_1 = Conv2D(128, (3, 3), activation='relu', padding='same')(conc_1)
conv_up_1_1 = Dropout(0.2) (conv_up_1_1)
conv_up_1_2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv_up_1_1)

up_2 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (conv_up_1_2) 
conc_2 = concatenate([conv_3_2, up_2])
conv_up_2_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conc_2)
conv_up_2_1 = Dropout(0.2) (conv_up_2_1)
conv_up_2_2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv_up_2_1)

up_3 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (conv_up_2_2)
conc_3 = concatenate([conv_2_2, up_3])
conv_up_3_1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conc_3)
conv_up_3_1 = Dropout(0.1) (conv_up_3_1)
conv_up_3_2 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv_up_3_1)

up_4 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (conv_up_3_2)
conc_4 = concatenate([conv_1_2, up_4])
conv_up_4_1 = Conv2D(16, (3, 3), activation='relu', padding='same')(conc_4)
conv_up_4_1 = Dropout(0.1) (conv_up_4_1)
conv_up_4_2 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv_up_4_1)
outputs = Conv2D(1, (1, 1), activation='sigmoid') (conv_up_4_2)

unet_model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
unet_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])
unet_model.summary()
###############################
#model checkpoint
X = np.load('train_data_maybe/x_train.npy')
Y = np.load('train_data_maybe/y_train.npy')

X = X[:2000]
Y = Y[:2000]
#import sklearn.model_selection

checkpointer = tf.keras.callbacks.ModelCheckpoint('unet_first.h5', verbose=1, save_best_only=True)

class CustomCallback(tf.keras.callbacks.Callback):
    def on_train_end(self, logs=None):
        print('saving model')
        unet_model.save(TARGET_FILE_NAME)

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=2, monitor= 'val_loss'),
    tf.keras.callbacks.TensorBoard(log_dir='logs'),
    CustomCallback()
]

results = unet_model.fit(X,Y, validation_split=0.1, batch_size=1, epochs=3, callbacks=callbacks)
print(results)

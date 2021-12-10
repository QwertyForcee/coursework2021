import tensorflow as tf
tf.config.run_functions_eagerly(True)
import numpy as np

ISFIRST = 1
TARGET_FILE_NAME = 'quantizedUnet.h5'

IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3

def mean_iou(y_true, y_pred):
    th = 0.5
    y_pred_ = tf.cast(y_pred > th, tf.int32)
    metric = tf.keras.metrics.MeanIoU(num_classes=2)
    score = metric(y_true, y_pred_)
    return score

from kmeans import get_kmeans, prepare_data

def quantize(l,index):
    weights = l.get_weights()[0]
        
    originalShape = weights.shape
    reshapedWeights = weights.reshape(-1)
    for i in range(len(reshapedWeights)):
        if l.labels[i][0]==index:
            reshapedWeights[i] = l.codebook[index][0]
            l.indecators[i] = 1
    weights = l.get_weights()
    weights[0] = reshapedWeights.reshape(originalShape)
    l.set_weights(weights)
    l.codebook[index] = (l.codebook[index][0],1)
    return l
            
def reset_weights(l):
    if hasattr(l,'indecators'):
        weights = l.get_weights()[0]

        indecators = l.indecators
        originalShape = weights.shape
        reshapedWeights = weights.reshape(-1)
        for i in range(len(reshapedWeights)):
            if indecators[i] == 1:
                reshapedWeights[i] = l.codebook[l.labels[i][0]][0]
        weights = l.get_weights()
        weights[0] = reshapedWeights.reshape(originalShape)
        l.set_weights(weights)
    return l

def reset_weights_with_q(l):
    if hasattr(l,'indecators'):
        weights = l.get_weights()[0]

        indecators = l.indecators
        originalShape = weights.shape
        reshapedWeights = weights.reshape(-1)
        for i in range(len(reshapedWeights)):
            if indecators[i] == 1:
                reshapedWeights[i] = l.codebook[l.labels[i][0]][0]
        weights = l.get_weights()
        w = reshapedWeights.reshape(originalShape).astype('float16')
        b = weights[1].astype('float16')
        l.set_weights(np.array([w,b]).astype('float16'))
    return l

def partition(l):
    for i in range(len(l.codebook)):
        if l.codebook[i][1] == 0:
            return i
    return -1

def first_cluster(l):
    weights = l.get_weights()
    if len(weights)>0:
        print('---------------------------------------------------')
        originalShape = weights[0].shape
        reshapedWeights = weights[0].reshape(-1)
        isQuantizedIndecators = np.zeros(len(reshapedWeights))
        
        kmeans = get_kmeans(prepare_data(reshapedWeights), kmax=len(reshapedWeights))
        
        l.codebook = [(k,0) for k in kmeans.cluster_centers_]
        l.indecators = isQuantizedIndecators
        l.labels = [(label,0) for label in kmeans.labels_]
        l.left_clusters = len(l.codebook)
        print(kmeans.cluster_centers_)
        print('---------------------------------------------------')
    return l


def cluster(l):
    if hasattr(l,'left_clusters'):
        if l.left_clusters<0:
            return l
    weights = l.get_weights()
    if len(weights)>0:
        print('---------------------------------------------------')
        originalShape = weights[0].shape
        reshapedWeights = weights[0].reshape(-1)
        
        notQuantizedWeights = [reshapedWeights[i] for i in range(len(reshapedWeights)) if l.indecators[i]==0]
        kmeans = get_kmeans(prepare_data(notQuantizedWeights),nclusters=l.left_clusters)
        
        newIndex = 0
        for oldIndex in range(len(l.codebook)):
            if l.codebook[oldIndex][1]==0:
                l.codebook[oldIndex] = (kmeans.cluster_centers_[newIndex],0)
                for j in range(len(kmeans.labels_)):
                    if kmeans.labels_[j] == newIndex:
                        kmeans.labels_[j] = oldIndex 
                newIndex+=1

        j = 0
        for i in range(len(l.labels)):
            if l.labels[i][1] == 0:
                if j < len(kmeans.labels_):
                    l.labels[i] = (kmeans.labels_[j],0)
                    j+=1
        l.left_clusters -= 1
        
        print(kmeans.cluster_centers_)
        print('---------------------------------------------------')
    return l

def isModelQuantized(m):
    for l in m.layers:
        if hasattr(l,'codebook'):
            if 0 in [c[1] for c in l.codebook]:
                return False
        if hasattr(l,'ISFIRST') and len(l.get_weights())>0:
            if l.ISFIRST:
                return False
    return True

class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if isModelQuantized(model):
            print('saving model')
            for l in model.layers:
                if hasattr(l,'codebook'):
                    print('---')
                    print(l.codebook)
                    print('---')
            for i in range(len(model.layers)):
                if hasattr(model.layers[i],'indecators'):
                    model.layers[i] = reset_weights_with_q(model.layers[i])       
            model.save(TARGET_FILE_NAME)
            print(model)
            print('model saved into `{TARGET_FILE_NAME}`')
            exit()
        for i in range(len(model.layers)):
            if len(model.layers[i].get_weights())>0:
                if model.layers[i].ISFIRST:
                    model.layers[i] = first_cluster(model.layers[i])
                    model.layers[i] = quantize(model.layers[i],0)
                    model.layers[i].ISFIRST = 0
                else:
                    model.layers[i] = cluster(model.layers[i])
                    model.layers[i] = quantize(model.layers[i],partition(model.layers[i]))

    def on_train_batch_begin(self, batch, logs=None):
        for i in range(len(model.layers)):
            if hasattr(model.layers[i],'indecators'):
                model.layers[i] = reset_weights(model.layers[i])         




#build the model
inputs = tf.keras.layers.Input((IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))

s = tf.keras.layers.Lambda(lambda x: x/255)(inputs)
c1 = tf.keras.layers.Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
p1 = tf.keras.layers.MaxPooling2D((2,2))(c1)

c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
 
c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
c3 = tf.keras.layers.Dropout(0.2)(c3)
c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
 
c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
c4 = tf.keras.layers.Dropout(0.2)(c4)
c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)
 
c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
c5 = tf.keras.layers.Dropout(0.3)(c5)
c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

#Expansive path 
u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
u6 = tf.keras.layers.concatenate([u6, c4])
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
c6 = tf.keras.layers.Dropout(0.2)(c6)
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
 
u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
u7 = tf.keras.layers.concatenate([u7, c3])
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
c7 = tf.keras.layers.Dropout(0.2)(c7)
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
 
u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
u8 = tf.keras.layers.concatenate([u8, c2])
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
c8 = tf.keras.layers.Dropout(0.1)(c8)
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
 
u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
c9 = tf.keras.layers.Dropout(0.1)(c9)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
 
outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
 
model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])
model.summary()

###############################
#model checkpoint
X = np.load('train_data_maybe/x_train.npy')
Y = np.load('train_data_maybe/y_train.npy')

X = X[:50]
Y = Y[:50]
#import sklearn.model_selection

checkpointer = tf.keras.callbacks.ModelCheckpoint('unet_first.h5', verbose=1, save_best_only=True)

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=2, monitor= 'val_loss'),
    tf.keras.callbacks.TensorBoard(log_dir='logs'),
    CustomCallback()
]

for i in range(len(model.layers)):
    model.layers[i].ISFIRST = 1

results = model.fit(X,Y, validation_split=0.1, batch_size=1, epochs=6, callbacks=callbacks)
print(results)

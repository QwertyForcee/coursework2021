import tensorflow as tf
tf.config.run_functions_eagerly(True)

import numpy as np
import random
import cv2
from matplotlib import pyplot as plt

def mean_iou(y_true, y_pred):
    th = 0.5
    y_pred_ = tf.cast(y_pred > th, tf.int32)
    metric = tf.keras.metrics.MeanIoU(num_classes=2)
    score = metric(y_true, y_pred_)
    return score

MODEL_FILENAME='unet_crosswalks.h5'
TARGET_FILE_NAME = 'al_quantizedUnet.h5'

IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3

model = tf.keras.models.load_model(MODEL_FILENAME, custom_objects={'mean_iou':mean_iou})
X = np.load('train_data_maybe/x_train.npy')[:500]
Y = np.load('train_data_maybe/y_train.npy')[:500]

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])

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
        weights = l.get_weights()
        if len(weights)>0:
            weights = weights[0]
            indecators = l.indecators
            originalShape = weights.shape
            reshapedWeights = weights.reshape(-1)
            for i in range(len(reshapedWeights)):
                if indecators[i] == 1:
                    reshapedWeights[i] = l.codebook[l.labels[i][0]][0]
            weights = l.get_weights()
            w = weights[0].astype('float16')
            b = weights[1].astype('float16')
            l.set_weights([w,b])
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


callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=2, monitor= 'val_loss'),
    tf.keras.callbacks.TensorBoard(log_dir='logs'),
    CustomCallback()
]

for i in range(len(model.layers)):
    model.layers[i].ISFIRST = 1

results = model.fit(X,Y, validation_split=0.1, batch_size=1, epochs=3, callbacks=callbacks)
print(results)



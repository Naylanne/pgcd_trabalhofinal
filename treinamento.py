# basic
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# image load
import random
import os
from PIL import Image

## tensorflow ##

# prepare dataset
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.data.experimental import cardinality

# data augmentation
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom, RandomTranslation, RandomContrast, RandomBrightness, GaussianNoise
from tensorflow.data import AUTOTUNE

# model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, BatchNormalization, Dropout, Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# scoring
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

# visualization
import plotly.graph_objects as go


from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

### CONSTANTS ###

DIR_PATH = r'./dataset'
BATCH_SIZE = 16
IMAGE_SIZE = 224
VALIDATION_SPLIT = 0.2
SEED = 12
AUTOTUNE = AUTOTUNE
EPOCHS = 5

## PARAMETERS OF MODEL ##

PATIENCE_EARLY_STOPPING = 8
VERBOSE_EARLY_STOPPING = 1
PATIENCE_REDUCE_LR = 4
FACTOR_REDUCE_LR = 0.5
VERBOSE_REDUCE_LR = 1
MIN_LR = 1e-7
NUM_OF_LAST_LAYERS_TRAINING = 100
DROPOUT = 0.7
LEARNING_RATE = 1e-4
L2_REGULARIZATION = 0.01
LABEL_SMOOTHING = 0.15
METRIC = 'accuracy'

# FUNCTIONS

def random_10_image_show(dir_path: 'str'):
    
    random_fake_real_faces = random.sample(os.listdir(os.path.join(dir_path, 'Fake faces')), 5) +\
                             random.sample(os.listdir(os.path.join(dir_path, 'Real faces')), 5)
    
    fig = plt.figure(figsize = (20, 8))
    for index, file in enumerate(random_fake_real_faces, 1):
        if index <= 5:
            path_dir = os.path.join(dir_path, 'Fake faces')
        else: 
            path_dir = os.path.join(dir_path, 'Real faces')
        plt.subplot(2, 5, index)
        a = Image.open(os.path.join(path_dir, file))
        plt.imshow(a)
        plt.title(f'{file}')
    fig.tight_layout()
    plt.show()

def prepare_tensorflow_dataset(
    dir_path: 'str',
    image_size: int,
    batch_size: int,
    validation_split: float,
    seed: int    
):
    
    train_set = image_dataset_from_directory(
        directory = dir_path,
        image_size = (image_size, image_size),
        batch_size = batch_size,
        validation_split = validation_split,
        seed = seed,
        subset = 'training', 
        label_mode = 'binary')

    temp_val_set = image_dataset_from_directory(
        directory = dir_path,
        image_size = (image_size, image_size),
        batch_size = batch_size,
        validation_split = validation_split,
        seed = seed,
        subset = 'validation',        
        label_mode = 'binary')

    val_batches = cardinality(temp_val_set)
    val_set = temp_val_set.take(val_batches // 2)
    test_set = temp_val_set.skip(val_batches // 2)
     
    print('*' * 50)
    print(f'Number of batches:')
    print(f'\tTrain: {cardinality(train_set)}')
    print(f'\tValidation: {cardinality(val_set)}')
    print(f'\tTest: {cardinality(test_set)}')
    print('*' * 50)
    print(f'Classes: {train_set.class_names}')

    return train_set, val_set, test_set

def data_augmentation(
    dataset, 
    shuffle: bool,
    augmentation: bool,
    buffer_size = 200,
    autotune = AUTOTUNE
):
    augmenter = Sequential(
        [
            RandomFlip("horizontal"),
            RandomRotation(0.1),      
            RandomZoom(0.1),          
            RandomTranslation(0.1, 0.1), 
            RandomContrast(0.2),      
            RandomBrightness(0.2),
            GaussianNoise(0.05)
        ]
    )

    # dataset = dataset.cache()
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size = buffer_size)

    if augmentation:
        dataset = dataset.map(
            lambda x, y: (
                augmenter(x, training = True),
                y), 
            num_parallel_calls = autotune)

    return dataset.prefetch(buffer_size = autotune)

def EfficientNetB0_model_training(
    train_set,
    validation_set,
    epochs: int,
    batch_size: int,
    patience_early_stopping: int,
    verbose_early_stopping: int,
    patience_reduce_lr: int,
    factor_reduce_lr: int,
    verbose_reduce_lr: int,
    min_lr: float,
    image_size: int,
    num_of_last_layers_training: int,
    dropout: float,
    learning_rate: float,
    l2_regularization: float,
    label_smoothing: float,
    metric: 'str'    
):
    early_stop = EarlyStopping(
        patience = patience_early_stopping, 
        restore_best_weights = True,
        verbose = verbose_early_stopping,
        monitor = 'val_loss')

    red_lr = ReduceLROnPlateau(
        patience = patience_reduce_lr, 
        factor = factor_reduce_lr, 
        verbose = verbose_reduce_lr,
        min_lr = min_lr,
        monitor = 'val_loss')
    
    base_model = EfficientNetB0(
        weights = 'imagenet',
        include_top = False,
        input_shape = (image_size, image_size, 3))
    
    base_model.trainable = True
    for layer in base_model.layers[:-num_of_last_layers_training]: 
        layer.trainable = False
    
    model = Sequential([
        Input(shape = (image_size, image_size, 3)),
        base_model,
        GlobalAveragePooling2D(),
        BatchNormalization(),
        Dropout(dropout),
        Dense(1, activation = 'sigmoid',
              kernel_regularizer = l2(l2_regularization))
        ]
    )
    
    model.compile(optimizer = Adam(learning_rate = learning_rate), 
                  loss = BinaryCrossentropy(label_smoothing = label_smoothing),
                  # loss = 'binary_crossentropy',
                  metrics = [metric])

    results = model.fit(x = train_set, 
                  validation_data = validation_set, 
                  epochs = epochs, 
                  batch_size = batch_size, 
                  callbacks = [early_stop, red_lr])

    return model, results

def plot_metric_loss(source, metric):

    fig = go.Figure()
    fig.add_trace(go.Scatter(y = source.history[metric], 
                             mode = 'lines+markers', 
                             name = f'train {metric}',
                             line = {'color': 'firebrick', 'width': 4}))
    
    fig.add_trace(go.Scatter(y = source.history[f'val_{metric}'], 
                             mode = 'lines+markers', 
                             name = f'val {metric}',
                             line = {'color': 'firebrick', 'width': 4, 'dash': 'dot'}))
    
    fig.add_trace(go.Scatter(y = source.history['loss'], 
                             mode = 'lines+markers', 
                             name = 'train loss',
                             line = {'color': 'royalblue', 'width': 4}))
                  
    fig.add_trace(go.Scatter(y = source.history['val_loss'], 
                             mode = 'lines+markers', 
                             name = 'val loss',
                             line = {'color': 'royalblue', 'width': 4, 'dash': 'dot'}))
    
    fig.show()

def evaluate_model(dataset, model):
    
    imgs, y_true, y_true_label, y_preds, y_preds_binary, y_preds_label = [], [], [], [], [], []
    for images, labels in tqdm(dataset):
        preds = model.predict(images, verbose = 0)
        preds_binary = (preds > 0.5).astype(int).flatten()
        imgs.extend(images)
        y_true.extend(labels.numpy().astype(int))
        y_preds.extend(preds)
        y_preds_binary.extend(preds_binary)
        y_true_label = ['Fake' if i == 0 else 'Real' for i in y_true]
        y_preds_label = ['Fake' if i == 0 else 'Real' for i in y_preds_binary]

    print(classification_report(y_true, y_preds_binary, target_names = ['Fake', 'Real']))
    print('*' * 50)
    
    ConfusionMatrixDisplay.from_predictions(y_true, y_preds_binary, display_labels = ['Fake', 'Real'])
    
    return imgs, y_true, y_true_label, y_preds, y_preds_binary, y_preds_label

def random_10_image_predictions(
    imgs: list, 
    y_true: list, 
    y_true_label: list, 
    y_preds: list, 
    y_preds_binary: list,
    y_preds_label: list
):
    
    misclassified_images, classified_images = [], []
    for i in range(len(y_true)):
        if y_preds_binary[i] != y_true[i]:
            misclassified_images.append({
                'image': imgs[i].numpy() / 255,
                'true': y_true[i],
                'true_label': y_true_label[i],
                'pred': y_preds[i],
                'pred_binary': y_preds_binary[i],
                'preds_label': y_preds_label[i]
            })
        else:
            classified_images.append({
                'image': imgs[i].numpy() / 255,
                'true': y_true[i],
                'true_label': y_true_label[i],
                'pred': y_preds[i],
                'pred_binary': y_preds_binary[i],
                'preds_label': y_preds_label[i]
            })

    true_images = [d for d in classified_images if d.get('true') == 1]
    false_images = [d for d in classified_images if d.get('true') == 0]
    random_10_images = random.sample(true_images, 5) + random.sample(false_images, 5)
    
    fig = plt.figure(figsize = (20, 10))
    for index in range(len(random_10_images)):
        plt.subplot(2, 5, index + 1)
        plt.imshow(random_10_images[index]['image'])
        plt.title(f"\nTrue label: {random_10_images[index]['true_label']}\nPredicted label: {random_10_images[index]['preds_label']}\nProb. of Real label: {np.round(float(random_10_images[index]['pred']), 5)}")
    fig.suptitle('10 random correct predictions', fontsize = 20, y = 1.05) 
    fig.tight_layout()
    plt.show() 
    print(f'\n\n')
    true_images = [d for d in misclassified_images if d.get('true') == 1]
    false_images = [d for d in misclassified_images if d.get('true') == 0]
    random_10_images = random.sample(true_images, 5) + random.sample(false_images, 5)
    
    fig = plt.figure(figsize = (20, 10))
    for index in range(len(random_10_images)):
        plt.subplot(2, 5, index + 1)
        plt.imshow(random_10_images[index]['image'])
        plt.title(f"True label: {random_10_images[index]['true_label']}\nPredicted label: {random_10_images[index]['preds_label']}\nProb. of Real label: {np.round(float(random_10_images[index]['pred']), 5)}")
    fig.suptitle('10 random incorrect predictions', fontsize = 20, y = 1.05) 
    fig.tight_layout()
    plt.show() 

# MAIN

if __name__ == "__main__":

    print("Preparando dataset...")

    train_set, val_set, test_set = prepare_tensorflow_dataset(
        dir_path=DIR_PATH,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        seed=SEED
    )

    print("Aplicando augmentation...")

    train_set = data_augmentation(
        train_set,
        shuffle=True,
        augmentation=True
    )

    val_set = data_augmentation(
        val_set,
        shuffle=False,
        augmentation=False
    )

    print("Treinando modelo...")

    model, history = EfficientNetB0_model_training(
        train_set=train_set,
        validation_set=val_set,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        patience_early_stopping=PATIENCE_EARLY_STOPPING,
        verbose_early_stopping=VERBOSE_EARLY_STOPPING,
        patience_reduce_lr=PATIENCE_REDUCE_LR,
        factor_reduce_lr=FACTOR_REDUCE_LR,
        verbose_reduce_lr=VERBOSE_REDUCE_LR,
        min_lr=MIN_LR,
        image_size=IMAGE_SIZE,
        num_of_last_layers_training=NUM_OF_LAST_LAYERS_TRAINING,
        dropout=DROPOUT,
        learning_rate=LEARNING_RATE,
        l2_regularization=L2_REGULARIZATION,
        label_smoothing=LABEL_SMOOTHING,
        metric=METRIC
    )

    print("Salvando modelo...")

    model.save("model/face_detector_model.keras")

    print("Modelo salvo com sucesso!")
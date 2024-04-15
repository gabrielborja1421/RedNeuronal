import os
import numpy as np
from PIL import Image
from sklearn.model_selection import KFold
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import ReduceLROnPlateau
import random
import matplotlib.pyplot as plt

class Model:
    def __init__(self, training_folder, trainingCategory, widthImg, heightImg, epochs, k):
        self.training_folder = training_folder
        self.trainingCategory = trainingCategory
        self.widthImg = widthImg
        self.heightImg = heightImg
        self.epochs = epochs
        self.k = k
        self.datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

    def chargeData(self):
        imagenes = []
        etiquetas = []
        for etiqueta in os.listdir(self.training_folder):
            carpeta = os.path.join(self.training_folder, etiqueta)
            for imagen_nombre in os.listdir(carpeta):
                imagen_ruta = os.path.join(carpeta, imagen_nombre)
                img = Image.open(imagen_ruta).convert('L').resize((self.widthImg, self.heightImg))
                img_array = np.asarray(img)
                imagenes.append(img_array)
                etiquetas.append(etiqueta)
        self.trainingImages = np.asarray(imagenes) / 255.0
        self.labels_entrenamiento = np.asarray(etiquetas)
        label_encoder = LabelEncoder()
        self.labels_entrenamiento = label_encoder.fit_transform(self.labels_entrenamiento)
        
        # Asegurarse de que los datos tienen la cuarta dimensión para canales.
        self.trainingImages = np.expand_dims(self.trainingImages, axis=-1)

    def show_images_random(self, images, labels, num_images_to_show=5):
        fig, axes = plt.subplots(5, 5, figsize=(10, 10))
        random_indices = random.sample(range(len(images)), min(len(images), 25))

        for i in range(5):
            for j in range(5):
                index = random_indices[i * 5 + j]
                axes[i, j].imshow(images[index].squeeze(), cmap='gray')
                axes[i, j].set_title(f'Label: {labels[index]}')
                axes[i, j].axis('off')

        plt.tight_layout()
        plt.show()

    def createModel(self):
        self.modelo = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(self.widthImg, self.heightImg, 1)),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(len(self.trainingCategory), activation='softmax')
        ])

        self.modelo.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    def trainModel(self):
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
        kfold = KFold(n_splits=self.k, shuffle=True)
        scores = []

        for train, val in kfold.split(self.trainingImages, self.labels_entrenamiento):
            train_generator = self.datagen.flow(self.trainingImages[train], self.labels_entrenamiento[train], batch_size=32)
            val_generator = self.datagen.flow(self.trainingImages[val], self.labels_entrenamiento[val], batch_size=32)
            self.trainingRecord = self.modelo.fit(
                train_generator,
                epochs=self.epochs,
                validation_data=val_generator,
                callbacks=[reduce_lr]
            )
            scores.append(self.trainingRecord.history['val_accuracy'][-1])

        self.mean_score = np.mean(scores)
        self.standard_deviation = np.std(scores)

    def saveModel(self, directorio_modelo):
        if not os.path.exists(directorio_modelo):
            os.makedirs(directorio_modelo)
        self.modelo.save(os.path.join(directorio_modelo, 'modelo.h5'))
        self.modelo.save_weights(os.path.join(directorio_modelo, 'pesos.h5'))

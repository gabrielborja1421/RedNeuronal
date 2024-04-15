import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from keras.models import load_model
import tensorflow as tf

# Asegúrate de que la clase Model esté disponible desde el archivo 'Modelo.py'
from Modelo import Model

class Prediction:
    def __init__(self):
        self.model_path = './modelo/modelo.h5'
        self.widthImg = 200
        self.heightImg = 200

    def normalize_image(self, img):
        return np.asarray(img) / 255.0

    def load_images_from_directory(self, directory, labels_dict):
        images = []
        labels = []
        for label, subdir in enumerate(os.listdir(directory)):
            subdir_path = os.path.join(directory, subdir)
            for filename in os.listdir(subdir_path):
                file_path = os.path.join(subdir_path, filename)
                image = Image.open(file_path).convert('L').resize((self.widthImg, self.heightImg))
                images.append(self.normalize_image(image))
                labels.append(labels_dict[subdir])
        return np.array(images), np.array(labels)

    def predictDate(self):
        labels_dict = {name: idx for idx, name in enumerate(sorted(os.listdir('data/validacion/')))}
        validate, labels = self.load_images_from_directory('data/validacion/', labels_dict)
        validate = validate.reshape(-1, self.widthImg, self.heightImg, 1)
        record = load_model(self.model_path)
        predictions = record.predict(validate)
        predicted_classes = np.argmax(predictions, axis=1)
        plt.figure()
        for i in range(len(validate)):
            plt.imshow(validate[i].squeeze(), cmap='gray')
            plt.colorbar()
            plt.title('Predicción: ' + list(labels_dict.keys())[predicted_classes[i]])
            plt.show()
        return record, labels, predicted_classes

    def confusionMatrix(self, record, true_labels, predicted_classes, labels_dict):
        # Generar la matriz de confusión
        confusion_mtx = tf.math.confusion_matrix(true_labels, predicted_classes)
        # Convertir las etiquetas numéricas a nombres usando el diccionario de etiquetas
        labels_names = [labels_dict[idx] for idx in sorted(labels_dict.keys(), key=lambda x: labels_dict[x])]
        # Crear la figura para visualizar la matriz
        plt.figure(figsize=(10, 8))
        # Dibujar la matriz de confusión con nombres de etiquetas
        sns.heatmap(confusion_mtx, annot=True, fmt='g', xticklabels=labels_names, yticklabels=labels_names)
        plt.xlabel('Predicción')
        plt.ylabel('Etiqueta Verdadera')
        plt.title('Matriz de Confusión')
        plt.show()


    def execModel(self):
        training_folder = './data/Entrenamiento/'
        trainingCategory = os.listdir(training_folder)
        widthImg = 200
        heightImg = 200
        epochs = 100
        k = 2
        model = Model(training_folder, trainingCategory, widthImg, heightImg, epochs, k)
        model.chargeData()  # Se asegura de expandir las imágenes a 4 dimensiones
        model.createModel()
        model.trainModel()
        print('Precisión media:', model.mean_score)
        print('Desviación estándar:', model.standard_deviation)
        plt.xlabel('# Iteraciones')
        plt.ylabel('magnitud de perdida')
        plt.plot(model.trainingRecord.history['loss'], label='Error', color='blue')
        plt.legend()
        plt.show()
        directory_model = './modelo/'
        model.saveModel(directory_model)

if __name__ == "__main__":
    prediction = Prediction()
    #prediction.execModel()
    model, true_labels, predicted_classes = prediction.predictDate()
    labels_dict = {name: idx for idx, name in enumerate(sorted(os.listdir('data/validacion/')))}
    prediction.confusionMatrix(model, true_labels, predicted_classes, labels_dict)

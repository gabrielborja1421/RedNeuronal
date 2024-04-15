import tensorflow as tf

# Cargar el modelo Keras
model = tf.keras.models.load_model('modelo\modelo.h5')

# Convertir el modelo a TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# (Opcional) Configurar el convertidor para soportar operaciones no incluidas por defecto
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # Operaciones básicas de TensorFlow Lite
    tf.lite.OpsSet.SELECT_TF_OPS  # Permite operaciones de TensorFlow no cubiertas por las básicas
]

# Realizar la conversión
tflite_model = converter.convert()

# Guardar el modelo convertido a un archivo
with open('modelo_convertido.tflite', 'wb') as f:
    f.write(tflite_model)

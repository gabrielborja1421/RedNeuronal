import os

# Asegúrate de que esta ruta coincida con el directorio donde se guardó el modelo y los pesos.
directorio_modelo = './modelo/'
modelo_file = os.path.join(directorio_modelo, 'modelo.h5')
pesos_file = os.path.join(directorio_modelo, 'pesos.h5')

# Obtener el tamaño de los archivos.
tamaño_modelo = os.path.getsize(modelo_file)
tamaño_pesos = os.path.getsize(pesos_file)

print(f"Tamaño del archivo de modelo: {tamaño_modelo} bytes")
print(f"Tamaño del archivo de pesos: {tamaño_pesos} bytes")

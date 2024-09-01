import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
# Carrega o conjunto de dados MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocessa os dados
x_train = x_train.reshape(-1, 784).astype('float32') / 255
x_test = x_test.reshape(-1, 784).astype('float32') / 255

# Define a camada de entrada
inputs = Input(shape=(784,))    # Array unidimensional

# Camadas ocultas
x = Dense(128, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)
x = Dense(32, activation='relu')(x)

# Define a camada de saída
outputs = Dense(10,activation='softmax')(x)

# Cria o modelo
model = Model(inputs = inputs, outputs=outputs)

# Cria um modelo sequencial
# model = Sequential()
# model.add(Dense(128, activation='relu', input_shape=(784,)))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(10, activation='softmax'))

# Compila o modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Treina o modelo
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Avalia o modelo
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test loss: {loss}")
print(f"Test accuracy: {accuracy}")

# Faz previsões
predictions = model.predict(x_test)
print(predictions[0])

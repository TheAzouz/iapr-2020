import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def train_model(x_train,y_train,x_test,y_test):
    
    IMG_HEIGHT = 28
    IMG_WIDTH = 28
    
    model = Sequential()

    model.add(layers.Conv2D(filters=6, kernel_size=(5, 5), activation='tanh', input_shape=(28,28,1)))
    model.add(layers.AveragePooling2D())
    model.add(layers.Conv2D(filters=16, kernel_size=(5, 5), activation='tanh'))
    model.add(layers.AveragePooling2D())
    model.add(layers.Conv2D(filters=120, kernel_size=(3, 3), activation='tanh'))
    model.add(layers.Flatten())
    model.add(layers.Dense(units=84, activation='tanh'))
    model.add(layers.Dense(units=10, activation = 'softmax'))
    
    model.compile(optimizer='sgd',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    datagen = ImageDataGenerator(rotation_range=45, vertical_flip=True, shear_range = 0.2, zoom_range = 0.2)
    training_data = datagen.flow(x_train,y_train,batch_size = 100)
    testing_data = datagen.flow(x_test,y_test,batch_size = 100)
    
    epochs = 10
    history = model.fit(training_data, epochs=epochs, validation_data = testing_data)
    
    model.save('LeNet_Model')
    
    return model
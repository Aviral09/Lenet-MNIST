from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten
from keras.utils.np_utils import to_categorical
from tensorflow.keras import optimizers

(input_train, output_train), (input_test, output_test) = cifar10.load_data()
model = Sequential()

# First set of CONV => ACTIVATION => POOL layers
# Convolutional layers - 20
model.add(Conv2D(20, kernel_size=(3, 3), padding='VALID', activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPool2D(pool_size=(2, 2), padding='VALID'))

# Second set of CONV => ACTIVATION => POOL layers
# Convolutional layers - 50
model.add(Conv2D(50, kernel_size=(3,3), padding='VALID', activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2), padding='VALID'))

# Flattening the layer for transitioning to the Fully Connected Layers
model.add(Flatten())
# Fully Connected Layers with 120 outputs
model.add(Dense(120, activation='relu'))
# Fully Connected Layers with 84 outputs
model.add(Dense(84, activation='relu'))
# Fully Connected Layers with 10 outputs
model.add(Dense(10, activation='softmax'))

# Compiling
model.compile(optimizer=optimizers.SGD(lr=0.1, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

# Training
h = model.fit(input_train/255, to_categorical(output_train), epochs=20, batch_size=64)

# Testing
score = model.evaluate(input_test/255, to_categorical(output_test), batch_size=64)

print("Test loss: %f" % score[0])
print("Test accuracy: %f" % score[1])

# Test accuracy : 65.33%
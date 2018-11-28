import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import pickle
from tensorflow.keras.callbacks import TensorBoard
import sys
from keras.models import load_model

NAME = "MY_CNN"
tensorboard = TensorBoard(log_dir="logs\{}".format(NAME))

if len(sys.argv) > 1:
    epochs = int(sys.argv[2])
else:
    epochs = 1

pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)
pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)

pickle_in = open("Z.pickle","rb")
Z = pickle.load(pickle_in)
pickle_in = open("p.pickle","rb")
p = pickle.load(pickle_in)



model = Sequential()

model.add(Conv2D(128, (3, 3), input_shape=(250, 250, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(64))
model.add(Activation('relu'))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'],)
model.fit(X, y, batch_size=10, epochs=epochs, validation_split=0.2, callbacks=[tensorboard])

scores = model.evaluate(Z, p, batch_size = 2)
print('Accuracy:', scores[1]*100, '%')
print('Loss:', scores[0])


#save data
model_json = model.to_json()
json_file = open("CNN.json", "w")

json_file.write(model_json)
json_file.close()

model.save_weights("MY_CNNchka.h5")
print("Saved")
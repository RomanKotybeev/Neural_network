import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pickle
import random
import sys

DATADIR = "E:\\DATASET\\101_ObjectCategories\\train"
DATATEST  = "E:\\DATASET\\101_ObjectCategories\\test"
CATEGORIES = ["airplanes", "BACKGROUND_Google"]

DATADIR = sys.argv[2]
DATATEST = sys.argv[3]
img_size = 250
batch_size = 2


X_train = []

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR,category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_COLOR)
                new_array = cv2.resize(img_array, (img_size, img_size))
                X_train.append([new_array, class_num])
            except:
                pass

create_training_data()
print(len(X_train))

random.shuffle(X_train)
X = []
y = []
for features,label in X_train:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, img_size, img_size, 3)
print(X.shape)


X_test = []
def create_testing_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR,category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_COLOR)
                new_array = cv2.resize(img_array, (img_size, img_size))
                X_test.append([new_array, class_num])
            except:
                pass
create_testing_data()

random.shuffle(X_test)


print(len(X_test))
Z = []
p = []
for features,label in X_test:
    Z.append(features)
    p.append(label)

Z = np.array(Z).reshape(-1, img_size, img_size, 3)

pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()
pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()

pickle_out = open("Z.pickle","wb")
pickle.dump(Z, pickle_out)
pickle_out.close()
pickle_out = open("p.pickle","wb")
pickle.dump(p, pickle_out)
pickle_out.close()
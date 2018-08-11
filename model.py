from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.callbacks import LambdaCallback
from keras.layers import Dense
from keras.optimizers import RMSprop
from keras.models import Sequential
import numpy as np
import pandas

vgg_model = VGG16(weights='imagenet', include_top=False)

# utils

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    features = vgg_model.predict(x)
    feature_vector = features.reshape(7*7*512)
    return feature_vector


# prepare data
def load_dataset(path):
    dataset = pandas.read_csv(path)
    dataset = [list(r) for r in list(dataset.values)]
    print('loaded dataset')
    X = np.array([np.array(row[:-1]) for row in dataset])
    print('loaded X')
    _Y = [row[-1] for row in dataset]
    categories = list(set(_Y))
    Y = np.array([np.array([1 if cat == y else 0 for cat in categories]) for y in _Y])
    print('loaded Y')
    return X, Y, categories

validation_size = 0.2
rawX, rawY, categories = load_dataset('data/training_data.csv')
validation_count = int(len(rawY) * validation_size)
Xv = rawX[:validation_count]
Yv = rawY[:validation_count]
X = rawX[validation_count:]
Y = rawY[validation_count:]

print("Shape X: %s" % (str(X.shape),))
print("Shape Y: %s" % (str(Y.shape),))

# model params
print('building model')
n_input = len(X[0]) # num input features
n_hidden1 = 3000
n_hidden2 = 300
n_classes = len(Y[0])

lr = 0.001
batch_size = 20
epochs = 100

# build model

model = Sequential()
model.add(Dense(n_hidden1, input_shape=(n_input,)))
model.add(Dense(n_hidden2, input_shape=(n_input,)))
model.add(Dense(n_classes, activation='sigmoid'))
optimizer = RMSprop(lr=lr)
model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])


best_error = 999999

def save_model():
    # serialize model to JSON
    print("Saving Model...")
    model_json = model.to_json()
    with open("model/model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model/model.h5")
    print("Saved model to disk")

def on_epoch_end(epoch, logs):
    global best_error
    # Function invoked at end of each epoch.
    print()
    print('----- Epoch: %s' % (int(epoch)+1,))
    print('Running Validation...')
    pred = model.predict(Xv)
    diff = np.absolute(pred - Yv)
    test_error = diff.sum() / len(Yv)
    if test_error < best_error:
        print("New Best")
        best_error = test_error
        save_model()
    print(pred.shape)
    print(Yv.shape)
    #print("Train Error: %s" % str(train_error))
    print("Test Error: %s" % str(test_error))
    print("Best: %s" % str(best_error))

# train model
print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
on_epoch_end(-1, None)
model.fit(X, Y, batch_size=batch_size,
          epochs=epochs,
          callbacks=[print_callback])

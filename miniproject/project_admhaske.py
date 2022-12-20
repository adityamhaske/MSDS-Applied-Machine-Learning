import tensorflow as tf
import pandas as pd
import numpy as np
import sys


def run(train_image, train_label, test_image):
    model = tf.keras.models.load_model('model_admhaske.h5')
    test_image = np.array([np.array(val) for val in test_image])
    prediction_result = model.predict(test_image)
    result = [np.argmax(prediction) for prediction in prediction_result]
    with open('project_admhaske.txt', 'w') as file:
        file.write('\n'.join(map(str, result)))
        file.flush()
        return True
    return False

'''

#model code

model = keras.Sequential([
    keras.layers.Rescaling(1./255, input_shape=(28, 28, 1)),
    keras.layers.Conv2D(32,(3,3),input_shape=(28,28,1),activation='relu'),
    keras.layers.BatchNormalization(axis=-1),
    keras.layers.Conv2D(32,(3,3),activation='relu'),
    keras.layers.BatchNormalization(axis=-1),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    keras.layers.Conv2D(64,(3,3),activation='relu'),
    keras.layers.BatchNormalization(axis=-1),
    keras.layers.Conv2D(64,(3,3),activation='relu'),
    keras.layers.BatchNormalization(axis=-1),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(512),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(100),
    keras.layers.Dense(100,activation=tf.nn.softmax)
])

#model Complie

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# model fit

model.fit(train_x,train_y,epochs=10)

#model Validation

val_loss, val_acc = model.evaluate(test_x[:10000], test_y[:10000])
print('Valdiation accuracy:', val_acc)

'''


if __name__ == "__main__":
    try:
        train = pd.read_pickle(sys.argv[1])
        train_data = train['data'].values
        train_target = train['target'].values
        test = pd.read_pickle(sys.argv[2])
        test_data = test['data'].values
        result = run(train_data, train_target, test_data)
    except Exception:
        print("Error occured")

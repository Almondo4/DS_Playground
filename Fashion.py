from sklearn.datasets import load_sample_image
import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
#
# china = load_sample_image("china.jpg")/255
# flower = load_sample_image("flower.jpg")


# STEP ONE: Preprocessing
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_aug = ImageDataGenerator(rescale=1./255)
#TODO here you can try later the rescale without the point and with the other options
training_set = train_aug.flow_from_directory(
    "./dataset/training_set",
    batch_size=32,
    target_size=(64,64),
    class_mode="binary"
)


## testing set

test_aug = ImageDataGenerator(rescale=1./255)
testing_set = test_aug.flow_from_directory(
    "./dataset/test_set",
    batch_size=32,
    target_size=(64, 64),
    class_mode="binary"
)
## TODO: If i chage the target size would it still work

# STEP TWO : building the CNN

CNN = tf.keras.models.Sequential()
## convolutional Layer
CNN.add(tf.keras.layers.Conv2D(32,3,activation="relu",input_shape=(64,64,3)))
# TODO: check the idea of standard filters and which to use where
CNN.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))
# Pool size is a square (n*n)
## Second conv Layer
CNN.add(tf.keras.layers.Conv2D(32,3,activation="relu"))
# TODO: check the idea of standard filters and which to use where
CNN.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))

# STEP 3 : flattening
CNN.add(tf.keras.layers.Flatten())

# STEP 4: Fully connected layer

CNN.add(tf.keras.layers.Dense(activation="relu",units=64))
CNN.add(tf.keras.layers.Dense(units=128, activation='relu'))
# Output layer
CNN.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

# Training the

        # which optimzer
        # which loss
# compilation
CNN.compile(optimizer="adam",loss="binary_crossentropy",metrics="acc")
# running the training
CNN.fit(x= training_set,validation_data= testing_set, batch_size=64, epochs=30)

# Making a Prediction:

import numpy as np
from tensorflow.keras.preprocessing import image

dog_image = image.load_img("./dataset/single_prediction/cat_or_dog_1.jpg", target_size=(64,64))
cat_image = image.load_img("./dataset/single_prediction/cat_or_dog_2.jpg", target_size=(64,64))

dog_image = image.img_to_array(dog_image)
dog_image =np.expand_dims(dog_image,axis =0) # to add an extra dimension at the begining for batches

result =CNN.predict(dog_image)

#TODO How to know indices of each class
print("Indices: /n",training_set.class_indices)
if result[0][0] == 0:
        print("Predicted: ",result[0][0],", Answer: Dog (0)")
else:
    print("Predicted: ", result[0][0], ", Answer: Cat (1)")

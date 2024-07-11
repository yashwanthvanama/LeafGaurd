#!/usr/bin/env python
# coding: utf-8

# In[ ]:


try:
  # Use the %tensorflow_version magic if in colab.
  get_ipython().run_line_magic('tensorflow_version', '2.x')
except Exception:
  get_ipython().system('pip install -U "tensorflow-gpu==2.0.0rc0"')


# In[ ]:


get_ipython().system('pip install -U tensorflow_hub')
get_ipython().system('pip install -U tensorflow_datasets')


# In[ ]:


import time
import numpy as np
import matplotlib.pylab as plt
import os
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

from tensorflow.keras import layers


# In[ ]:


#Load data
zip_f=tf.keras.utils.get_file(origin='https://storage.googleapis.com/plantdata/PlantVillage.zip', 
 fname='PlantVillage.zip', extract=True)
#Create the training and validation directories
data_directory = os.path.join(os.path.dirname(zip_f), 'PlantVillage')
train_directory = os.path.join(data_directory, 'train')
validation_directory = os.path.join(data_directory, 'validation')


# In[ ]:


get_ipython().system('wget https://github.com/obeshor/Plant-Diseases-Detector/archive/master.zip')
get_ipython().system('unzip master.zip;')
import json
with open('Plant-Diseases-Detector-master/categories.json', 'r') as s:
    category_to_name = json.load(s)
    classes = list(category_to_name.values())


# In[ ]:


batch_size = 32
IMG_SHAPE = 224


# In[ ]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
image_generator_train = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=45,
                    width_shift_range=.15,
                    height_shift_range=.15,
                    horizontal_flip=True,
                    zoom_range=0.5
                    )


train_data_generator = image_generator_train.flow_from_directory(
                                                batch_size=batch_size,
                                                directory=train_directory,
                                                shuffle=True,
                                                target_size=(IMG_SHAPE,IMG_SHAPE),
                                                class_mode='sparse'
                                                )


# In[ ]:


image_generator_validation = ImageDataGenerator(rescale=1./255)

validation_data_generator = image_generator_validation.flow_from_directory(batch_size=batch_size,
                                                 directory=validation_directory,
                                                 target_size=(IMG_SHAPE, IMG_SHAPE),
                                                 class_mode='sparse')


# In[ ]:


URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
featuresextractor = hub.KerasLayer(URL,
                                   input_shape=(IMG_SHAPE, IMG_SHAPE,3))


# In[ ]:


featuresextractor.trainable = False


# In[ ]:


plant_model = tf.keras.Sequential([
  featuresextractor,
  layers.Dense(38, activation='softmax')
])

plant_model.summary()


# In[ ]:


plant_model.compile(
  optimizer='adam', 
  loss=tf.losses.SparseCategoricalCrossentropy(),
  metrics=['accuracy'])
history = plant_model.fit(train_data_generator,
                    epochs=3,
                    validation_data=validation_data_generator)


# In[ ]:


t = time.time()

export_path_sm = "./{}".format(int(t))
print(export_path_sm)
tf.saved_model.save(plant_model, export_path_sm)


# In[ ]:


reload_sm_keras = tf.keras.models.load_model(
  export_path_sm,
  custom_objects={'KerasLayer': hub.KerasLayer})

reload_sm_keras.summary()


# In[ ]:


def predict_reload(image):
    probabilities = reload_sm_keras.predict(np.asarray([img]))[0]
    class_idx = np.argmax(probabilities)
    
    return {classes[class_idx]: probabilities[class_idx]}


# In[ ]:


import random
import cv2

# Utility
import itertools
import random
from collections import Counter
from glob import iglob


def load_image(filename):
    img = cv2.imread(os.path.join(data_directory, validation_directory, filename))
    img = cv2.resize(img, (IMG_SHAPE, IMG_SHAPE) )
    img = img /255
    
    return img
for idx, filename in enumerate(random.sample(validation_data_generator.filenames, 2)):
    print("SOURCE: class: %s, file: %s" % (os.path.split(filename)[0], filename))
    
    img = load_image(filename)
    prediction = predict_reload(img)
    print("PREDICTED: class: %s, confidence: %f" % (list(prediction.keys())[0], list(prediction.values())[0]))
    plt.imshow(img)
    plt.figure(idx)    
    plt.show()


# In[ ]:


get_ipython().system('mkdir "tflite_models"')
TFLITE_MODEL = "tflite_models/plant_disease_model.tflite"


# Get the concrete function from the Keras model.
run_model = tf.function(lambda x : reload_sm_keras(x))

# Save the concrete function.
concrete_func = run_model.get_concrete_function(
    tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype)
)

# Convert the model to standard TensorFlow Lite model
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
converted_tflite_model = converter.convert()
open(TFLITE_MODEL, "wb").write(converted_tflite_model)


# In[ ]:


with open('labels.txt', 'w') as f:
  f.write('\n'.join(class_names))


# In[ ]:


try:
  from google.colab import files

  files.download(tflite_model_file)
  files.download('labels.txt')
except:
  pass


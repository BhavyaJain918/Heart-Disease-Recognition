import os
import cv2
import keras
import numpy
import Augmentor
from keras.api.preprocessing.image import load_img , img_to_array
from tensorflow.python.keras.layers.advanced_activations import ReLU , Softmax

img_augmentor = Augmentor.Pipeline("" , output_directory = "")     # Path to dataset
img_augmentor.rotate(0.4 , 14 , 8)
img_augmentor.random_color(0.4 , 0.4 , 0.8)
img_augmentor.random_brightness(0.4 , 0.6 , 1.6)

img_augmentor.sample(14000)

(dataset_train , dataset_valid) = keras.utils.image_dataset_from_directory("" , labels = "inferred" , label_mode = "categorical" , image_size = (512 , 512) , batch_size = 32 , validation_split = 0.30 , subset = "both" , seed = 42)   # Path to training directory
 

print(dataset_train)

model = keras.Sequential()

model.add(keras.layers. Conv2D(filters = 20 , kernel_size = 2 , strides = 2 , activation = ReLU() , input_shape = [512 , 512 , 3]))

model.add(keras.layers. Conv2D(filters = 20 , kernel_size = 4 , strides = 2 , activation = ReLU()))
model.add(keras.layers.MaxPool2D(pool_size = 4 , strides = 2))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(units = 32 , activation = ReLU()))
model.add(keras.layers.Dense(units = 64 , activation = ReLU()))

model.add(keras.layers.Dense(units = 4 , activation = Softmax()))

model.summary()
model.compile(loss = "categorical_crossentropy" , optimizer = "adam" , metrics = ["accuracy"])

model.fit(x = dataset_train , validation_data = dataset_valid , batch_size = 32 , epochs = 16)

def predict(img):
    ready_img = img_to_array(img.resize((512 , 512)))
    ready_img = numpy.expand_dims(ready_img , axis = 0)

    result = (model.predict(ready_img))
    result_idx = result.argmax()
    for images in img_augmentor.class_labels:
        if images[1] == result_idx:
            print("Predicted class is : " , images[0])
    print("Probability of each class : " , result)

cam = cv2.VideoCapture(0)

while True:
  ret , frame = cam.read()
  cv2.imshow("Frame" , frame)
  key = cv2.waitKey(1)
  if key == ord("q") or cv2.getWindowProperty("Frame" , cv2.WND_PROP_VISIBLE) < 1:
    cv2.destroyAllWindows()
    break
  elif key == ord("c"):
    cv2.imwrite("Saved_Image.jpg" , frame)
    image = load_img(os.getcwd() + "\\Saved_Image.jpg")
    predict(image)
  else:
    continue

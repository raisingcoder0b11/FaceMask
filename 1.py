import os,cv2
import numpy as np
from keras.utils import np_utils
data_path = 'train'
categories = os.listdir(data_path)
labels = [i for i in range(len(categories))]
label_dict = dict(zip(categories,labels))
print(label_dict)
print(categories)
print(labels)
image_size = 100
data = []
target = []


for category in categories:
    folder_path = os.path.join(data_path,category)
    image_names = os.listdir(folder_path)
    for image_name in image_names:
        image_path = os.path.join(folder_path,image_name)
        img = cv2.imread(image_path)

        try:
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            resize = cv2.resize(gray,(image_size,image_size))
            data.append(resize)
            target.append(label_dict[category])

        except Exception as e:
            print("Exception ",e)
data=np.array(data)/255.0
data=np.reshape(data,(data.shape[0],image_size,image_size,1))
target=np.array(target)
new_target=np_utils.to_categorical(target)
np.save('data',data)
np.save('target',new_target)
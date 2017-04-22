
# coding: utf-8

# In[1]:

import cv2
import keras
from keras.applications.imagenet_utils import preprocess_input
from keras.backend.tensorflow_backend import set_session
from keras.models import Model
from keras.preprocessing import image
import numpy as np
import pickle
import glob
import os
from random import shuffle
from scipy.misc import imread
from scipy.misc import imresize
from sklearn.utils import shuffle
import pandas as pd
import tensorflow as tf

from ssd import SSD300
from ssd_training import MultiboxLoss
from ssd_utils import BBoxUtility



get_ipython().magic(u'load_ext autoreload')
get_ipython().magic(u'autoreload 2')

np.set_printoptions(suppress=True)

# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.9
# set_session(tf.Session(config=config))


# In[2]:

# some constants
INPUT_WIDTH = 300
INPUT_HEIGHT = 300 
NUM_CLASSES = 8 + 1 # 1 because of background (needed for training)
DATA_PATH = '/a/data/fisheries_monitoring/data/'

input_shape = (INPUT_HEIGHT, INPUT_WIDTH, 3)
classes = ['ALB', 'DOL', 'NoF', 'SHARK', 'BET', 'LAG', 'OTHER', 'YFT']


# In[3]:

priors = pickle.load(open('prior_boxes_ssd300.pkl', 'rb'))
bbox_util = BBoxUtility(NUM_CLASSES, priors)


# In[4]:

aug_folders = glob.glob(DATA_PATH + 'localizers/*')
for i, folder in enumerate(aug_folders):
    print "index:", i, '\t', "folder name:", folder
    
aug_folders = aug_folders[1:]


# In[5]:

def data_generator(batch_size, bbox_util, boxes, classes, INPUT_WIDTH, INPUT_HEIGHT):
    # Shuffle labels to ensure 
    labels = shuffle(boxes)
    while True:
        #img_batch = np.zeros((batch_size, INPUT_HEIGHT, INPUT_WIDTH, 3))
        #box_batch = np.zeros((batch_size, 4))
        img_batch = []
        target_batch = []
        for index, row in boxes.iterrows():
            # Get next file to read
            file_name = row["img"]
            
            # Obtain label for that file (remember file path in pattern aug_type/label/frame.png)
            label = file_name.split('/')[1]
            # Then create a one-hot encoding vector for the classes
            label_vector = [1 if i == label else 0 for i in classes]

            # Read img and preprocess it
            path = DATA_PATH + 'localizers/' + file_name
            img = image.load_img(path)
            width, height = img.size
            img = img.resize((INPUT_HEIGHT, INPUT_WIDTH))
            img = image.img_to_array(img)
            img /= 255
            img_batch.append(img)
            
            # Read box and normalize its measures
            old_x, old_y, old_w, old_h = row[["x","y","w","h"]]
            new_x = old_x / width
            new_y = old_y / height
            new_w = old_w / width
            new_h = old_h / height
            
            target = [new_x, new_y, new_x + new_w, new_y + new_h] + label_vector
            target = bbox_util.assign_boxes(np.array(target)[np.newaxis, :])
            target_batch.append(target)
            
            if len(target_batch) == batch_size:
                tmp_img_batch = np.array(img_batch)
                tmp_target_batch = np.array(target_batch)
                img_batch = []
                target_batch = []
                yield (tmp_img_batch, tmp_target_batch)
                
        
def load_all_labels(aug_folders):
    all_targets = pd.DataFrame(columns = ["img", "x","y","w","h"])
    for folder in aug_folders:
        folder_name = os.path.basename(folder)
        
        print "Loading data augmentation folder:", folder_name
        targets = pd.read_csv(folder + '/superboxes.csv', names = ["img", "x","y","w","h"])
        targets = targets.sort_values(by = "img")
        targets["img"] = folder_name + '/' + targets["img"]
        print "Number of examples:", len(targets)
        print
        
        all_targets = all_targets.append(targets)
    print "total number of examples: ", len(all_targets)
    return all_targets


def train_val_test_split(all_labels, val_size, test_size):
    all_labels = shuffle(all_labels)
    test_labels = all_labels[0:test_size]
    val_labels = all_labels[test_size:test_size + val_size]
    train_labels = all_labels[test_size + val_size:]
    return train_labels, val_labels, test_labels


# In[6]:

#gt = pickle.load(open('gt_pascal.pkl', 'rb'))
#keys = sorted(gt.keys())
#num_train = int(round(0.8 * len(keys)))
#train_keys = keys[:num_train]
#val_keys = keys[num_train:]
#num_val = len(val_keys)

all_labels = load_all_labels(aug_folders)

train_labels, val_labels, test_labels = train_val_test_split(all_labels, 2000, 1000)


# In[7]:

batch_size = 30
train_steps = np.ceil(len(train_labels)/batch_size)

gen = data_generator(batch_size, bbox_util, train_labels, classes, INPUT_HEIGHT, INPUT_WIDTH)


# In[8]:

model = SSD300(input_shape, num_classes=NUM_CLASSES)
model.load_weights(DATA_PATH + 'models/localizers/weights_SSD300.hdf5', by_name=True)


# In[9]:

freeze = ['input_1', 'conv1_1', 'conv1_2', 'pool1',
          'conv2_1', 'conv2_2', 'pool2',
          'conv3_1', 'conv3_2', 'conv3_3', 'pool3',
          'conv4_1', 'conv4_2'] 
#'conv4_3', 'pool4']

for L in model.layers:
    if L.name in freeze:
        L.trainable = False


# In[10]:

def schedule(epoch, decay=0.9):
    return base_lr * decay**(epoch)

callbacks = [keras.callbacks.ModelCheckpoint('./checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                                             verbose=1,
                                             save_weights_only=True),
             keras.callbacks.LearningRateScheduler(schedule)]


# In[11]:

base_lr = 3e-4
optim = keras.optimizers.Adam(lr=base_lr)
# optim = keras.optimizers.RMSprop(lr=base_lr)
# optim = keras.optimizers.SGD(lr=base_lr, momentum=0.9, decay=decay, nesterov=True)
model.compile(optimizer=optim,
              loss=MultiboxLoss(NUM_CLASSES, neg_pos_ratio=2.0).compute_loss)


# In[ ]:

nb_epoch = 30
val_gen = data_generator(batch_size, bbox_util, val_labels, classes, INPUT_WIDTH, INPUT_HEIGHT)
val_steps = np.ceil(len(val_labels)/batch_size)

history = model.fit_generator(gen, len(train_labels),
                              nb_epoch, verbose=2,
                              callbacks=callbacks,
                              validation_data=val_gen,
                              nb_val_samples=len(val_labels),
                              nb_worker=1)


# In[ ]:

inputs = []
images = []
img_path = path_prefix + sorted(val_keys)[0]
img = image.load_img(img_path, target_size=(300, 300))
img = image.img_to_array(img)
images.append(imread(img_path))
inputs.append(img.copy())
inputs = preprocess_input(np.array(inputs))


# In[ ]:

preds = model.predict(inputs, batch_size=1, verbose=1)
results = bbox_util.detection_out(preds)


# In[ ]:

for i, img in enumerate(images):
    # Parse the outputs.
    det_label = results[i][:, 0]
    det_conf = results[i][:, 1]
    det_xmin = results[i][:, 2]
    det_ymin = results[i][:, 3]
    det_xmax = results[i][:, 4]
    det_ymax = results[i][:, 5]

    # Get detections with confidence higher than 0.6.
    top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.6]

    top_conf = det_conf[top_indices]
    top_label_indices = det_label[top_indices].tolist()
    top_xmin = det_xmin[top_indices]
    top_ymin = det_ymin[top_indices]
    top_xmax = det_xmax[top_indices]
    top_ymax = det_ymax[top_indices]

    for i in range(top_conf.shape[0]):
        xmin = int(round(top_xmin[i] * img.shape[1]))
        ymin = int(round(top_ymin[i] * img.shape[0]))
        xmax = int(round(top_xmax[i] * img.shape[1]))
        ymax = int(round(top_ymax[i] * img.shape[0]))
        score = top_conf[i]
        label = int(top_label_indices[i])
#         label_name = voc_classes[label - 1]
        display_txt = '{:0.2f}, {}'.format(score, label)
        coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
    


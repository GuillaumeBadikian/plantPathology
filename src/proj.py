
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# %%

# %%


# %%

import os
import gc
import re

import cv2
import math
import numpy as np
import scipy as sp
import pandas as pd

import tensorflow as tf
from IPython.display import SVG
import efficientnet.tfkeras as efn
from keras.utils import plot_model
import tensorflow.keras.layers as L
from keras.utils import model_to_dot
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
#from kaggle_datasets import KaggleDatasets
from tensorflow.keras.applications import DenseNet121

import seaborn as sns
from tqdm import tqdm
import matplotlib.cm as cm
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

tqdm.pandas()
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

np.random.seed(0)
tf.random.set_seed(0)

import warnings

warnings.filterwarnings("ignore")

# %%

os.chdir(os.path.dirname(__file__))

'''strategy = tf.distribute.experimental.TPUStrategy(tpu)

AUTO = tf.data.experimental.AUTOTUNE
tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)
strategy = tf.distribute.experimental.TPUStrategy(tpu)

BATCH_SIZE = 16 * strategy.num_replicas_in_sync
'''
'''gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Create 2 virtual GPUs with 1GB memory each
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, False)    
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=512),
         tf.config.experimental.VirtualDeviceConfiguration(memory_limit=512)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    strategy = tf.distribute.MirroredStrategy()
    print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)
'''
'''gpus = tf.config.experimental.list_physical_devices('GPU')


if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    #for gpu in gpus:
    #tf.config.experimental.set_memory_growth(gpus[3], True)
    #tf.config.experimental.set_visible_devices(gpus[3], 'GPU')

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
    #logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    strategy = tf.distribute.MirroredStrategy()
    #print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)


tf.debugging.set_log_device_placement(True)
'''





gpus = tf.config.experimental.list_physical_devices("GPU")
#for device in gpus:
#    tf.config.experimental.set_memory_growth(device, True)
if gpus:
  # Create 2 virtual GPUs with 1GB memory each
  try:
    #tf.config.experimental.set_virtual_device_configuration(
     #  gpus[1],
     #   [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024),
     #    tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)


strategy = tf.distribute.MirroredStrategy(devices=["GPU:0","GPU:1","GPU:2","GPU:3"])
#strategy = tf.distribute.MirroredStrategy(devices=["/GPU:0", "/job:localhost/replica:0/task:0/device:GPU:1","/job:localhost/replica:0/task:0/device:GPU:2"])
#trategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(
#   tf.distribute.experimental.CollectiveCommunication.NCCL)
#strategy = tf.distribute.MirroredStrategy(
#    cross_device_ops=tf.distribute.ReductionToOneDevice())
GCS_DS_PATH = "./data/"
AUTO = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 8  #strategy.num_replicas_in_sync 
# %%

EPOCHS = 20
SAMPLE_LEN = 100
IMAGE_PATH = "./data/images/"
TEST_PATH = "./data/test.csv"
TRAIN_PATH = "./data/train.csv"
SUB_PATH = "./data/sample_submission.csv"

sub = pd.read_csv(SUB_PATH)
test_data = pd.read_csv(TEST_PATH)
train_data = pd.read_csv(TRAIN_PATH)

# %%

train_data.head()

# %%

def load_image(image_id):
    file_path = image_id + ".jpg"
    image = cv2.imread(IMAGE_PATH + file_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


train_images = train_data["image_id"][:SAMPLE_LEN].progress_apply(load_image)


def display_training_curves(training, validation, yaxis):
    if yaxis == "loss":
        ylabel = "Loss"
        title = "Loss vs. Epochs"
    else:
        ylabel = "Accuracy"
        title = "Accuracy vs. Epochs"

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(x=np.arange(1, EPOCHS + 1), mode='lines+markers', y=training, marker=dict(color="dodgerblue"),
                   name="Train"))

    fig.add_trace(
        go.Scatter(x=np.arange(1, EPOCHS + 1), mode='lines+markers', y=validation, marker=dict(color="darkorange"),
                   name="Val"))

    fig.update_layout(title_text=title, yaxis_title=ylabel, xaxis_title="Epochs", template="plotly_white")
    fig.show()
    fig.write_image(file="accu.png", format="svg")
# %%

def format_path(st):
    return GCS_DS_PATH + '/images/' + st + '.jpg'


test_paths = test_data.image_id.apply(format_path).values
train_paths = train_data.image_id.apply(format_path).values

train_labels = np.float32(train_data.loc[:, 'healthy':'scab'].values)
train_paths, valid_paths, train_labels, valid_labels = \
    train_test_split(train_paths, train_labels, test_size=0.15, random_state=2020)


# %%

def decode_image(filename, label=None, image_size=(512, 512)):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, image_size)

    if label is None:
        return image
    else:
        return image, label


def data_augment(image, label=None):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    if label is None:
        return image
    else:
        return image, label


# %%

train_dataset = (
    tf.data.Dataset
        .from_tensor_slices((train_paths, train_labels))
        .map(decode_image, num_parallel_calls=AUTO)
        .map(data_augment, num_parallel_calls=AUTO)
        .repeat()
        .shuffle(512)
        .batch(BATCH_SIZE)
        .prefetch(AUTO)
)

valid_dataset = (
    tf.data.Dataset
        .from_tensor_slices((valid_paths, valid_labels))
        .map(decode_image, num_parallel_calls=AUTO)
        .batch(BATCH_SIZE)
        .cache()
        .prefetch(AUTO)
)

test_dataset = (
    tf.data.Dataset
        .from_tensor_slices(test_paths)
        .map(decode_image, num_parallel_calls=AUTO)
        .batch(BATCH_SIZE)
)


# %%

def build_lrfn(lr_start=0.00001, lr_max=0.00005,
               lr_min=0.00001, lr_rampup_epochs=5,
               lr_sustain_epochs=0, lr_exp_decay=.8):
    #strategy = 8
    lr_max = lr_max * strategy.num_replicas_in_sync

    def lrfn(epoch):
        if epoch < lr_rampup_epochs:
            lr = (lr_max - lr_start) / lr_rampup_epochs * epoch + lr_start
        elif epoch < lr_rampup_epochs + lr_sustain_epochs:
            lr = lr_max
        else:
            lr = (lr_max - lr_min) * \
                 lr_exp_decay ** (epoch - lr_rampup_epochs \
                                  - lr_sustain_epochs) + lr_min
        return lr

    return lrfn


# %%

lrfn = build_lrfn()
STEPS_PER_EPOCH = train_labels.shape[0] // BATCH_SIZE
lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)

# %%

with strategy.scope():
    print("create model")
    model = tf.keras.Sequential([DenseNet121(input_shape=(512, 512, 3),
                                             weights='imagenet',
                                             include_top=False),
                                 L.GlobalAveragePooling2D(),
                                 L.Dense(train_labels.shape[1],
                                         activation='softmax')])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])
    model.summary()

# %%
print("training...")
history = model.fit(train_dataset,
                    epochs=EPOCHS,
                    callbacks=[lr_schedule],
                    steps_per_epoch=STEPS_PER_EPOCH,
                    validation_data=valid_dataset)

# %%
print(history.history['categorical_accuracy'], 
    history.history['val_categorical_accuracy'])
print("test")
probs_dnn = model.predict(test_dataset, verbose=1)
sub.loc[:, 'healthy':] = probs_dnn
sub.to_csv('submission_dnn.csv', index=False)
sub.head()

# %%

f = open("demofile2.txt", "a")
model.save("model_softmax_epoch_40.sav")

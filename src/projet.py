import cv2
import os
import numpy as np
import pandas as pd
from keras_applications.resnet50 import ResNet50
from pandas import DataFrame
import tensorflow as tf
import efficientnet.tfkeras as efn
import tensorflow.keras.layers as L
from tensorflow.keras.models import Model
from tensorflow.keras.applications import DenseNet121
from tensorflow.python.keras.layers import GlobalAveragePooling2D, Dense
from tqdm import tqdm

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import warnings
tqdm.pandas()

import plotly.graph_objects as go



class Data:
    def __init__(self, sub: DataFrame, train: DataFrame, test: DataFrame):
        self.sub = sub
        self.train = train
        self.test = test


class PlantPathology:

    def __init__(self):
        self.__epoch = 20
        self.__sample_len = 100
        self.__path = "./data/"
        self.__image_path = self.__path + "/images/"
        self.__train_path = self.__path + "/train.csv"
        self.__test_path = self.__path + "/test.csv"
        self.__sub_path = self.__path + "/sample_submission.csv"
        self.__preprocess_path = "image_preprocessing"
        self.__data = None
        self.__batch_size = 10
        self.__auto = None
        self.__strategy = None
        self.__model = None
        self.__gpu_devices = ["GPU:0", "GPU:1", "GPU:2", "GPU:3"]
        self.__history_file = "./data/history"

    def __load_image(self, image_id):
        file_path = image_id + ".jpg"
        image = cv2.imread(self.__image_path + os.sep + file_path)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def __preprocessing(self) -> list:
        self.__load()
        train_images = self.__data.train["image_id"][:2].progress_apply(self.__load_image)
        train_images = self.__crop(train_images)
        train_images = self.__blur_data(train_images, (20, 20))
        #dir_prepro = self.__path + os.sep + self.__preprocess_path
        #shutil.rmtree(dir_prepro, ignore_errors=True)
        #if not os.path.exists(dir_prepro):
        #   os.makedirs(dir_prepro)

        #for i,j in enumerate(train_images):
        #    cv2.imwrite("{}/Test_{}.jpg".format(dir_prepro,i), cv2.resize(j, (2048,1365), interpolation = cv2.INTER_AREA))

        return train_images

    def __load(self) -> Data:
        sub = pd.read_csv(self.__sub_path)
        test_data = pd.read_csv(self.__test_path)
        train_data = pd.read_csv(self.__train_path)
        # train_images = train_data["image_id"][:100].progress_apply(self.load_image)
        self.__data = Data(sub, train_data, test_data)
        return self.__data

    @staticmethod
    def showPieChart(data):
        fig = go.Figure([go.Pie(labels=data.columns[1:],
                                values=data.iloc[:, 1:].sum().values)])
        fig.update_layout(title_text="Pie chart of targets", template="simple_white")
        fig.data[0].marker.line.color = 'rgb(0, 0, 0)'
        fig.data[0].marker.line.width = 0.5
        fig.show()

    def __edge_and_cut_ref(self, img, size=(100, 200)):
        emb_img = img.copy()
        edges = cv2.Canny(img, size[0], size[1])
        edge_coors = []
        for i in range(edges.shape[0]):
            for j in range(edges.shape[1]):
                if edges[i][j] != 0:
                    edge_coors.append((i, j))
        row_min = edge_coors[np.argsort([coor[0] for coor in edge_coors])[0]][0]
        row_max = edge_coors[np.argsort([coor[0] for coor in edge_coors])[-1]][0]
        col_min = edge_coors[np.argsort([coor[1] for coor in edge_coors])[0]][1]
        col_max = edge_coors[np.argsort([coor[1] for coor in edge_coors])[-1]][1]

        im3 = emb_img[row_min:row_max, col_min:col_max]
        return im3

    def __crop(self, data):
        re = []
        for i in data:
            # i = self.__edge_and_cut_ref(i)
            re.append(self.__edge_and_cut_ref(i))
        return re

    @staticmethod
    def __edge_and_cut(img):
        emb_img = img.copy()
        edges = cv2.Canny(img, 100, 200)
        edge_coors = []
        for i in range(edges.shape[0]):
            for j in range(edges.shape[1]):
                if edges[i][j] != 0:
                    edge_coors.append((i, j))

        row_min = edge_coors[np.argsort([coor[0] for coor in edge_coors])[0]][0]
        row_max = edge_coors[np.argsort([coor[0] for coor in edge_coors])[-1]][0]
        col_min = edge_coors[np.argsort([coor[1] for coor in edge_coors])[0]][1]
        col_max = edge_coors[np.argsort([coor[1] for coor in edge_coors])[-1]][1]
        new_img = img[row_min:row_max, col_min:col_max]

        emb_img[row_min - 10:row_min + 10, col_min:col_max] = [255, 0, 0]
        emb_img[row_max - 10:row_max + 10, col_min:col_max] = [255, 0, 0]
        emb_img[row_min:row_max, col_min - 10:col_min + 10] = [255, 0, 0]
        emb_img[row_min:row_max, col_max - 10:col_max + 10] = [255, 0, 0]

        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(30, 20))
        ax[0].imshow(img, cmap='gray')
        ax[0].set_title('Original Image', fontsize=24)
        ax[1].imshow(edges, cmap='gray')
        ax[1].set_title('Canny Edges', fontsize=24)
        ax[2].imshow(emb_img, cmap='gray')
        ax[2].set_title('Bounding Box', fontsize=24)
        plt.show()

    @staticmethod
    def invert(img):
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(30, 20))
        ax[0].imshow(img)
        ax[0].set_title('Original Image', fontsize=24)
        ax[1].imshow(cv2.flip(img, 0))
        ax[1].set_title('Vertical Flip', fontsize=24)
        ax[2].imshow(cv2.flip(img, 1))
        ax[2].set_title('Horizontal Flip', fontsize=24)
        plt.show()

    @staticmethod
    def convolution(img):
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 20))
        kernel = np.ones((7, 7), np.float32) / 25
        conv = cv2.filter2D(img, -1, kernel)
        ax[0].imshow(img)
        ax[0].set_title('Original Image', fontsize=24)
        ax[1].imshow(conv)
        ax[1].set_title('Convolved Image', fontsize=24)
        plt.show()

    @staticmethod
    def blur(img, val=(100, 100)):
        return cv2.blur(img, val)

    def __blur_data(self, data, blur):
        li = []
        for i in data:
            li.append(self.blur(i, blur))
        return li

    def useTPU(self):
        self.__auto = tf.data.experimental.AUTOTUNE
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        self.__strategy = tf.distribute.experimental.TPUStrategy(tpu)
        self.__batch_size = 16 * self.__strategy.num_replicas_in_sync

    def useGPU(self):
        gpus = tf.config.experimental.list_physical_devices("GPU")
        if gpus:
            try:
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            except RuntimeError as e:
                print(e)

        self.__strategy = tf.distribute.MirroredStrategy(devices=self.__gpu_devices)
        self.__batch_size = 8 * self.__strategy.num_replicas_in_sync
    def format_path(self, st):
        #return self.__path + os.sep + self.__preprocess_path + os.sep + st + os.sep + '.jpg'
        return self.__path + os.sep + "images/" + os.sep + st + '.jpg'

    def decode_image(self,filename, label=None, image_size=(512, 512)):
        bits = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(bits, channels=3)
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.image.resize(image, image_size)
        if label is None:
            return image
        else:
            return image, label

    def data_augment(self, image, label=None):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.adjust_contrast(image, 2)

        if label is None:
            return image
        else:
            return image, label

    def build_lrfn(self, lr_start=0.00001, lr_max=0.00005,
                   lr_min=0.00001, lr_rampup_epochs=5,
                   lr_sustain_epochs=0, lr_exp_decay=.8):
        if self.__strategy is not None:
            lr_max = lr_max * self.__strategy.num_replicas_in_sync
        else:
            lr_max = lr_max * 8

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

    def dense_net(self, train_labels):
        if self.__strategy is not None:
            with self.__strategy.scope():
                model = tf.keras.Sequential([DenseNet121(input_shape=(512, 512, 3),
                                                         weights='imagenet',
                                                         include_top=False),
                                             L.GlobalAveragePooling2D(),
                                             L.Dense(train_labels.shape[1],
                                                     activation='softmax')])
                model.compile(optimizer='adam',
                              loss='categorical_crossentropy',
                              metrics=['categorical_accuracy'])
                self.__model = model
        else:
            model = tf.keras.Sequential([DenseNet121(input_shape=(512, 512, 3),
                                                     weights='imagenet',
                                                     include_top=False),
                                         L.GlobalAveragePooling2D(),
                                         L.Dense(train_labels.shape[1],
                                                 activation='softmax')])
            model.compile(optimizer='adam',
                          loss='categorical_crossentropy',
                          metrics=['categorical_accuracy'])
            self.__model = model

    def efficient_net(self, train_labels):
        if self.__strategy is not None:
            with self.__strategy.scope():
                model = tf.keras.Sequential([efn.EfficientNetB7(input_shape=(512, 512, 3),
                                                                weights='imagenet',
                                                                include_top=False),
                                             L.GlobalAveragePooling2D(),
                                             L.Dense(train_labels.shape[1],
                                                     activation='softmax')])

                model.compile(optimizer='adam',
                              loss='categorical_crossentropy',
                              metrics=['categorical_accuracy'])
                self.__model = model
        else:
            model = tf.keras.Sequential([efn.EfficientNetB7(input_shape=(512, 512, 3),
                                                            weights='imagenet',
                                                            include_top=False),
                                         L.GlobalAveragePooling2D(),
                                         L.Dense(train_labels.shape[1],
                                                 activation='softmax')])

            model.compile(optimizer='adam',
                          loss='categorical_crossentropy',
                          metrics=['categorical_accuracy'])
            self.__model = model

    def res_net(self, train_labels):
        model_finetuned = ResNet50(include_top=False, weights='imagenet', input_shape=(384, 384, 3))
        x = model_finetuned.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation="relu")(x)
        x = Dense(64, activation="relu")(x)
        predictions = Dense(4, activation="softmax")(x)
        model_finetuned = Model(inputs=model_finetuned.input, outputs=predictions)
        model_finetuned.compile(optimizer='adam',
                                loss='categorical_crossentropy',
                                metrics=['accuracy'])
        model_finetuned.summary()

    def __display_training_curves(self, training, validation, yaxis):
        if yaxis == "loss":
            ylabel = "Loss"
            title = "Loss vs. Epochs"
        else:
            ylabel = "Accuracy"
            title = "Accuracy vs. Epochs"

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(x=np.arange(1, self.__epoch + 1), mode='lines+markers', y=training, marker=dict(color="dodgerblue"),
                       name="Train"))

        fig.add_trace(
            go.Scatter(x=np.arange(1, self.__epoch + 1), mode='lines+markers', y=validation, marker=dict(color="darkorange"),
                       name="Val"))

        fig.update_layout(title_text=title, yaxis_title=ylabel, xaxis_title="Epochs", template="plotly_white")
        fig.show()

    def __test(self, test_dataset):
        probs_dnn = self.__model.predict(test_dataset, verbose=1)
        self.__data.sub.loc[:, 'healthy':] = probs_dnn
        self.__data.sub.to_csv('submission_dnn.csv', index=False)
        self.__data.sub.head()
        print(self.__data.sub.head())

    def run(self):
        self.__load()
        #self.__preprocessing()
        test_paths = self.__data.test.image_id.apply(self.format_path).values
        train_paths = self.__data.train.image_id.apply(self.format_path).values
        train_labels = np.float32(self.__data.train.loc[:, 'healthy':'scab'].values)
        train_paths, valid_paths, train_labels, valid_labels = \
            train_test_split(train_paths, train_labels, test_size=0.15, random_state=2020)
        train_dataset = (
            tf.data.Dataset
                .from_tensor_slices((train_paths, train_labels))
                .map(self.decode_image, num_parallel_calls=self.__auto)
                .map(self.data_augment, num_parallel_calls=self.__auto)
                .repeat()
                .shuffle(512)
                .batch(self.__batch_size)
                .prefetch(self.__auto)
        )

        valid_dataset = (
            tf.data.Dataset
                .from_tensor_slices((valid_paths, valid_labels))
                .map(self.decode_image, num_parallel_calls=self.__auto)
                .batch(self.__batch_size)
                .cache()
                .prefetch(self.__auto)
        )

        test_dataset = (
            tf.data.Dataset
                .from_tensor_slices(test_paths)
                .map(self.decode_image, num_parallel_calls=self.__auto)
                .batch(self.__batch_size)
        )
        lrfn = self.build_lrfn()
        STEPS_PER_EPOCH = 200 # train_labels.shape[0]
        lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)

        self.dense_net(train_labels)

        history = self.__model.fit(train_dataset,
                            epochs=self.__epoch,
                            callbacks=[lr_schedule],
                            steps_per_epoch=STEPS_PER_EPOCH,
                            validation_data=valid_dataset)

        hist_df = pd.DataFrame(history.history)

        with open(self.__history_file+".json", mode='w') as f:
            hist_df.to_json(f)

        with open(self.__history_file+".csv", mode='w') as f:
            hist_df.to_csv(f)

        self.__test(test_dataset)


if __name__ == '__main__':
    os.chdir(os.path.dirname(__file__))
    warnings.filterwarnings("ignore")
    tqdm.pandas()
    plant = PlantPathology()
    plant.useGPU()
    plant.run()

import typer
from tqdm import tqdm
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from tensorflow.keras import layers, Model, backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
import time
import os

# Check for GPU availability
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Configure GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Set global TensorFlow configurations
K.set_image_data_format('channels_last')

# Define constants
data_location = 'resources/words'
words_txt_location = 'resources/words.txt'
input_length = 30
max_text_len = 16
img_w, img_h = 128, 64

# Define character set
letters = [' ', '!', '"', '#', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/',
           '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?',
           'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
           'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
           'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
           'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
num_classes = len(letters) + 1

def get_paths_and_gts(partition_split_file):
    paths_and_gts = []
    with open(partition_split_file) as f:
        for line in f:
            if not line or line.startswith('#'):
                continue
            line_split = line.strip().split(' ')
            directory_split = line_split[0].split('-')
            image_location = f'{data_location}/{directory_split[0]}/{directory_split[0]}-{directory_split[1]}/{line_split[0]}.png'
            gt_text = ' '.join(line_split[8:])
            if not os.path.exists(image_location):
                print(f"Warning: Image file does not exist: {image_location}")
            paths_and_gts.append([image_location, gt_text])
    return paths_and_gts

def preprocess(path, img_w, img_h):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error reading image: {path}")
        return np.zeros((img_h, img_w), dtype=np.float32)
    img = cv2.resize(img, (img_w, img_h))
    img = img.astype(np.float32) / 255.0
    return img

def text_to_labels(text):
    return [letters.index(c) for c in text]

def labels_to_text(labels):
    return ''.join([letters[i] for i in labels if i != -1])

class TextImageGenerator:
    def __init__(self, data, img_w, img_h, batch_size, i_len, max_text_len):
        self.img_h = img_h
        self.img_w = img_w
        self.batch_size = batch_size
        self.max_text_len = max_text_len
        self.samples = data
        self.n = len(self.samples)
        self.i_len = i_len
        self.indexes = list(range(self.n))
        self.cur_index = 0

    def build_data(self):
        self.imgs = np.zeros((self.n, self.img_h, self.img_w))
        self.texts = []
        for i, (img_filepath, text) in tqdm(enumerate(self.samples), total=self.n, desc="Building Data"):
            try:
                img = preprocess(img_filepath, self.img_w, self.img_h)
                self.imgs[i, :, :] = img
                self.texts.append(text)
            except Exception as e:
                print(f"Error processing image {img_filepath}: {str(e)}")

    def next_sample(self):
        self.cur_index += 1
        if self.cur_index >= self.n:
            self.cur_index = 0
            random.shuffle(self.indexes)
        return self.imgs[self.indexes[self.cur_index]], self.texts[self.indexes[self.cur_index]]

    def next_batch(self):
        while True:
            X_data = np.ones([self.batch_size, self.img_w, self.img_h, 1])
            Y_data = np.zeros([self.batch_size, self.max_text_len])
            input_length = np.ones((self.batch_size, 1)) * self.i_len
            label_length = np.zeros((self.batch_size, 1))
            for i in range(self.batch_size):
                img, text = self.next_sample()
                img = img.T
                img = np.expand_dims(img, -1)
                X_data[i] = img
                Y_data[i, :len(text)] = text_to_labels(text)
                label_length[i] = len(text)
            inputs = (X_data, Y_data, input_length, label_length)
            outputs = np.zeros([self.batch_size])
            yield (inputs, outputs)

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def create_model():
    input_data = layers.Input(name='the_input', shape=(img_w, img_h, 1), dtype='float32')
    iam_layers = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1')(input_data)
    iam_layers = layers.MaxPooling2D(pool_size=(2, 2), name='max1')(iam_layers)
    iam_layers = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2')(iam_layers)
    iam_layers = layers.MaxPooling2D(pool_size=(2, 2), name='max2')(iam_layers)
    iam_layers = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3')(iam_layers)
    iam_layers = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv4')(iam_layers)
    iam_layers = layers.MaxPooling2D(pool_size=(1, 2), name='max3')(iam_layers)
    iam_layers = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5')(iam_layers)
    iam_layers = layers.BatchNormalization()(iam_layers)
    iam_layers = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv6')(iam_layers)
    iam_layers = layers.BatchNormalization()(iam_layers)
    iam_layers = layers.MaxPooling2D(pool_size=(1, 2), name='max4')(iam_layers)
    iam_layers = layers.Conv2D(512, (2, 2), activation='relu', padding='same', name='conv7')(iam_layers)
    iam_layers = layers.Reshape(target_shape=((32, 2048)), name='reshape')(iam_layers)
    iam_layers = layers.Dense(64, activation='relu', name='dense1')(iam_layers)
    iam_layers = layers.Bidirectional(layers.LSTM(256, return_sequences=True, dropout=0.2))(iam_layers)
    iam_layers = layers.Bidirectional(layers.LSTM(256, return_sequences=True, dropout=0.2))(iam_layers)
    iam_outputs = layers.Dense(num_classes, activation='softmax', name='dense2')(iam_layers)

    labels = layers.Input(name='the_labels', shape=[max_text_len], dtype='float32')
    input_length = layers.Input(name='input_length', shape=[1], dtype='int64')
    label_length = layers.Input(name='label_length', shape=[1], dtype='int64')

    loss_out = layers.Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([iam_outputs, labels, input_length, label_length])
    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)
    return model

class TqdmCallback(Callback):
    def __init__(self, epochs, train_steps, valid_steps):
        super().__init__()
        self.epochs = epochs
        self.train_steps = train_steps
        self.valid_steps = valid_steps
        self.train_progbar = None
        self.valid_progbar = None
        self.epoch_count = 0

    def on_train_begin(self, logs=None):
        print("Training started")

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_count += 1
        print(f"\nEpoch {self.epoch_count}/{self.epochs}")
        self.train_progbar = tqdm(total=self.train_steps, desc="Training", leave=False)

    def on_train_batch_end(self, batch, logs=None):
        self.train_progbar.update(1)
        self.train_progbar.set_postfix(loss=f"{logs['loss']:.4f}")

    def on_epoch_end(self, epoch, logs=None):
        if self.train_progbar:
            self.train_progbar.close()
        self.valid_progbar = tqdm(total=self.valid_steps, desc="Validation", leave=False)

    def on_test_batch_end(self, batch, logs=None):
        if self.valid_progbar:  # Check if valid_progbar is not None
            self.valid_progbar.update(1)
            self.valid_progbar.set_postfix(val_loss=f"{logs['loss']:.4f}")

    def on_test_end(self, logs=None):
        if self.valid_progbar:  # Check if valid_progbar is not None
            self.valid_progbar.close()

def main(epochs: int = typer.Option(50, help="Number of epochs to train"),
         batch_size: int = typer.Option(64, help="Batch size for training"),
         learning_rate: float = typer.Option(0.001, help="Learning rate for optimizer")):

    train_files = get_paths_and_gts('resources/train_files.txt')
    valid_files = get_paths_and_gts('resources/valid_files.txt')

    train_data = TextImageGenerator(train_files, img_w, img_h, batch_size, input_length, max_text_len)
    train_data.build_data()

    validation_data = TextImageGenerator(valid_files, img_w, img_h, batch_size, input_length, max_text_len)
    validation_data.build_data()

    with tf.device('/GPU:0'):
        model = create_model()
        model.compile(loss={'ctc': lambda y_true, y_pred: y_pred},
                      optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
        model.summary()

        model_save_cb = ModelCheckpoint('best_model.keras', monitor='val_loss', verbose=0, save_best_only=True)
        early_stopping_cb = EarlyStopping(monitor='val_loss', patience=10, verbose=0)
        tqdm_callback = TqdmCallback(epochs, train_data.n // batch_size, validation_data.n // batch_size)

        with tf.device('/GPU:0'):
            history = model.fit(
                train_data.next_batch(),
                steps_per_epoch=train_data.n // batch_size,
                epochs=epochs,
                validation_data=validation_data.next_batch(),
                validation_steps=validation_data.n // batch_size,
                callbacks=[model_save_cb, early_stopping_cb, tqdm_callback],
                verbose=0  # Set to 0 as we're using our custom progress bar
            )

    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.savefig('model_loss.png')
    plt.close()

    print(f"Training completed. Model saved as 'best_model.h5'. Loss plot saved as 'model_loss.png'.")

if __name__ == '__main__':
    typer.run(main)

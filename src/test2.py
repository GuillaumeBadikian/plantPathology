import tensorflow as tf
import pandas as pd

SUB_PATH = "./data/sample_submission.csv"

sub = pd.read_csv(SUB_PATH)
AUTO = tf.data.experimental.AUTOTUNE
TEST_PATH = "./data/test.csv"
test_data = pd.read_csv(TEST_PATH)
GCS_DS_PATH = "./data/"

def format_path(st):
    return GCS_DS_PATH + '/images/' + st + '.jpg'
test_paths = test_data.image_id.apply(format_path).values

def decode_image(filename, label=None, image_size=(512, 512)):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, image_size)

    if label is None:
        return image
    else:
        return image, label

test_dataset = (
    tf.data.Dataset
        .from_tensor_slices(test_paths)
        .map(decode_image, num_parallel_calls=AUTO)
        .batch(8)
)



model = tf.keras.models.load_model('./model_softmax_epoch_40.sav')

probs_dnn = model.predict(test_dataset, verbose=1)
sub.loc[:, 'healthy':] = probs_dnn
sub.to_csv('submission_dnn.csv', index=False)
#print(sub.head()
print(sub.to_string(max_rows=500))

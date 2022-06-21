import pandas as pd
import tensorflow as tf
import Estimator

shuffle_num = 500

train_csv = "train" + str(Estimator.seq) + ".csv"
eval_csv = "test" + str(Estimator.seq) + ".csv"
# train_csv = "TotalSynTrain.csv"
# eval_csv = "TotalSynTest.csv"
data_path = "G:\\workspace_python\\DNN\\Estimator\\data\\"
TRAIN_URL = data_path + train_csv
TEST_URL = data_path + eval_csv
# print(TEST_URL, TRAIN_URL)

CSV_COLUMN_NAMES = ['Ag', 'As', 'Ba', 'Cd', 'Cu',
                    'Hf', 'Pb', 'Rb', 'S', 'Sb',
                    'Sr', 'Te', 'W', 'Zn', 'Zr',
                    'SiO2', 'Al2O3', 'TFe2O3', 'MgO', 'CaO',
                    'Na2O', 'K2O', 'Pth', 'Qpx', 'Qbg',
                    'Fault', 'Species']
SPECIES = ['Normal', 'Abnormal']


def maybe_download():
    train_path = tf.keras.utils.get_file(TRAIN_URL.split('/')[-1], TRAIN_URL)
    test_path = tf.keras.utils.get_file(TEST_URL.split('/')[-1], TEST_URL)

    return train_path, test_path


def load_data(y_name='Species'):
    """Returns the iris dataset as (train_x, train_y), (test_x, test_y)."""
    train_path, test_path = maybe_download()

    train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
    train_x, train_y = train, train.pop(y_name)

    test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
    test_x, test_y = test, test.pop(y_name)

    return (train_x, train_y), (test_x, test_y)


def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(shuffle_num).repeat().batch(batch_size)

    # Return the dataset.
    return dataset


def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features = dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset


# The remainder of this file contains a simple example of a csv parser,
#     implemented using the `Dataset` class.

# `tf.parse_csv` sets the types of the outputs to match the examples given in
#     the `record_defaults` argument.
CSV_TYPES = [[0.0], [0.0], [0.0], [0.0], [0.0],
             [0.0], [0.0], [0.0], [0.0], [0.0],
             [0.0], [0.0], [0.0], [0.0], [0.0],
             [0.0], [0.0], [0.0], [0.0], [0.0],
             [0.0], [0.0], [0.0], [0.0], [0.0],
             [0.0], [0]]


def _parse_line(line):
    # Decode the line into its fields
    fields = tf.decode_csv(line, record_defaults=CSV_TYPES)

    # Pack the result into a dictionary
    features = dict(zip(CSV_COLUMN_NAMES, fields))

    # Separate the label from the features
    label = features.pop('Species')

    return features, label


def csv_input_fn(csv_path, batch_size):
    # Create a dataset containing the text lines.
    dataset = tf.data.TextLineDataset(csv_path).skip(1)

    # Parse each line.
    dataset = dataset.map(_parse_line)

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(shuffle_num).repeat().batch(batch_size)

    # Return the dataset.
    return dataset

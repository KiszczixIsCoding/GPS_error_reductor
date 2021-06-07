# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from pandas import DataFrame
from tensorflow.python.keras.layers import Normalization
from sklearn import preprocessing

from dataset_loader import load_csv_file
import tensorflow as tf


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    training_input_data = DataFrame()
    training_output_data = DataFrame()

    testing_input_data = DataFrame()
    testing_output_data = DataFrame()

    for index in range(10, 225):
        data = load_csv_file('datasets/F10/f10_stat_' + str(index) + '.xlsx')
        training_input_data = training_input_data.append(data[0])
        training_output_data = training_output_data.append(data[1])


    data = load_csv_file('datasets/F10/f10_2p.xlsx')

    tf_data_z = tf.convert_to_tensor(data[0].astype('float32'))
    tf_data = tf.convert_to_tensor(training_input_data.astype('float32'))
    normalizer_input = preprocessing.MinMaxScaler(feature_range=(0, 1))

    normalized_data = normalizer_input.fit_transform(tf_data)
    tf_data_z_n = normalizer_input.transform(tf_data_z)

    tf_data_out = tf.convert_to_tensor(training_output_data.astype('float32'))
    normalizer_output = preprocessing.MinMaxScaler(feature_range=(0, 1))
    normalized_data_out = normalizer_output.fit_transform(tf_data_out)

    # w sumie to jest to nie używane - jak próbowałem to też wychodzą bzdury
    dataset = tf.data.Dataset.from_tensor_slices((normalized_data, normalized_data_out)).shuffle(
        buffer_size=1024).batch(64)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation=tf.nn.sigmoid, input_shape=(24,)),  # input shape required
        tf.keras.layers.Dense(64, activation=tf.nn.sigmoid),
        tf.keras.layers.Dense(28, activation=tf.nn.sigmoid),
        # tf.keras.layers.Dense(16, activation=tf.nn.relu),
        tf.keras.layers.Dense(2, activation=tf.nn.sigmoid)
    ])

    model.compile(optimizer='adam', loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(normalized_data, normalized_data_out, epochs=5)

    # still not working
    pr = model.predict(tf_data_z_n)
    print(normalizer_output.inverse_transform(pr))


# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from pandas import DataFrame
from tensorflow.python.keras.layers import Normalization
from sklearn import preprocessing

from dataset_loader import load_csv_file
import tensorflow as tf

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

    tf_data = tf.convert_to_tensor(training_input_data.astype('float32'))
    normalizer = preprocessing.MinMaxScaler(feature_range=(0, 1))
    # dodanie danych do transformatora znormalizowanego
    normalized_data = normalizer.fit_transform(tf_data)

    # normalized_tf_data = preprocessing.normalize(tf_data)

    tf_data_out = tf.convert_to_tensor(training_output_data.astype('float32'))
    print('A ', normalizer.fit_transform(tf_data_out))
    print('B ', normalizer.inverse_transform(normalizer.fit_transform(tf_data_out)))

    # normalized_tf_data_out = preprocessing.normalize(tf_data_out)
    normalized_data_out = normalizer.fit_transform(tf_data_out)
    # print(normalized_tf_data_out)

    dataset = tf.data.Dataset.from_tensor_slices((normalized_data, normalized_data_out)).shuffle(
        buffer_size=1024).batch(64)
    print(dataset)

    # tworzenie modelu składającego się z 3 warstw: dwie o prostokątnej funkcji aktywacji, jedna sigmoidalna
    # 25 kolumn tabeli tworzy model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation=tf.nn.relu, input_shape=(25,)),  # input shape required
        tf.keras.layers.Dense(64, activation=tf.nn.relu),
        tf.keras.layers.Dense(2, activation=tf.nn.sigmoid)
    ])

    # optymalizator, funkcja utraty oraz dokładność - dodatkowe ustawienia
    model.compile(optimizer='adam', loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # trenowanie modelu na danych, 5 epok
    model.fit(normalized_data, normalized_data_out, epochs=5)

    data = load_csv_file('datasets/F10/f10_1p.xlsx')

    tf_data_z = tf.convert_to_tensor(data[0].astype('float32'))
    normalizer_1 = preprocessing.MinMaxScaler(feature_range=(0, 1))
    tf_data_z_n = normalizer_1.fit_transform(tf_data_z)

    # not working
    pr = model.predict(tf_data_z_n)
    print(normalizer.inverse_transform(pr))
    # print(tf_data_out)

# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from pandas import DataFrame
from sklearn import preprocessing
from dataset_loader import load_csv_file
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

STAT_NUMBER = 225
DYN_NUMBER = 3

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    training_input_data = DataFrame()
    training_output_data = DataFrame()

    testing_input_data = DataFrame()
    testing_output_data = DataFrame()

    for index in range(1, 225):
        data = load_csv_file('datasets/F8/f8_stat_' + str(index) + '.xlsx')
        training_input_data = training_input_data.append(data[0])
        training_output_data = training_output_data.append(data[1])
        data = load_csv_file('datasets/F10/f10_stat_' + str(index) + '.xlsx')
        training_input_data = training_input_data.append(data[0])
        training_output_data = training_output_data.append(data[1])

    # for index in range(1, DYN_NUMBER):
    #     data = load_csv_file('datasets/F8/f8_' + str(index) + 'p.xlsx')
    #     testing_input_data = testing_input_data.append(data[0])
    #     testing_output_data = testing_output_data.append(data[1])
    #     data = load_csv_file('datasets/F10/f10_' + str(index) + 'p.xlsx')
    #     testing_input_data = testing_input_data.append(data[0])
    #     testing_output_data = testing_output_data.append(data[1])


    data = load_csv_file('datasets/F10/f10_1p.xlsx')

    tf_data = tf.convert_to_tensor((training_input_data.astype('float32')) / 10000)
    tf_data_out = tf.convert_to_tensor((training_output_data.astype('float32')) / 10000)
    tf_data_z_n = tf.convert_to_tensor((data[0].astype('float32')) / 10000)


    # tworzenie modelu składającego się z 3 warstw: dwie o prostokątnej funkcji aktywacji, jedna sigmoidalna
    # 24 kolumn tabeli tworzy model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation=tf.nn.relu, input_shape=(24,)),  # input shape required
        tf.keras.layers.Dense(64, activation=tf.nn.relu),
        tf.keras.layers.Dense(32, activation=tf.nn.relu),
        tf.keras.layers.Dense(16, activation=tf.nn.relu),
        tf.keras.layers.Dense(8, activation=tf.nn.relu),
        tf.keras.layers.Dense(2, activation=tf.nn.sigmoid),
    ])

    # optymalizator, funkcja utraty oraz dokładność - dodatkowe ustawienia
    model.compile(optimizer='adam', loss=tf.losses.MeanSquaredError(),
                  metrics=['accuracy'])

    # trenowanie modelu na danych, 150 epok
    model.fit(tf_data, tf_data_out, epochs=200, batch_size=512)
    pr = model.predict(tf_data_z_n)

    result = (pr * 10000)
    dtdt = DataFrame(result)

    mse = []
    for index in range(0, len(dtdt[0])):
        mean_xy = (dtdt[0][index] + dtdt[1][index]) / 2
        pow_x = pow(dtdt[0][index] - mean_xy, 2)
        pow_y = pow(dtdt[1][index] - mean_xy, 2)
        mse.append((pow_x + pow_y) / 2)


    print(result)
    print(data[1])
    print(np.sqrt(mse))

    plt.bar(np.arange(0, 2240), mse)
    plt.show()
# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from pandas import DataFrame
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


def load_training_dataset(room_num):
    training_input_data = DataFrame()
    training_output_data = DataFrame()
    if room_num == 8:
        for index in range(1, 226):
            data = load_csv_file('datasets/F8/f8_stat_' + str(index) + '.xlsx')
            training_input_data = training_input_data.append(data[0])
            training_output_data = training_output_data.append(data[1])
    if room_num == 10:
        for index in range(1, 226):
            data = load_csv_file('datasets/F10/f10_stat_' + str(index) + '.xlsx')
            training_input_data = training_input_data.append(data[0])
            training_output_data = training_output_data.append(data[1])
    return training_input_data, training_output_data


def load_testing_dataset(room_num):
    testing_input_data = DataFrame()
    testing_output_data = DataFrame()

    if room_num == 8:
        for index in range(1, 4):
            data = load_csv_file('datasets/F8/f8_' + str(index) + 'p.xlsx')
            testing_input_data = testing_input_data.append(data[0])
            testing_output_data = testing_output_data.append(data[1])
    if room_num == 10:
        for index in range(1, 4):
            data = load_csv_file('datasets/F10/f10_' + str(index) + 'p.xlsx')
            testing_input_data = testing_input_data.append(data[0])
            testing_output_data = testing_output_data.append(data[1])
    return testing_input_data, testing_output_data


def draw_plots(data_out):
    fig1, (ax1, ax2) = plt.subplots(2)
    ax1.bar(np.arange(0, len(data_out[0])), data_out[0])
    ax1.set_title('Pierwiastek błędu średniokwadratowego dla zbioru testowego')
    ax1.set(xlabel='Numer pomiaru', ylabel='RMSE')
    ax2.bar(np.arange(0, len(data_out[1])), data_out[1], color='tab:orange')
    ax2.set_title('Średnia arytmetyczna z wag danych wejściowych')
    ax2.set(xlabel='Index kolumny wejściowej', ylabel='Waga kolumny')
    plt.show()


def save_to_excel(filename, data):
    DataFrame(data).to_excel(filename)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    training_input, training_output = load_training_dataset(8)
    testing_input, testing_output = load_testing_dataset(8)

    training_input_tensor = tf.convert_to_tensor((training_input.astype('float32')) / 10000)
    training_output_tensor = tf.convert_to_tensor((training_output.astype('float32')) / 10000)
    testing_input_tensor = tf.convert_to_tensor((testing_input.astype('float32')) / 10000)
    testing_output_tensor = tf.convert_to_tensor((testing_output.astype('float32')) / 10000)

    # tworzenie modelu składającego się z 3 warstw: dwie o prostokątnej funkcji aktywacji, jedna sigmoidalna
    # 23 kolumn tabeli tworzy model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(23,)),  # input shape required
        tf.keras.layers.Dense(256, activation=tf.nn.relu),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(64, activation=tf.nn.relu),
        tf.keras.layers.Dense(32, activation=tf.nn.relu),
        tf.keras.layers.Dense(16, activation=tf.nn.relu),
        tf.keras.layers.Dense(8, activation=tf.nn.relu),
        tf.keras.layers.Dense(2, activation=tf.nn.sigmoid),
    ])

    # optymalizator, funkcja utraty oraz dokładność - dodatkowe ustawienia
    model.compile(optimizer='adam', loss=tf.losses.MeanSquaredError(), metrics=['accuracy'])

    # trenowanie modelu na danych, 25 epok
    model.fit(training_input_tensor, training_output_tensor, epochs=25)
    model.evaluate(testing_input_tensor, testing_output_tensor)
    weights = model.layers[0].get_weights()[0]
    pr = model.predict(testing_input_tensor)
    result = (pr * 10000)
    result_df = DataFrame(result)

    mse = []
    for index in range(0, len(result_df[0])):
        mse_xy = mean_squared_error([result_df[0][index], result_df[1][index]],
                                    [DataFrame(testing_output['reference__x']).iloc[0],
                                     DataFrame(testing_output['reference__y']).iloc[0]])
        mse.append(mse_xy)

    mse = np.sqrt(mse)
    weights_mean = np.mean(weights, axis=1)
    print(mse)
    print(weights_mean)

    draw_plots([mse, weights_mean])
    save_to_excel("rmse.xlsx", mse)

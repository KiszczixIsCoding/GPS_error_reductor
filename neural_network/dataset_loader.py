import pandas as pd
from pandas import DataFrame

from column_names import rejected_labels


def load_csv_file(filename):
    input_data = pd.read_excel(filename)

    output_data = DataFrame()
    output_data['reference__x'] = input_data['reference__x']
    output_data['reference__y'] = input_data['reference__y']

    for index, row in input_data.iterrows():
        if row['success'] is False:
            input_data = input_data.drop(index)
            output_data = output_data.drop(index)


    for column_name in rejected_labels:
        if column_name in input_data:
            input_data.pop(column_name)

    return input_data, DataFrame(output_data)

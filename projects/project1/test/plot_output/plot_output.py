import matplotlib.pyplot as plt
import numpy as np
import copy

default_args = {
    "batch_size":               1024,
    "num_kernels1":               16,
    "num_hidden":               1024,
    "regularization_factor":    1e-4,
    "dropout_keep_probability":  0.5,
    "learning_rate":           0.001,
    "kernel1_size":                5,
    "test_interval":             100,
    "num_batches":             10001,
    "seed":                      666,
    "pool":                        4,
}

test_values = {
    "batch_size":              [2048, 1024, 512, 256, 128, 64, 32, 16],
    "num_kernels1":            [64, 32, 16, 8, 4, 2, 1],
    "num_hidden":              [4096, 2048, 1024, 512, 256, 128, 64, 32, 16],
    #"regularization_factor":   [10, 1, 0, 1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-7],
    "dropout_keep_probability":[0.1, 0.25, 0.5, 0.9],
    "learning_rate":           [1e-2, 1e-3, 1e-4, 1e-6],
    "kernel1_size":            [11, 9, 7, 5, 3],
    "pool":                    [1, 2, 4],
}

with open("output.txt", "rb") as f:
    rows = f.read().decode("utf-8").strip().split("\n")
    datas = []
    for row in rows:
        data = eval(row)
        datas.append(data)

def find_data_with_args(args):
    for data in datas:
        if data['arguments'] == args:
            return data
    raise ValueError("Data does not exist")

# for each configurable parameter
for key, values in test_values.items():
    plt.figure()
    plt.title(key)
    handles = []
    # try all parameters
    for value in values:
        # copy default arguments
        args = copy.copy(default_args)
        # change one value
        args[key] = value

        if key == 'learning_rate' and value == 1e-6:
            continue

        data = find_data_with_args(args)
        accuracies = data['accuracies']
        accuracies = data['accuracies'][5:]
        handle, = plt.plot(accuracies, label=str(value))
        handles.append(handle)
    plt.legend(handles=handles, loc=4)
    plt.tight_layout()
    #plt.show()
    plt.savefig(key + '.pdf', bbox_inches='tight')

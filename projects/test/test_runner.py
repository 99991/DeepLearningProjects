#!/usr/bin/env python

import copy
import subprocess

default_args = {
	"batch_size":                 64,
	"num_kernels1":                4,
	"num_hidden":                100,
	"regularization_factor":    1e-4,
	"dropout_keep_probability":  0.5,
	"learning_rate":           0.001,
	"kernel1_size":                3,
	"test_interval":             100,
	"num_batches":               1,
	"seed":                      666,
        "pool":                        4,
}

test_values = {
    "batch_size":              [2048, 1024, 512, 256, 128, 64, 32, 16],
    "num_kernels1":            [32, 16, 8, 4, 2, 1],
    "num_hidden":              [2048, 1024, 512, 256, 128, 64, 32, 16],
    "regularization_factor":   [10, 1, 0, 1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-7],
    "dropout_keep_probability":[0.0, 0.1, 0.25, 0.5, 0.9, 0.99],
    "learning_rate":           [1e-2, 1e-3, 1e-4, 1e-6],
    "kernel1_size":            [7, 5, 3],
    "pool":                    [1, 2, 4],
}

# for each configurable parameter
for key, values in test_values.items():
    # try all parameters
    for value in values:
        # copy default arguments
        args = copy.copy(default_args)
        # change one value
        args[key] = value

        # build command to execute test.py with arguments
        command = ['python', 'test.py']
        for k, v in args.items():
            command.append(k)
            command.append(str(v))

        print(" ".join(command))

        # try to execute command
        try:
            output = subprocess.check_output(command)
            print(output)
        except Exception as e:
            print(e)

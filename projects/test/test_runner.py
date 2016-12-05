#!/usr/bin/env python

import copy
import subprocess

default_args = {
	"batch_size":                512,
	"num_kernels1":               16,
	"num_hidden":                512,
	"regularization_factor":    1e-4,
	"dropout_keep_probability": 0.25,
	"learning_rate":           0.001,
	"kernel1_size":                5,
	"test_interval":             100,
	"num_batches":              2001,
	"seed":                      666,
    "pool":                        4,
}

test_values = {
    "kernel1_size":                    [5]
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

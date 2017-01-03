from nn import run_nn
import sys, time
import json

def to_numeric(s):
    try:
        return int(s)
    except ValueError:
        return float(s)

keys = sys.argv[1::2]
vals = sys.argv[2::2]
args = [(key, to_numeric(val)) for key, val in zip(keys, vals)]
args = dict(args)

start_time = time.clock()

accuracies, losses, _ = run_nn(**args)

delta_time = time.clock() - start_time

with open('output.txt', 'a') as f:
    data = {
        "date":time.strftime("%X %x %Z"),
        "running_time":delta_time,
        "arguments":args,
        "accuracies":accuracies,
        "losses":losses,
    }
    s = str(data)
    print(s)
    f.write(s)
    f.write("\n")


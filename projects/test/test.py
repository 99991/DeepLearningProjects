from nn import run_nn
import sys, time

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
    f.write("Finished at: %s\n"%time.strftime("%X %x %Z"))
    f.write("Running time: %f seconds\n"%delta_time)
    f.write("Arguments:\n")
    f.write("\n".join("%s %s"%kv for kv in zip(keys, vals)))
    f.write("\n")
    for accuracy, loss in zip(accuracies, losses):
        f.write("accuracy %f, loss %f\n"%(accuracy, loss))
    f.write("\n")


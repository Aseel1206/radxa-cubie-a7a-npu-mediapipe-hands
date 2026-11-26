import sys, collections, numpy as np
import tensorflow as tf
p = sys.argv[1]
it = tf.lite.Interpreter(model_path=p); it.allocate_tensors()
for d in it.get_input_details(): print("IN ", d["name"], d["shape"], d["dtype"].__name__, d["quantization"])
for d in it.get_output_details(): print("OUT", d["name"], d["shape"], d["dtype"].__name__)
ops = collections.Counter(o["op_name"] for o in it._get_ops_details())
print(dict(ops))

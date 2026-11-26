# runs in container: float TFLite reference + writes raw input bins
import sys,numpy as np,cv2,glob,os
sys.path.insert(0,'/workspace/mp/scripts'); from lb import letterbox
import tensorflow as tf
it=tf.lite.Interpreter(model_path='/workspace/mp/palm_det/hand_detector.tflite'); it.allocate_tensors()
i=it.get_input_details()[0]; o=it.get_output_details()
os.makedirs('/workspace/mp/palm_test/out',exist_ok=True)
for f in sorted(glob.glob('/workspace/mp/palm_test/t_*.png')):
    n=os.path.basename(f)[:-4]; im,_=letterbox(cv2.imread(f)); rgb=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
    rgb.tofile(f'/workspace/mp/palm_test/out/{n}.in.bin')
    it.set_tensor(i['index'],(rgb.astype(np.float32)/255)[None]); it.invoke()
    for k,d in enumerate(o): np.save(f'/workspace/mp/palm_test/out/{n}.ref{k}.npy',it.get_tensor(d['index']))
print('ok')

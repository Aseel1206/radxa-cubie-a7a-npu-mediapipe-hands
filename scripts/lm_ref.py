import glob,os,numpy as np,cv2,tensorflow as tf
it=tf.lite.Interpreter(model_path='/workspace/mp/hand_lm/hand_landmarks_detector.tflite'); it.allocate_tensors()
i=it.get_input_details()[0]; o=it.get_output_details(); D='/workspace/mp/test224/out'; os.makedirs(D,exist_ok=True)
for f in sorted(glob.glob('/workspace/mp/test224/h_*.png')):
    n=os.path.basename(f)[:-4]; rgb=cv2.cvtColor(cv2.imread(f),cv2.COLOR_BGR2RGB); rgb.tofile(f'{D}/{n}.in.bin')
    it.set_tensor(i['index'],(rgb.astype(np.float32)/255)[None]); it.invoke()
    np.save(f'{D}/{n}.ref.npy',np.concatenate([it.get_tensor(d['index']).ravel() for d in o[:3]]).astype(np.float32)); np.save(f'{D}/{n}.reflm.npy',it.get_tensor(o[0]['index']).ravel())

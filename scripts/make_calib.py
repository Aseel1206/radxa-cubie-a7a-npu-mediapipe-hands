"""Build calibration + test sets for the two hand models from a folder of hand photos (run inside the container, needs tensorflow).

  python3 /workspace/mp/scripts/make_calib.py /workspace/mp/hand_photos 8

The first N images become the test set, the rest are calibration data:
  calib192h/  letterboxed 192x192 images for the palm detector   (+ dataset.txt)
  calib224/   rotated 224x224 hand crops for the landmark model   (+ dataset.txt, crops come from the float palm detector)
  palm_test/, test224/   same thing for the test images
Calibration images should look like what you will feed at run time. I used ~40-47 photos of people holding up a hand.
"""
import sys, glob, os, numpy as np, cv2
sys.path.insert(0, '/workspace/mp/scripts')
from lb import letterbox
from palm_decode import decode
from palm_roi import roi_from_det, crop
import tensorflow as tf

src = sys.argv[1]; ntest = int(sys.argv[2]) if len(sys.argv) > 2 else 8
R = '/workspace/mp/'
files = sorted(glob.glob(src + '/*.jp*g') + glob.glob(src + '/*.png'))
test, calib = files[:ntest], files[ntest:]
it = tf.lite.Interpreter(model_path=R + 'palm_det/hand_detector.tflite'); it.allocate_tensors()
inp = it.get_input_details()[0]; outs = it.get_output_details()

def palm(im):
    lbimg, par = letterbox(im)
    it.set_tensor(inp['index'], (cv2.cvtColor(lbimg, cv2.COLOR_BGR2RGB).astype(np.float32) / 255)[None]); it.invoke()
    dets, _ = decode(it.get_tensor(outs[0]['index']), it.get_tensor(outs[1]['index']))
    return lbimg, par, dets

for name, fs, d192, d224 in (('calib', calib, 'calib192h', 'calib224'), ('test', test, 'palm_test', 'test224')):
    for d in (d192, d224): os.makedirs(R + d, exist_ok=True)
    n = 0
    for i, f in enumerate(fs):
        im = cv2.imread(f); lbimg, (r, l, t), dets = palm(im)
        cv2.imwrite(R + f"{d192}/{'c' if name=='calib' else 't'}_{i:03d}.png", lbimg if name == 'calib' else im)
        for s, b, kp in dets:
            cx, cy, sz, rot = roi_from_det(b, kp, r, l, t); c, _ = crop(im, cx, cy, sz, rot)
            cv2.imwrite(R + f'{d224}/h_{n:03d}.png', c); n += 1
    if name == 'calib':
        open(R + 'calib192h/dataset.txt', 'w').write(''.join(f'/workspace/mp/calib192h/c_{i:03d}.png\n' for i in range(len(fs))))
        open(R + 'calib224/dataset.txt', 'w').write(''.join(f'/workspace/mp/calib224/h_{k:03d}.png\n' for k in range(n)))
    print(name, len(fs), 'images ->', n, 'hand crops')

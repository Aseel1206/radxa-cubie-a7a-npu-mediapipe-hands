#!/usr/bin/env python3
"""Hand tracking on a still image with both MediaPipe models running on the Radxa NPU.

palm detector (.nb)  ->  decode anchors + NMS  ->  rotated hand ROI  ->  landmark model (.nb)  ->  draw

Run it ON THE BOARD. It shells out to the nb_bench tool for each network, so it is slow (model load every call)
and only meant to show the data flow and let you check results. For real use, keep the sessions open in C++.

  python3 hands_demo.py photo.jpg out.png --palm palm_pcq.nb --lm landmark_pcq.nb --bench ~/nb_bench/nb_bench
Needs: numpy, opencv-python (cv2).
"""
import argparse, os, subprocess, sys, tempfile
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lb import letterbox
from palm_decode import decode
from palm_roi import roi_from_det, crop

EDGES = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]

def run_nb(bench, nb, rgb_bytes, tmp, nouts):
    inp = os.path.join(tmp, 'in.bin'); out = os.path.join(tmp, 'out.bin')
    open(inp, 'wb').write(rgb_bytes)
    env = dict(os.environ, LD_LIBRARY_PATH=os.path.expanduser('~/awnn_inference/install/lib'))
    subprocess.run([bench, '-m', nb, '-n', '1', '-i0', inp, '-o', out], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return [np.fromfile(out if k == 0 else f'{out}.{k}', np.float32) for k in range(nouts)]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('image'); ap.add_argument('out')
    ap.add_argument('--palm', required=True); ap.add_argument('--lm', required=True); ap.add_argument('--bench', default=os.path.expanduser('~/nb_bench/nb_bench'))
    a = ap.parse_args(); im = cv2.imread(a.image); assert im is not None, 'cannot read image'
    lb, (r, l, t) = letterbox(im)                      # 192x192, black padding, aspect ratio kept
    with tempfile.TemporaryDirectory() as tmp:
        reg, sc = run_nb(a.bench, a.palm, cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).tobytes(), tmp, 2)
        dets, _ = decode(reg, sc)
        print(f'{len(dets)} hand(s) found')
        for k, (score, box, kp) in enumerate(dets):
            cx, cy, size, rot = roi_from_det(box, kp, r, l, t)
            patch, M = crop(im, cx, cy, size, rot)       # 224x224 rotated crop (M maps crop px -> image px)
            lms, flag, hand = run_nb(a.bench, a.lm, cv2.cvtColor(patch, cv2.COLOR_BGR2RGB).tobytes(), tmp, 3)
            pts = (M @ np.c_[lms.reshape(21, 3)[:, :2], np.ones(21)].T).T
            col = [(0, 0, 255), (255, 128, 0)][k % 2]; lw = max(1, round(im.shape[1] / 400))
            print(f'  hand {k}: palm score {score:.2f}, landmark presence {flag[0]:.2f}')
            for p, q in EDGES: cv2.line(im, tuple(pts[p].astype(int)), tuple(pts[q].astype(int)), col, lw + 1)
            for q in pts: cv2.circle(im, tuple(q.astype(int)), lw + 2, (255, 255, 255), -1)
            box_px = (M @ np.array([[0, 0, 1], [224, 0, 1], [224, 224, 1], [0, 224, 1]]).T).T.astype(np.int32)
            cv2.polylines(im, [box_px], True, (0, 255, 255), lw)
    cv2.imwrite(a.out, im); print('wrote', a.out)

if __name__ == '__main__': main()

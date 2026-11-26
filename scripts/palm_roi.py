import numpy as np, cv2, math
def hand_roi(kp, r, l, t, W, H, scale=2.6, shift_y=-0.5):
    """kp: (7,2) normalised in 192 letterbox space -> rotated square ROI in original px: (cx,cy,size,rot)"""
    px=(kp[:,0]*192-l)/r; py=(kp[:,1]*192-t)/r
    x0,y0=px[0],py[0]; x2,y2=px[2],py[2]
    rot=math.pi/2-math.atan2(-(y2-y0),x2-x0); rot=rot-2*math.pi*math.floor((rot+math.pi)/(2*math.pi))
    # box from all keypoints' bbox is not used by MediaPipe; it uses the detection box -> pass in via roi_from_box
    return rot
def roi_from_det(box,kp,r,l,t,scale=2.6,shift_y=-0.5):
    rot=hand_roi(kp,r,l,t,0,0)
    cx=((box[0]+box[2])/2*192-l)/r; cy=((box[1]+box[3])/2*192-t)/r
    w=(box[2]-box[0])*192/r; h=(box[3]-box[1])*192/r
    # shift along rotated y axis
    cx+= -h*shift_y*math.sin(rot)*-1*-1 if False else 0
    dx=-(h*shift_y)*math.sin(rot)*-1  # placeholder replaced below
    # MediaPipe: x_shift=(w*sx*cos - h*sy*sin); y_shift=(w*sx*sin + h*sy*cos), sx=0
    cx+= -h*shift_y*math.sin(rot); cy+= h*shift_y*math.cos(rot)
    size=max(w,h)*scale
    return cx,cy,size,rot
def crop(img,cx,cy,size,rot,out=224):
    c,s=math.cos(rot),math.sin(rot); k=size/out
    # output (u,v) -> source: center + R*(u-out/2, v-out/2)*k
    M=np.array([[c*k,-s*k,cx-(c*k*out/2-s*k*out/2)],[s*k,c*k,cy-(s*k*out/2+c*k*out/2)]],np.float32)
    return cv2.warpAffine(img,M,(out,out),flags=cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0),M

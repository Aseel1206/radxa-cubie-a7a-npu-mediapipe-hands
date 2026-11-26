import cv2, numpy as np
def letterbox(im, s=192):
    h,w=im.shape[:2]; r=s/max(h,w); nh,nw=round(h*r),round(w*r)
    o=np.zeros((s,s,3),np.uint8); t,l=(s-nh)//2,(s-nw)//2
    o[t:t+nh,l:l+nw]=cv2.resize(im,(nw,nh),interpolation=cv2.INTER_AREA)
    return o,(r,l,t)

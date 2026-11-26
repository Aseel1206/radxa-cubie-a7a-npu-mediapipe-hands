import numpy as np
def anchors():
    a=[]
    for g,n in ((24,2),(12,6)):
        for y in range(g):
            for x in range(g):
                for _ in range(n): a.append(((x+.5)/g,(y+.5)/g))
    return np.array(a,np.float32)
def decode(reg,sc,thr=0.5,nms=0.3):
    A=anchors(); p=1/(1+np.exp(-np.clip(sc.reshape(-1),-80,80)))
    r=reg.reshape(2016,18)/192.0
    cx=r[:,0]+A[:,0]; cy=r[:,1]+A[:,1]; w=r[:,2]; h=r[:,3]
    kp=r[:,4:].reshape(2016,7,2)+A[:,None,:]
    idx=np.argsort(-p); keep=[]
    box=np.stack([cx-w/2,cy-h/2,cx+w/2,cy+h/2],1)
    for i in idx:
        if p[i]<thr: break
        ok=True
        for j in keep:
            xx1=max(box[i,0],box[j,0]);yy1=max(box[i,1],box[j,1]);xx2=min(box[i,2],box[j,2]);yy2=min(box[i,3],box[j,3])
            inter=max(0,xx2-xx1)*max(0,yy2-yy1); u=w[i]*h[i]+w[j]*h[j]-inter
            if inter/u>nms: ok=False;break
        if ok: keep.append(i)
    return [(float(p[i]),box[i],kp[i]) for i in keep], p

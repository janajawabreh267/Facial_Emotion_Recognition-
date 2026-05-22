#!/usr/bin/env python3
import argparse, time, cv2, numpy as np, torch
import torchvision.transforms as T
from PIL import Image

EMOTIONS=['angry','disgust','fear','happy','neutral','sad','surprise']
MEAN=[0.5123,0.5123,0.5123]; STD=[0.2107,0.2107,0.2107]; SZ=224
PALETTE={'angry':(78,75,226),'disgust':(34,153,99),'fear':(221,119,127),
         'happy':(39,159,239),'neutral':(128,135,136),'sad':(221,138,55),'surprise':(126,83,212)}

preprocess=T.Compose([T.Resize((SZ,SZ),interpolation=T.InterpolationMode.BICUBIC),
                       T.ToTensor(),T.Normalize(mean=MEAN,std=STD)])

def predict(model,face_bgr,dev):
    pil=Image.fromarray(cv2.cvtColor(face_bgr,cv2.COLOR_BGR2RGB))
    x=preprocess(pil).unsqueeze(0).to(dev)
    with torch.no_grad(): _,probs=model(x)
    return sorted(zip(EMOTIONS,probs.squeeze(0).cpu().tolist()),key=lambda x:-x[1])

def draw(frame,x,y,w,h,preds,fps):
    em,cf=preds[0]; col=PALETTE[em]
    cv2.rectangle(frame,(x,y),(x+w,y+h),col,2)
    cv2.rectangle(frame,(x,y-30),(x+w,y),col,-1)
    cv2.putText(frame,f'{em.upper()} {cf*100:.0f}%',(x+4,y-8),cv2.FONT_HERSHEY_SIMPLEX,0.65,(255,255,255),2)
    bx=x+w+10
    for r,(e,c) in enumerate(preds[:3]):
        by=y+r*26; fi=int(c*120)
        cv2.rectangle(frame,(bx,by),(bx+120,by+20),(50,50,50),-1)
        cv2.rectangle(frame,(bx,by),(bx+fi,by+20),PALETTE[e],-1)
        cv2.putText(frame,f'{e}:{c:.2f}',(bx+2,by+15),cv2.FONT_HERSHEY_SIMPLEX,0.45,(255,255,255),1)
    cv2.putText(frame,f'FPS:{fps:.0f}',(10,28),cv2.FONT_HERSHEY_SIMPLEX,0.8,(200,200,200),2)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',default=0)
    p.add_argument('--model',default='exports/fer_model_torchscript.pt')
    args=p.parse_args()
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=torch.jit.load(args.model,map_location=dev); model.eval()
    cascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    src=int(args.source) if str(args.source).isdigit() else args.source
    cap=cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
    print(f'Running on {dev}. Press q to quit.')
    t_prev=time.perf_counter()
    while True:
        ret,frame=cap.read()
        if not ret: break
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces=cascade.detectMultiScale(gray,1.2,5,minSize=(60,60))
        for (x,y,w,h) in faces:
            pad=int(0.1*min(w,h))
            x1,y1=max(x-pad,0),max(y-pad,0)
            x2,y2=min(x+w+pad,frame.shape[1]),min(y+h+pad,frame.shape[0])
            preds=predict(model,frame[y1:y2,x1:x2],dev)
            draw(frame,x,y,w,h,preds,0)
        fps=1.0/max(time.perf_counter()-t_prev,1e-6); t_prev=time.perf_counter()
        cv2.putText(frame,f'FPS:{fps:.0f}',(10,28),cv2.FONT_HERSHEY_SIMPLEX,0.8,(200,200,200),2)
        cv2.imshow('FER Demo - press q to quit',frame)
        if cv2.waitKey(1)&0xFF==ord('q'): break
    cap.release(); cv2.destroyAllWindows()

if __name__=='__main__': main()
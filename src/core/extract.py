import cv2, torch, sys
from pathlib import Path
from facenet_pytorch import MTCNN

device= "cuda" if torch.cuda.is_available() else "cpu"
mtcnn= MTCNN(keep_all=False, device=device, post_process=False)

def extract_faces(video_path, out_dir, every_n=5, size=224, margin=0.3):
    out_dir= Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cap= cv2.VideoCapture(str(video_path))
    idx=saved=0
    while True:
        ok,frame= cap.read()
        if not ok: break
        if idx%every_n==0:
            rgb= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes, probs= mtcnn.detect(rgb)
            if boxes is not None and probs[0]>0.95:
                x1,y1,x2,y2= boxes[0]
                w,h= x2-x1, y2-y1
                x1,y1= max(0, int(x1-margin*w)), max(0, int(y1-margin*h))
                x2,y2= int(x2+margin*w), int(y2+margin*h)
                face= cv2.resize(frame[y1:y2, x1:x2], (size,size))
                cv2.imwrite(str(out_dir / f"{idx:06d}.jpg"), face)
                saved+=1
        idx+=1
    cap.release()
    return saved

if __name__=="__main__":
    video= Path(sys.argv[1])
    n= extract_faces(video, Path("data/faces") / video.stem)
    print(f"Saved {n} faces")
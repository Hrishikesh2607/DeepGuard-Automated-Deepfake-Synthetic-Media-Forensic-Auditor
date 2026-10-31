import random
from pathlib import Path
from tqdm import tqdm
from src.core.extract import extract_faces

RAW= Path("data/raw")
OUT= Path("data/faces")
N_REAL=300
random.seed(42)

def list_videos():
    real= sorted((RAW/"original_sequences/youtube/c23/videos").glob("*.mp4"))
    items= [(p,0,"real") for p in random.sample(real, min(N_REAL, len(real)))]
    methods= ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
    per_method= N_REAL//len(methods)
    for m in methods:
        vids= sorted((RAW/f"manipulated_sequences/{m}/c23/videos").glob("*.mp4"))
        items += [(p,1,m) for p in random.sample(vids, min(per_method, len(vids)))]
    return items

if __name__=="__main__":
    rows=[]
    for path,label,method in tqdm(list_videos()):
        vid= f"{method}_{path.stem}"
        out_dir= OUT/vid
        if out_dir.exists() and any(out_dir.iterdir()):
            n= len(list(out_dir.glob("*.jpg")))
        else:
            try:
                n= extract_faces(path, out_dir)
            except Exception as e:
                print(f"FAILED {path}: {e}"); continue
        rows.append((vid, path.stem, label,method, n, str(out_dir)))

    import pandas as pd
    pd.DataFrame(rows, columns=["video_id","stem","label","method","n_faces","faces_dir"]) \
      .to_csv("data/manifest_raw.csv", index=False)
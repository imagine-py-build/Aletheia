import subprocess,sys
for script in ['training/train_image.py','training/train_audio.py','training/train_video.py','training/train_nlp.py']:
 r=subprocess.run([sys.executable,script]);
 if r.returncode: raise SystemExit(r.returncode)

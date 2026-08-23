from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForSequenceClassification
class AbuseClassifier:
 def __init__(self,path):
  if not Path(path).exists(): raise FileNotFoundError(f'NLP checkpoint not found: {path}')
  self.tok=AutoTokenizer.from_pretrained(path); self.m=AutoModelForSequenceClassification.from_pretrained(path); self.m.eval()
 def predict(self,text):
  x=self.tok(text,return_tensors='pt',truncation=True,max_length=256)
  with torch.no_grad(): p=torch.sigmoid(self.m(**x).logits)[0]
  return {'labels':{self.m.config.id2label[i]:float(v) for i,v in enumerate(p)},'model_version':'aletheia-xlm-r-abuse-v1'}

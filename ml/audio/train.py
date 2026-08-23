from pathlib import Path
import yaml, torch, torchaudio
from torch.utils.data import Dataset,DataLoader
from .model import SpectrogramCNN
class AudioFolder(Dataset):
    def __init__(self,root):
        self.items=[(p,0 if p.parent.name.lower() in ('human','real','bonafide') else 1) for p in Path(root).rglob('*') if p.suffix.lower() in ('.wav','.flac','.mp3')]
    def __len__(self): return len(self.items)
    def __getitem__(self,i):
        p,y=self.items[i]; w,s=torchaudio.load(p); w=w.mean(0,keepdim=True)
        if s!=16000:w=torchaudio.functional.resample(w,s,16000)
        w=w[:,:16000*4]; w=torch.nn.functional.pad(w,(0,max(0,16000*4-w.shape[1])))
        x=torchaudio.transforms.MelSpectrogram(sample_rate=16000,n_mels=96)(w).log1p(); return x,y

def main(config='configs/audio_train.yaml'):
    c=yaml.safe_load(open(config)); torch.manual_seed(c['seed']); ds=AudioFolder(c['data_dir']); loader=DataLoader(ds,batch_size=c['batch_size'],shuffle=True); m=SpectrogramCNN(); opt=torch.optim.AdamW(m.parameters(),lr=c['lr']); ce=torch.nn.CrossEntropyLoss(); Path(c['output_dir']).mkdir(parents=True,exist_ok=True)
    for e in range(c['epochs']):
        m.train(); total=0
        for x,y in loader: opt.zero_grad(); l=ce(m(x),y); l.backward(); opt.step(); total+=l.item()
        print({'epoch':e+1,'loss':total/max(1,len(loader))})
    torch.save(m.state_dict(),Path(c['output_dir'])/'best.pt')
if __name__=='__main__':main()

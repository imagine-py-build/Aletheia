from pathlib import Path
import yaml, torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

def main(config='configs/image_train.yaml'):
    c=yaml.safe_load(open(config)); torch.manual_seed(c['seed']); root=Path(c['data_dir'])
    tf=transforms.Compose([transforms.Resize((c['image_size'],c['image_size'])),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    ds=datasets.ImageFolder(root,transform=tf); n=len(ds); a=int(n*(1-c['val_fraction']-c['test_fraction'])); b=int(n*c['val_fraction']); tr,va,te=random_split(ds,[a,b,n-a-b],generator=torch.Generator().manual_seed(c['seed']))
    train=DataLoader(tr,batch_size=c['batch_size'],shuffle=True,num_workers=c['num_workers']); val=DataLoader(va,batch_size=c['batch_size'])
    m=efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT); m.classifier[1]=nn.Linear(m.classifier[1].in_features,2); opt=torch.optim.AdamW(m.parameters(),lr=c['lr']); loss=nn.CrossEntropyLoss(); best=0
    Path(c['output_dir']).mkdir(parents=True,exist_ok=True)
    for epoch in range(c['epochs']):
        m.train()
        for x,y in train: opt.zero_grad(); loss(m(x),y).backward(); opt.step()
        m.eval(); correct=total=0
        with torch.no_grad():
            for x,y in val: correct+=(m(x).argmax(1)==y).sum().item(); total+=len(y)
        acc=correct/max(total,1); print({'epoch':epoch+1,'val_accuracy':acc})
        if acc>best: best=acc; torch.save(m.state_dict(),Path(c['output_dir'])/'best.pt')
if __name__=='__main__': main()

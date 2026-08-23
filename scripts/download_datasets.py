"""Dataset acquisition guide.
Restricted datasets are intentionally not auto-downloaded or redistributed.
Use each dataset's official portal, then pass the local source directory to prepare_datasets.py.
"""
SOURCES={
 'FaceForensics++':'https://github.com/ondyari/FaceForensics',
 'Celeb-DF':'https://github.com/yuezunli/celeb-deepfakeforensics',
 'ASVspoof':'https://www.asvspoof.org/',
 'DFDC':'https://www.kaggle.com/c/deepfake-detection-challenge/data'
}
if __name__=='__main__':
 for k,v in SOURCES.items(): print(f'{k}: {v}')

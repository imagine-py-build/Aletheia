from fastapi import FastAPI, Depends, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pathlib import Path
import json
from backend.app.db.session import Base,engine,get_db
from backend.app.models.entities import Incident,Evidence,AnalysisResult,ChainOfCustody,Entity,Report
from backend.app.schemas.api import IncidentCreate,IncidentOut,AnalysisOut
from backend.app.services.evidence import save_upload
from backend.app.forensics.metadata import extract_metadata
from backend.app.forensics.ocr import extract_text
from backend.app.forensics.entities import extract_entities
from backend.app.forensics.risk import assess
from backend.app.core.config import settings

Base.metadata.create_all(engine)
app=FastAPI(title='Aletheia Forensics API',version='1.0.0')
app.add_middleware(CORSMiddleware,allow_origins=['*'],allow_methods=['*'],allow_headers=['*'])

def incident_or_404(db,id):
    obj=db.get(Incident,id)
    if not obj: raise HTTPException(404,'Incident not found')
    return obj

def evidence_or_404(db,id):
    obj=db.get(Evidence,id)
    if not obj: raise HTTPException(404,'Evidence not found')
    return obj

@app.get('/health')
def health(): return {'status':'ok','service':'aletheia'}

@app.post('/incidents',response_model=IncidentOut)
def create_incident(body:IncidentCreate,db:Session=Depends(get_db)):
    obj=Incident(title=body.title,description=body.description); db.add(obj); db.commit(); db.refresh(obj); return obj

@app.get('/incidents/{id}',response_model=IncidentOut)
def get_incident(id:str,db:Session=Depends(get_db)): return incident_or_404(db,id)

@app.post('/evidence/upload')
def upload_evidence(file:UploadFile=File(...),incident_id:str|None=None,db:Session=Depends(get_db)):
    # Media can be analysed immediately. If the user has not created a case,
    # Aletheia creates an internal workspace incident solely for evidence
    # integrity/risk/report persistence; the user is never forced to create one.
    standalone = incident_id is None
    if incident_id is None:
        obj=Incident(title=f'Standalone Media Analysis — {file.filename}',description='Auto-created workspace container for standalone media analysis.')
        db.add(obj); db.flush(); incident_id=obj.id
    else:
        incident_or_404(db,incident_id)
    data=file.file.read()
    path,h,mime=save_upload(data,file.filename,incident_id); meta=extract_metadata(path)
    e=Evidence(incident_id=incident_id,original_filename=file.filename,storage_path=path,sha256=h,mime_type=mime,file_size=len(data),metadata_json=meta); db.add(e); db.flush()
    db.add(ChainOfCustody(evidence_id=e.id,action='UPLOADED',actor='system',details={'sha256':h,'filename':file.filename,'standalone':standalone})); db.commit(); db.refresh(e)
    return {'evidence_id':e.id,'incident_id':incident_id,'standalone':standalone,'sha256':h,'mime_type':mime,'metadata':meta}

@app.get('/evidence/{id}')
def get_evidence(id:str,db:Session=Depends(get_db)):
    e=evidence_or_404(db,id); return {'id':e.id,'incident_id':e.incident_id,'filename':e.original_filename,'sha256':e.sha256,'mime_type':e.mime_type,'file_size':e.file_size,'metadata':e.metadata_json}

@app.post('/analysis/metadata',response_model=AnalysisOut)
def analyze_metadata(evidence_id:str,db:Session=Depends(get_db)):
    e=evidence_or_404(db,evidence_id); result={'metadata':extract_metadata(e.storage_path),'evidence_sha256':e.sha256}; a=AnalysisResult(evidence_id=e.id,analysis_type='metadata',result_json=result,model_version='forensic-exif-v1'); db.add(a); db.commit(); db.refresh(a); return a

@app.post('/analysis/ocr',response_model=AnalysisOut)
def analyze_ocr(evidence_id:str,db:Session=Depends(get_db)):
    e=evidence_or_404(db,evidence_id); result=extract_text(e.storage_path); result['entities']=extract_entities(result.get('text','')); a=AnalysisResult(evidence_id=e.id,analysis_type='ocr',result_json=result,model_version=result.get('engine')); db.add(a)
    for x in result['entities']: db.add(Entity(incident_id=e.incident_id,entity_type=x['type'],value=x['value']))
    db.commit(); db.refresh(a); return a

@app.post('/analysis/text',response_model=AnalysisOut)
def analyze_text(evidence_id:str, text:str, db:Session=Depends(get_db)):
    e=evidence_or_404(db,evidence_id)
    from ml.nlp.infer import AbuseClassifier
    try: out=AbuseClassifier(str(Path(settings.model_dir)/'nlp/best')).predict(text)
    except FileNotFoundError as ex: raise HTTPException(503,str(ex))
    out['entities']=extract_entities(text); a=AnalysisResult(evidence_id=e.id,analysis_type='text',result_json=out,model_version=out['model_version']); db.add(a); db.commit(); db.refresh(a); return a

@app.post('/analysis/image',response_model=AnalysisOut)
def analyze_image(evidence_id:str,db:Session=Depends(get_db)):
    e=evidence_or_404(db,evidence_id)
    if not e.mime_type.startswith('image/'):
        raise HTTPException(400,'Deepfake image detection requires an image file. Use OCR/metadata for documents.')
    from ml.image.infer import ImageDetector
    try: out=ImageDetector().predict(e.storage_path)
    except Exception as ex: raise HTTPException(503,f'Image detector unavailable: {ex}')
    a=AnalysisResult(evidence_id=e.id,analysis_type='image',result_json=out,model_version=out['model_version']); db.add(a); db.commit(); db.refresh(a); return a

@app.post('/analysis/audio',response_model=AnalysisOut)
def analyze_audio(evidence_id:str,db:Session=Depends(get_db)):
    e=evidence_or_404(db,evidence_id)
    from ml.audio.infer import AudioDetector
    try: out=AudioDetector().predict(e.storage_path)
    except Exception as ex: raise HTTPException(503,f'Audio detector unavailable: {ex}')
    a=AnalysisResult(evidence_id=e.id,analysis_type='audio',result_json=out,model_version=out['model_version']); db.add(a); db.commit(); db.refresh(a); return a

@app.post('/analysis/video',response_model=AnalysisOut)
def analyze_video(evidence_id:str,db:Session=Depends(get_db)):
    e=evidence_or_404(db,evidence_id)
    from ml.video.infer import VideoDetector
    try: out=VideoDetector().predict(e.storage_path)
    except Exception as ex: raise HTTPException(503,f'Video detector unavailable: {ex}')
    a=AnalysisResult(evidence_id=e.id,analysis_type='video',result_json=out,model_version=out['model_version']); db.add(a); db.commit(); db.refresh(a); return a

@app.get('/analysis/{id}',response_model=AnalysisOut)
def get_analysis(id:str,db:Session=Depends(get_db)):
    a=db.get(AnalysisResult,id)
    if not a: raise HTTPException(404,'Analysis not found')
    return a

@app.get('/graph/{incident_id}')
def graph(incident_id:str,db:Session=Depends(get_db)):
    incident_or_404(db,incident_id); entities=db.query(Entity).filter(Entity.incident_id==incident_id).all(); return {'nodes':[{'id':e.id,'type':e.entity_type,'value':e.value} for e in entities],'edges':[]}

@app.get('/timeline/{incident_id}')
def timeline(incident_id:str,db:Session=Depends(get_db)):
    incident_or_404(db,incident_id); return {'events':[]}

@app.post('/reports/generate')
def generate_report(incident_id:str,db:Session=Depends(get_db)):
    from backend.app.services.report import generate_pdf
    incident=incident_or_404(db,incident_id); path=generate_pdf(db,incident); r=Report(incident_id=incident_id,path=path); db.add(r); db.commit(); db.refresh(r); return {'report_id':r.id,'path':path}

@app.post('/findings/risk')
def risk(incident_id:str,db:Session=Depends(get_db)):
    incident=incident_or_404(db,incident_id); analyses=db.query(AnalysisResult).join(Evidence).filter(Evidence.incident_id==incident_id).all(); findings={}
    for a in analyses: findings[a.analysis_type]=a.result_json
    result=assess(findings); incident.risk_level=result['level']; db.commit(); return result

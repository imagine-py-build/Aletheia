from fastapi import FastAPI, Depends, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pathlib import Path
from fastapi.responses import FileResponse
from datetime import datetime, timezone
import json
from backend.app.db.session import Base,engine,get_db
from backend.app.models.entities import Incident,Evidence,AnalysisResult,ChainOfCustody,Entity,Report,CaseSubmission,CaseMessage
from backend.app.schemas.api import IncidentCreate,IncidentOut,AnalysisOut,CaseStatusUpdate,CaseMessageCreate
from backend.app.services.evidence import save_upload
from backend.app.forensics.metadata import extract_metadata
from backend.app.forensics.ocr import extract_text
from backend.app.forensics.entities import extract_entities
from backend.app.forensics.risk import assess
from backend.app.core.config import settings


def make_json_safe(value):
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(v) for v in value]
    if hasattr(value, "item") and callable(value.item):
        try: return make_json_safe(value.item())
        except (ValueError, TypeError): pass
    if hasattr(value, "tolist") and callable(value.tolist):
        try: return make_json_safe(value.tolist())
        except (ValueError, TypeError): pass
    return value

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
    obj=Incident(title=body.title,description=body.description); db.add(obj); db.flush()
    if body.citizen_username:
        submission=CaseSubmission(
            incident_id=obj.id,
            citizen_username=body.citizen_username,
            citizen_name=body.citizen_name or body.citizen_username,
            citizen_email=body.citizen_email or '',
            status='PENDING_REVIEW'
        )
        db.add(submission)
        db.add(CaseMessage(
            incident_id=obj.id,
            sender_role='system',
            sender_name='Aletheia',
            message='Case submitted successfully and is awaiting investigator review.'
        ))
    db.commit(); db.refresh(obj); return obj


def submission_payload(row, db):
    evidences=db.query(Evidence).filter(Evidence.incident_id==row.incident_id).order_by(Evidence.created_at.asc() if hasattr(Evidence,'created_at') else Evidence.id.asc()).all()
    attachments=[{
        'id':e.id,
        'filename':e.original_filename,
        'mime_type':e.mime_type,
        'file_size':e.file_size,
        'sha256':e.sha256,
        'url':f'/evidence/{e.id}/content'
    } for e in evidences]
    return {
        'id':row.id,'incident_id':row.incident_id,'citizen_username':row.citizen_username,
        'citizen_name':row.citizen_name,'citizen_email':row.citizen_email,'status':row.status,
        'investigator_note':row.investigator_note,'created_at':row.created_at.isoformat(),
        'updated_at':row.updated_at.isoformat(),'attachments':attachments
    }

@app.get('/submissions')
def list_submissions(db:Session=Depends(get_db)):
    rows=db.query(CaseSubmission).order_by(CaseSubmission.updated_at.desc()).all()
    return [submission_payload(r,db) for r in rows]

@app.get('/submissions/mine')
def list_my_submissions(citizen_username:str,db:Session=Depends(get_db)):
    rows=db.query(CaseSubmission).filter(CaseSubmission.citizen_username==citizen_username).order_by(CaseSubmission.updated_at.desc()).all()
    return [submission_payload(r,db) for r in rows]

@app.patch('/submissions/{submission_id}')
def update_submission(submission_id:str,body:CaseStatusUpdate,db:Session=Depends(get_db)):
    row=db.get(CaseSubmission,submission_id)
    if not row: raise HTTPException(404,'Submission not found')
    row.status=body.status
    row.investigator_note=body.note
    row.updated_at=datetime.now(timezone.utc)
    db.add(CaseMessage(
        incident_id=row.incident_id,
        sender_role='investigator',
        sender_name='Investigator',
        message=(body.note.strip() if body.note and body.note.strip() else f'Case status changed to {body.status.replace("_"," ").title()}.')
    ))
    db.commit()
    return {'ok':True}

@app.get('/submissions/{submission_id}/messages')
def get_submission_messages(submission_id:str,db:Session=Depends(get_db)):
    row=db.get(CaseSubmission,submission_id)
    if not row: raise HTTPException(404,'Submission not found')
    msgs=db.query(CaseMessage).filter(CaseMessage.incident_id==row.incident_id).order_by(CaseMessage.created_at.asc()).all()
    return [{'id':m.id,'sender_role':m.sender_role,'sender_name':m.sender_name,'message':m.message,'created_at':m.created_at.isoformat()} for m in msgs]

@app.post('/submissions/{submission_id}/messages')
def send_submission_message(submission_id:str,body:CaseMessageCreate,sender_role:str='citizen',sender_name:str='Citizen',db:Session=Depends(get_db)):
    row=db.get(CaseSubmission,submission_id)
    if not row: raise HTTPException(404,'Submission not found')
    if not body.message.strip(): raise HTTPException(400,'Message cannot be empty')
    db.add(CaseMessage(incident_id=row.incident_id,sender_role=sender_role,sender_name=sender_name,message=body.message.strip()))
    row.updated_at=datetime.now(timezone.utc)
    db.commit()
    return {'ok':True}

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
    try:
        out=ImageDetector().predict(e.storage_path)
        out=make_json_safe(out)
    except Exception as ex: raise HTTPException(503,f'Image detector unavailable: {ex}')
    a=AnalysisResult(evidence_id=e.id,analysis_type='image',result_json=out,model_version=out['model_version']); db.add(a); db.commit(); db.refresh(a); return a

@app.post('/analysis/document',response_model=AnalysisOut)
def analyze_document(evidence_id:str,db:Session=Depends(get_db)):
    e=evidence_or_404(db,evidence_id)
    if not e.mime_type.startswith('image/'):
        raise HTTPException(400,'Document forgery screening requires an image file.')
    from ml.document.infer import DocumentForgeryDetector
    try:
        out=DocumentForgeryDetector().predict(e.storage_path)
        out=make_json_safe(out)
    except Exception as ex:
        raise HTTPException(503,f'Document forgery detector unavailable: {ex}')
    a=AnalysisResult(evidence_id=e.id,analysis_type='document',result_json=out,model_version=out['model_version'])
    db.add(a); db.commit(); db.refresh(a); return a

@app.post('/analysis/audio',response_model=AnalysisOut)
def analyze_audio(evidence_id:str,db:Session=Depends(get_db)):
    e=evidence_or_404(db,evidence_id)
    from ml.audio.infer import AudioDetector
    try:
        out=AudioDetector().predict(e.storage_path)
        out=make_json_safe(out)
    except Exception as ex: raise HTTPException(503,f'Audio detector unavailable: {ex}')
    a=AnalysisResult(evidence_id=e.id,analysis_type='audio',result_json=out,model_version=out['model_version']); db.add(a); db.commit(); db.refresh(a); return a

@app.post('/analysis/video',response_model=AnalysisOut)
def analyze_video(evidence_id:str,db:Session=Depends(get_db)):
    e=evidence_or_404(db,evidence_id)
    from ml.video.infer import VideoDetector
    try:
        out=VideoDetector().predict(e.storage_path)
        out=make_json_safe(out)
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

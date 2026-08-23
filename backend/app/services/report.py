from pathlib import Path
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from backend.app.core.config import settings
from backend.app.models.entities import Evidence,AnalysisResult

def generate_pdf(db,incident):
    out=Path(settings.storage_dir)/f'{incident.id}_forensic_report.pdf'; doc=SimpleDocTemplate(str(out),pagesize=A4); styles=getSampleStyleSheet(); story=[Paragraph('Aletheia — Forensic Report',styles['Title']),Paragraph(f'Incident: {incident.title}',styles['Heading2']),Paragraph(f'Incident ID: {incident.id}',styles['BodyText']),Spacer(1,12)]
    evs=db.query(Evidence).filter(Evidence.incident_id==incident.id).all()
    rows=[['Evidence','SHA-256','MIME','Size']]+[[e.original_filename,e.sha256,e.mime_type,str(e.file_size)] for e in evs]
    t=Table(rows,repeatRows=1); t.setStyle(TableStyle([('GRID',(0,0),(-1,-1),.5,colors.grey),('BACKGROUND',(0,0),(-1,0),colors.lightgrey)])); story += [Paragraph('Evidence Inventory',styles['Heading2']),t,Spacer(1,12)]
    for e in evs:
        story.append(Paragraph(e.original_filename,styles['Heading3']));
        for a in e.analyses: story.append(Paragraph(f'{a.analysis_type}: {a.result_json}',styles['BodyText']))
    story += [Spacer(1,12),Paragraph(f'Risk level: {incident.risk_level}',styles['Heading2']),Paragraph('AI findings are not legal certainty. Human verification is required.',styles['BodyText'])]
    doc.build(story); return str(out)

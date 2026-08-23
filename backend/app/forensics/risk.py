WEIGHTS={'image_deepfake':2.0,'video_deepfake':2.5,'audio_deepfake':2.0,'threat':2.0,'blackmail':2.5,'sextortion':3.0,'financial_coercion':2.0,'metadata_anomaly':1.0,'provenance_invalid':1.0}
def assess(findings):
    score=0.0; reasons=[]
    for key,val in findings.items():
        if isinstance(val,dict) and val.get('label') in ('FAKE','AI_GENERATED','SUSPICIOUS','TRUE'):
            w=WEIGHTS.get(key,1); score+=w; reasons.append(f'{key}: {val.get("label")}')
        elif isinstance(val,bool) and val:
            w=WEIGHTS.get(key,1); score+=w; reasons.append(key)
    level='LOW' if score<2 else 'MEDIUM' if score<4 else 'HIGH' if score<7 else 'CRITICAL'
    return {'level':level,'score':score,'reasons':reasons,'method':'transparent weighted evidence rules; not a probability'}

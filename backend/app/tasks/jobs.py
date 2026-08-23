from .celery_app import celery
@celery.task
def analysis_job(analysis_type,evidence_id):
    # The synchronous API paths remain available for development; production deployments can enqueue these jobs.
    return {'analysis_type':analysis_type,'evidence_id':evidence_id,'status':'queued'}

import uuid
from datetime import datetime, timezone
from sqlalchemy import String, Text, DateTime, Integer, Float, ForeignKey, JSON, Boolean
from sqlalchemy.orm import Mapped, mapped_column, relationship
from backend.app.db.session import Base

class User(Base):
    __tablename__='users'
    id: Mapped[str]=mapped_column(String(36), primary_key=True, default=lambda:str(uuid.uuid4()))
    username: Mapped[str]=mapped_column(String(120), unique=True)
    created_at: Mapped[datetime]=mapped_column(DateTime, default=lambda:datetime.now(timezone.utc))

class Incident(Base):
    __tablename__='incidents'
    id: Mapped[str]=mapped_column(String(36), primary_key=True, default=lambda:str(uuid.uuid4()))
    title: Mapped[str]=mapped_column(String(255))
    description: Mapped[str|None]=mapped_column(Text)
    risk_level: Mapped[str]=mapped_column(String(20), default='LOW')
    created_at: Mapped[datetime]=mapped_column(DateTime, default=lambda:datetime.now(timezone.utc))
    evidence=relationship('Evidence', back_populates='incident', cascade='all, delete-orphan')

class Evidence(Base):
    __tablename__='evidence'
    id: Mapped[str]=mapped_column(String(36), primary_key=True, default=lambda:str(uuid.uuid4()))
    incident_id: Mapped[str|None]=mapped_column(ForeignKey('incidents.id'), nullable=True)
    original_filename: Mapped[str]=mapped_column(String(500))
    storage_path: Mapped[str]=mapped_column(String(1000))
    sha256: Mapped[str]=mapped_column(String(64), index=True)
    mime_type: Mapped[str]=mapped_column(String(200))
    file_size: Mapped[int]=mapped_column(Integer)
    metadata_json: Mapped[dict]=mapped_column(JSON, default=dict)
    created_at: Mapped[datetime]=mapped_column(DateTime, default=lambda:datetime.now(timezone.utc))
    incident=relationship('Incident', back_populates='evidence')
    analyses=relationship('AnalysisResult', back_populates='evidence', cascade='all, delete-orphan')

class AnalysisResult(Base):
    __tablename__='analysis_results'
    id: Mapped[str]=mapped_column(String(36), primary_key=True, default=lambda:str(uuid.uuid4()))
    evidence_id: Mapped[str]=mapped_column(ForeignKey('evidence.id'))
    analysis_type: Mapped[str]=mapped_column(String(80))
    status: Mapped[str]=mapped_column(String(30), default='completed')
    result_json: Mapped[dict]=mapped_column(JSON, default=dict)
    model_version: Mapped[str|None]=mapped_column(String(255))
    created_at: Mapped[datetime]=mapped_column(DateTime, default=lambda:datetime.now(timezone.utc))
    human_verified: Mapped[bool]=mapped_column(Boolean, default=False)
    evidence=relationship('Evidence', back_populates='analyses')

class ModelVersion(Base):
    __tablename__='model_versions'
    id: Mapped[str]=mapped_column(String(36), primary_key=True, default=lambda:str(uuid.uuid4()))
    name: Mapped[str]=mapped_column(String(200))
    version: Mapped[str]=mapped_column(String(100))
    dataset: Mapped[str|None]=mapped_column(String(500))
    metrics: Mapped[dict]=mapped_column(JSON, default=dict)
    created_at: Mapped[datetime]=mapped_column(DateTime, default=lambda:datetime.now(timezone.utc))

class Entity(Base):
    __tablename__='entities'
    id: Mapped[str]=mapped_column(String(36), primary_key=True, default=lambda:str(uuid.uuid4()))
    incident_id: Mapped[str]=mapped_column(ForeignKey('incidents.id'))
    entity_type: Mapped[str]=mapped_column(String(80))
    value: Mapped[str]=mapped_column(String(500))
    confidence: Mapped[float|None]=mapped_column(Float)

class Relationship(Base):
    __tablename__='relationships'
    id: Mapped[str]=mapped_column(String(36), primary_key=True, default=lambda:str(uuid.uuid4()))
    incident_id: Mapped[str]=mapped_column(ForeignKey('incidents.id'))
    source_id: Mapped[str]=mapped_column(String(36))
    target_id: Mapped[str]=mapped_column(String(36))
    relation_type: Mapped[str]=mapped_column(String(100))

class TimelineEvent(Base):
    __tablename__='timeline_events'
    id: Mapped[str]=mapped_column(String(36), primary_key=True, default=lambda:str(uuid.uuid4()))
    incident_id: Mapped[str]=mapped_column(ForeignKey('incidents.id'))
    timestamp: Mapped[datetime]=mapped_column(DateTime)
    event_type: Mapped[str]=mapped_column(String(100))
    description: Mapped[str]=mapped_column(Text)
    evidence_id: Mapped[str|None]=mapped_column(String(36))

class Report(Base):
    __tablename__='reports'
    id: Mapped[str]=mapped_column(String(36), primary_key=True, default=lambda:str(uuid.uuid4()))
    incident_id: Mapped[str]=mapped_column(ForeignKey('incidents.id'))
    path: Mapped[str]=mapped_column(String(1000))
    created_at: Mapped[datetime]=mapped_column(DateTime, default=lambda:datetime.now(timezone.utc))

class ChainOfCustody(Base):
    __tablename__='chain_of_custody'
    id: Mapped[str]=mapped_column(String(36), primary_key=True, default=lambda:str(uuid.uuid4()))
    evidence_id: Mapped[str]=mapped_column(ForeignKey('evidence.id'))
    action: Mapped[str]=mapped_column(String(100))
    actor: Mapped[str]=mapped_column(String(200))
    timestamp: Mapped[datetime]=mapped_column(DateTime, default=lambda:datetime.now(timezone.utc))
    details: Mapped[dict]=mapped_column(JSON, default=dict)

class AuditLog(Base):
    __tablename__='audit_logs'
    id: Mapped[str]=mapped_column(String(36), primary_key=True, default=lambda:str(uuid.uuid4()))
    actor: Mapped[str]=mapped_column(String(200))
    action: Mapped[str]=mapped_column(String(200))
    target_id: Mapped[str|None]=mapped_column(String(36))
    details: Mapped[dict]=mapped_column(JSON, default=dict)
    created_at: Mapped[datetime]=mapped_column(DateTime, default=lambda:datetime.now(timezone.utc))

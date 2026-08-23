from pydantic import BaseModel, ConfigDict
from typing import Any
class IncidentCreate(BaseModel): title:str; description:str|None=None
class IncidentOut(BaseModel): model_config=ConfigDict(from_attributes=True); id:str; title:str; description:str|None; risk_level:str
class AnalysisOut(BaseModel): model_config=ConfigDict(from_attributes=True); id:str; analysis_type:str; status:str; result_json:dict[str,Any]

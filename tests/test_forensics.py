from backend.app.forensics.entities import extract_entities
from backend.app.forensics.risk import assess

def test_entities():
 x=extract_entities('Pay ₹10,000 to test@example.com or call 9876543210')
 assert any(e['type']=='money' for e in x); assert any(e['type']=='email' for e in x); assert any(e['type']=='phone' for e in x)

def test_risk_transparency():
 r=assess({'threat':True,'blackmail':True}); assert r['level'] in {'MEDIUM','HIGH','CRITICAL'}; assert 'reasons' in r

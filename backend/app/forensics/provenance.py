import json,subprocess

def verify_c2pa(path):
    """Use c2patool if installed. Absence of credentials is returned as absence, never as fake provenance."""
    try:
        p=subprocess.run(['c2patool',str(path),'--json'],capture_output=True,text=True,timeout=30)
        if p.returncode!=0:return {'present':False,'valid':None,'status':'tool_error','details':p.stderr[-500:]}
        data=json.loads(p.stdout); return {'present':bool(data),'valid':None,'status':'inspected','credentials':data}
    except FileNotFoundError:return {'present':False,'valid':None,'status':'c2patool_unavailable'}
    except Exception as e:return {'present':False,'valid':None,'status':'error','details':str(e)}

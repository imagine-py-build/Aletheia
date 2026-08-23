import re
PATTERNS={
 'phone':r'(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{9}(?!\d)',
 'email':r'\b[\w.+-]+@[\w.-]+\.\w{2,}\b',
 'url':r'https?://[^\s]+',
 'upi':r'\b[\w.-]+@[\w.-]+\b',
 'money':r'(?:₹|Rs\.?|INR)\s?[\d,]+(?:\.\d+)?',
 'date':r'\b(?:\d{1,2}[/-]){2}\d{2,4}\b'
}
def extract_entities(text):
    out=[]
    for typ,pat in PATTERNS.items():
        for m in re.finditer(pat,text,flags=re.I): out.append({'type':typ,'value':m.group(0),'start':m.start(),'end':m.end()})
    return out

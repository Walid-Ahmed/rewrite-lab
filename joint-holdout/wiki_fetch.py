import json,re,time,hashlib,urllib.request,urllib.parse,html
from html.parser import HTMLParser
OUT='wiki_candidates.json'
T={'survey_methods':["Participation bias","Response rate (survey)","Stratified sampling","Cluster sampling","Questionnaire construction","Social-desirability bias","Acquiescence bias","Computer-assisted telephone interviewing","Sampling frame","Opinion poll","Likert scale","Panel data"],
'operations':["Economic order quantity","Just-in-time manufacturing","Bullwhip effect","Safety stock","Kanban","Theory of constraints","Reorder point","Lean manufacturing","Vendor-managed inventory","Six Sigma","Inventory","Supply chain"],
'labor_economics':["Workforce productivity","Minimum wage","Collective bargaining","Human capital","Job polarization","Efficiency wage","Compensating differential","Gender pay gap","Structural unemployment","Labour economics","Unemployment","Labour force"],
'public_economics':["Public good","Free-rider problem","Externality","Pigouvian tax","Tragedy of the commons","Progressive tax","Cost–benefit analysis","Tax incidence","Deadweight loss","Merit good","Lindahl tax","Social security"],
'agriculture':["Irrigation","Drip irrigation","Crop rotation","Surface irrigation","Center-pivot irrigation","No-till farming","Soil salinity","Cover crop","Green Revolution","Integrated pest management","Aquifer","Terrace (earthworks)"],
'public_health':["Herd immunity","Epidemiology","Oral rehydration therapy","Cholera","Health promotion","Screening (medicine)","Malaria","Antimicrobial resistance","Social determinants of health","Water fluoridation","Vaccination","Tobacco smoking"],
'law_governance':["Judicial review","Separation of powers","Common law","Federalism","Administrative law","Due process","Contract","Tort","Ombudsman","Proportional representation","Constitutional amendment","Civil law (legal system)"],
'history':["Hanseatic League","Black Death","Industrial Revolution","Silk Road","Meiji Restoration","Congress of Vienna","Printing press","Peace of Westphalia","Columbian exchange","Enclosure","Abolitionism","Ottoman Empire"],
'psychology_education':["Working memory","Cognitive dissonance","Spacing effect","Classical conditioning","Bloom's taxonomy","Zone of proximal development","Attachment theory","Dunning–Kruger effect","Formative assessment","Stereotype threat","Confirmation bias","Mindset"],
'computing':["Garbage collection (computer science)","Public-key cryptography","Hash table","Compiler","Virtual memory","Deadlock","Domain Name System","Version control","Load balancing (computing)","B-tree","Cache replacement policies","Transport Layer Security"]}
API='https://en.wikipedia.org/w/api.php?'
UA={'User-Agent':'qwen3-humanizer-holdout/1.0 (research evaluation; polite single-threaded)'}
def get(**p):
    p.update(format='json',formatversion='2')
    for t in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(API+urllib.parse.urlencode(p),headers=UA),timeout=40) as r:return json.load(r)
        except urllib.error.HTTPError as e:
            err=e;wait=int(e.headers.get('Retry-After') or 0) or 60*(t+1);print('backoff',e.code,wait,flush=True);time.sleep(wait)
        except Exception as e:
            err=e;time.sleep(20*(t+1))
    raise err
class P(HTMLParser):
    SKIP={'sup','style','table','script','math'}
    def __init__(s):super().__init__();s.d=0;s.top=0;s.cur=None;s.out=[];s.bad=0;s.skip=0
    def handle_starttag(s,t,a):
        a=dict(a)
        if t in('sup','style','script','math') or (t=='span' and 'mwe-math' in (a.get('class') or '')) or t=='table':s.skip+=1
        if t=='p' and s.cur is None and not s.skip:s.cur=[]
    def handle_endtag(s,t):
        if t in('sup','style','script','math','table') or (False):s.skip=max(0,s.skip-1)
        if t=='p' and s.cur is not None:s.out.append(''.join(s.cur));s.cur=None
    def handle_data(s,x):
        if s.cur is not None and not s.skip:s.cur.append(x)
def good(t):
    w=len(t.split())
    if not 62<=w<=150:return False
    if not re.search(r'[.!?]["”)]?$',t):return False
    if len(re.findall(r'[.!?](?:\s|$)',t))<3:return False
    if re.search(r'displaystyle|\{|\}|\\|\[|\]|citation needed|ISBN|\bhttp|:$|following|see below|shown (?:here|below|above)|Figure \d',t):return False
    if t[0].islower() or re.match(r'^(However|Thus|This|These|It|They|He|She|In this)\b',t):return False
    if sum(ord(c)>0x24F and c not in '–—‘’“”…' for c in t)>2:return False
    return True
import os,urllib.error
res=json.load(open(OUT)) if os.path.exists(OUT) else {}
for dom,titles in T.items():
    for title in titles:
        have=res.get(dom,[])
        if any(x.get('asked')==title for x in have):continue
        if sum(1 for x in have if x['eligible'])>=10:break
        q=get(action='query',prop='revisions',titles=title,rvlimit=1,rvstart='2021-12-31T23:59:59Z',rvdir='older',rvprop='ids|timestamp',redirects=1)
        pg=q['query']['pages'][0]
        if 'revisions' not in pg:print('NOREV',title);continue
        rv=pg['revisions'][0];time.sleep(6)
        p=get(action='parse',oldid=rv['revid'],prop='text',disableeditsection=1,disablelimitreport=1)
        h=p['parse']['text'];par=P();par.feed(h)
        ps=[re.sub(r'\s+',' ',html.unescape(x)).strip() for x in par.out]
        el=[x for x in ps if good(x)]
        el.sort(key=lambda x:hashlib.sha256(x.encode()).hexdigest())
        res.setdefault(dom,[]).append(dict(asked=title,title=pg['title'],pageid=pg['pageid'],revid=rv['revid'],rev_timestamp=rv['timestamp'],url=f"https://en.wikipedia.org/w/index.php?oldid={rv['revid']}",eligible=len(el),options=el[:3]))
        json.dump(res,open(OUT,'w'),ensure_ascii=False,indent=1);print(dom,title,rv['timestamp'][:10],len(ps),len(el),flush=True);time.sleep(6)
json.dump(res,open(OUT,'w'),ensure_ascii=False,indent=1)

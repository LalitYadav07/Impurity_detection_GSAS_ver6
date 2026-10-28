"""Conservative literal binding for explicitly complete stage Free: lists.

This is deliberately not a general natural-language interpreter. Unrecognized
clauses remain model-planned and require review. Known literal lists are user
instructions, so they take precedence over a model's accidental carried flags.
"""
import copy
import re


def mask_quoted(text):
    result=list(text);quote=None;escaped=False
    for i,char in enumerate(text):
        if quote:
            result[i]=' '
            if escaped:escaped=False
            elif char=='\\':escaped=True
            elif char==quote:quote=None
        elif char in "\"'" and (i==0 or text[i-1].isspace() or text[i-1] in '(:=,'):
            quote=char;result[i]=' '
    return ''.join(result)


def split_unquoted(text,separator,first=False):
    result=[];start=0;quote=None;escaped=False
    for i,char in enumerate(text):
        if escaped:escaped=False;continue
        if quote:
            if char=='\\':escaped=True
            elif char==quote:quote=None
            continue
        if char in "\"'" and (i==0 or text[i-1].isspace() or text[i-1] in '(:=,'):
            quote=char;continue
        if char==separator:
            result.append(text[start:i]);start=i+1
            if first:return [result[0],text[start:]]
    if quote:return None
    result.append(text[start:]);return result


def stage_sections(request):
    matches=list(re.finditer(r'^\s*(?:stage\s+)?(\d+)\s*[:.)-]\s*',request,re.I|re.M))
    if not matches or [int(m.group(1)) for m in matches]!=list(range(1,len(matches)+1)):return []
    return [{'stage':int(m.group(1)),'text':request[m.end():matches[i+1].start() if i+1<len(matches) else len(request)].strip()} for i,m in enumerate(matches)]


def _literal_owner(item,records,key_prefix,role):
    # Prefer explicit canonical keys outside quoted metadata. Full names must be
    # introduced by their role, so an atom called B2 cannot select phase B2.
    names=[];remaining=mask_quoted(item)
    for record in sorted(records,key=lambda r:len(r.get('name','')),reverse=True):
        name=record.get('name')
        if not name:continue
        pattern=r'\b'+role+r'\s+'+re.escape(repr(name))
        matches=list(re.finditer(pattern,item,re.I))
        if any(remaining[m.start():m.start()+len(role)].lower()==role.lower() for m in matches):names.append(record['key'])
    keys=re.findall(r'(?<![A-Za-z0-9_])'+key_prefix+r'\d+(?![A-Za-z0-9_])',remaining)
    explicit=set(keys)
    valid={r['key'] for r in records}
    if explicit-valid:return None
    owners=explicit|set(names)
    if len(owners)==1:return next(iter(owners))
    if not owners and len(records)==1:
        # An unresolved explicit name cannot fall back to the only owner.
        if re.search(r'\b'+role+r'\s+[\"\']',remaining,re.I):return None
        return records[0]['key']
    return None


def parse_free_item(item,context):
    phases=context['phases'];histograms=context['histograms']
    phase=_literal_owner(item,phases,'p','phase');histogram=_literal_owner(item,histograms,'h','histogram')
    clean=item
    for r in [*phases,*histograms]:
        name=r.get('name')
        if name:clean=clean.replace(repr(name),' ')
    clean=re.sub(r'\b[ph]\d+\b',' ',clean)
    # Type words are read only after quoted scientific identities are removed.
    if re.search(r'\b(?:HAP\s+)?phase\s+(?:scale|fraction)\b',clean,re.I):
        target={'kind':'phase_scale','phase':phase,'histogram':histogram};words=r'\b(?:HAP|phase|scale|fraction)\b'
    elif re.search(r'\b(?:global\s+)?histogram\s+scale\b',clean,re.I):
        if re.search(r'\bphase\b',clean,re.I):return None
        target={'kind':'histogram_scale','histogram':histogram};words=r'\b(?:global|histogram|scale)\b'
    elif re.search(r'\bbackground\b',clean,re.I):
        if re.search(r'\bphase\b',clean,re.I):return None
        target={'kind':'background','histogram':histogram};words=r'\b(?:existing|polynomial|background)\b'
    elif re.search(r'\b(?:unit\s+)?cell\b',clean,re.I):
        if re.search(r'\b(?:only\s+for|for)\s+histogram\b',item,re.I):return None
        target={'kind':'cell','phase':phase};words=r'\b(?:unit|cell)\b'
    elif re.search(r'\bpreferred\s+orientation\b',clean,re.I):
        target={'kind':'preferred_orientation','phase':phase,'histogram':histogram};words=r'\b(?:preferred|orientation)\b'
    else:
        p=next((p for p in phases if p['key']==phase),None)
        if not p or re.search(r'\b(?:only\s+for|for)\s+histogram\b',item,re.I):return None
        labels=[a['label'] for a in p['atoms'] if re.search(r'(?<![A-Za-z0-9_])'+re.escape(a['label'])+r'(?![A-Za-z0-9_])',clean)]
        if len(labels)!=1:return None
        label=labels[0];clean=re.sub(r'(?<![A-Za-z0-9_])'+re.escape(label)+r'(?![A-Za-z0-9_])',' ',clean,count=1)
        flags=set()
        if re.search(r'\b(?:site\s+)?occupancy\b',clean,re.I):flags.add('F')
        if re.search(r'\bUiso\b|\bisotropic\s+displacement\b',clean,re.I):flags.add('U')
        if re.search(r'\bpositions?\b|\bcoordinates?\b',clean,re.I):flags.add('X')
        for code in re.findall(r'\b[FXU]{1,3}\b',clean):flags.update(code)
        if not flags:return None
        target={'kind':'atom','phase':phase,'atom':label,'flags':''.join(f for f in 'FXU' if f in flags)}
        words=r'\b(?:site|occupancy|Uiso|isotropic|displacement|symmetry-independent|positions?|coordinates?|[FXU]{1,3})\b'
    if any(v is None for v in target.values()):return None
    remainder=re.sub(words,' ',clean,flags=re.I)
    remainder=re.sub(r'\b(?:in|of|for|phase|histogram|the|existing|only|atom|and|together|jointly|with)\b',' ',remainder,flags=re.I)
    # Values, wildcard selectors and unknown operations must not be discarded.
    if re.sub(r'[\s(),:]+','',remainder):return None
    return target


def literal_stage_cards(request,context):
    complete=bool(re.search(r'complete\s+(?:free|active)\s+set|everything\s+unlisted\s+(?:stays?|is|remains?)\s+fixed|hold\s+all\s+unlisted',request,re.I))
    cards=stage_sections(request)
    for card in cards:
        card['literal_free']=None
        # Only bind an unconditional leading cycle clause. Do not infer a
        # limit from quoted text, a conditional clause, or a later revision.
        cycle=re.match(r'^(?:at most|up to|maximum(?: of)?)\s+(\d+)\s+cycles\b(?=\s*(?:[.;:]|$))',mask_quoted(card['text']).strip(),re.I)
        if cycle:card['literal_cycles']=int(cycle.group(1))
        if not complete:continue
        match=re.search(r'\bFree\s*:\s*',card['text'],re.I)
        if not match:continue
        split=split_unquoted(card['text'][match.end():],'.',first=True)
        if not split:continue
        clause=split[0];parts=split_unquoted(clause,';')
        if not parts or not all(p.strip() for p in parts):continue
        targets=[parse_free_item(p.strip(),context) for p in parts]
        if any(t is None for t in targets):continue
        merged={}
        for target in targets:
            identity=tuple(target.get(k) for k in ['kind','phase','histogram','atom'])
            if identity in merged and target['kind']=='atom':
                flags=set(merged[identity]['flags']+target['flags']);merged[identity]['flags']=''.join(f for f in 'FXU' if f in flags)
            else:merged[identity]=target
        card.update(literal_free=list(merged.values()),source_free_clause=clause.strip())
    return cards


def normalized_targets(targets):
    import json
    result=[]
    for target in targets:
        target=dict(target)
        if 'flags' in target:target['flags']=''.join(sorted(target['flags']))
        result.append(json.dumps(target,sort_keys=True))
    return sorted(result)


def check_literal_free(plan,cards):
    corrections=[]
    if not isinstance(plan,dict) or plan.get('questions') or plan.get('unsupported'):return corrections
    if not isinstance(plan.get('stages'),list) or len(plan['stages'])!=len(cards):return corrections
    for stage,card in zip(plan['stages'],cards):
        if not isinstance(stage,dict):continue
        if card.get('literal_cycles') is not None and stage.get('cycles')!=card['literal_cycles']:
            corrections.append({'stage':card['stage'],'model_cycles':stage.get('cycles'),'request_cycles':card['literal_cycles']})
        targets=card.get('literal_free')
        if targets is None:continue
        previous=stage.get('free')
        if not isinstance(previous,list):continue
        if normalized_targets(previous)!=normalized_targets(targets):
            corrections.append({'stage':card['stage'],'source_free_clause':card['source_free_clause'],'model_free':copy.deepcopy(previous),'request_free':copy.deepcopy(targets)})
    return corrections


def natural_atom_only_cards(request,context,count):
    """Bind a narrow, explicit 'stage ... refine only p0:O1 positions/Uiso'.

    This does not split mixed-family prose or infer missing atom identities.
    Unknown clauses stay unbound and remain subject to model and user review.
    """
    if not count or not 1<=count<=20:return []
    words={'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,'ten':10,
           'first':1,'second':2,'third':3,'fourth':4,'fifth':5,'sixth':6,'seventh':7,'eighth':8,'ninth':9,'tenth':10}
    choices='|'.join(words)
    clean=mask_quoted(request)
    ordinals='first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth'
    pattern=r'\bstage\s+(\d+|'+choices+r')\b|\b('+ordinals+r')\s+stage\b'
    markers=[]
    for match in re.finditer(pattern,clean,re.I):
        token=(match.group(1) or match.group(2)).lower();number=int(token) if token.isdigit() else words[token]
        if not 1<=number<=count:return []
        markers.append((number,match.start(),match.end()))
    if not markers and count==1:markers=[(1,0,0)]
    if not markers or len({n for n,_,_ in markers})!=len(markers):return []
    if [n for n,_,_ in markers]!=sorted(n for n,_,_ in markers):return []
    cards=[{'stage':i,'text':'','literal_free':None} for i in range(1,count+1)]
    for index,(number,start,end) in enumerate(markers):
        segment=request[end:markers[index+1][1] if index+1<len(markers) else len(request)];masked=mask_quoted(segment)
        matches=list(re.finditer(r'\brefine\s+only\s+',masked,re.I))
        cards[number-1]['text']=segment.strip()
        if len(matches)!=1:continue
        match=matches[0]
        if re.search(r'\brefine\b',masked[:match.start()],re.I):continue
        if re.search(r"(?:do\s+not|don['’]t|never|avoid|wrong\s+to)\s*$",masked[max(0,match.start()-20):match.start()],re.I):continue
        tail=segment[match.end():];masked_tail=mask_quoted(tail)
        stop=re.search(r'[,;.]|\s+for\s+(?:at\s+most|up\s+to|\d+)',masked_tail,re.I)
        clause=tail[:stop.start()] if stop else tail
        if stop and re.search(r'\brefine\b|\bthen\b',masked_tail[stop.end():],re.I):continue
        # A comma inside a multi-target clause must not turn its first item
        # into a sole target. Only a cycle-limit suffix may follow that comma.
        if stop and stop.group(0)==',' and not re.match(r'\s*for\s+(?:at\s+most|up\s+to|\d+)',masked_tail[stop.end():],re.I):continue
        target=parse_free_item(clause.strip(),context)
        if target and target['kind']=='atom':
            cards[number-1].update(literal_free=[target],source_free_clause=clause.strip(),binding='explicit_natural_atom_only_clause')
    return cards if any(c['literal_free'] is not None for c in cards) else []

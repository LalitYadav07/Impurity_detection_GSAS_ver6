"""Launch an NDIP-test viewer with selected Galaxy vault references, never secrets.

Example: python -m radar_pd_nova.launch --history HISTORY --version VERSION --with-illumine --with-bilbao
The Galaxy API key is read from GALAXY_API_KEY or this Mac's existing Keychain.
"""
import argparse
import json
import os
import subprocess
import urllib.parse
import urllib.request

BASE = 'https://ndip-test.ornl.gov'
TOOL = 'neutrons_radar_pd_interactive_prototype'


def credential_context(rows,names=('ornl_illumine',)):
    """Only selected, populated references belong in a tool launch request."""
    context=[]
    allowed={'ornl_illumine':'ORNL Illumine','bilbao':'Bilbao'}
    if not names or len(set(names))!=len(names) or set(names)-set(allowed):raise ValueError('Select supported, distinct credential services')
    for name in names:
        selected=[row for row in rows if row.get('name')==name and row.get('version')=='1.0']
        if len(selected)!=1:raise ValueError('Select one populated '+allowed[name]+' credential group for this tool version in Galaxy before launching')
        row=selected[0]
        group=next((g for g in row.get('groups',[]) if g['id']==row.get('current_group_id')),None)
        if not group or not any(s.get('name')=='api_key' and s.get('is_set') for s in group.get('secrets',[])):
            raise ValueError('Select one populated '+allowed[name]+' credential group for this tool version in Galaxy before launching')
        context.append(dict(user_credentials_id=row['id'],name=row['name'],version=row['version'],
            selected_group=dict(id=group['id'],name=group['name'])))
    return context


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history',required=True)
    parser.add_argument('--version',required=True)
    parser.add_argument('--with-illumine',action='store_true')
    parser.add_argument('--with-bilbao',action='store_true')
    args=parser.parse_args(argv)
    key=os.getenv('GALAXY_API_KEY','')
    if not key:
        key=subprocess.run(['security','find-generic-password','-a','ly6','-s',
            'radar-pd.ndip-test.galaxy-api','-w'],capture_output=True,text=True,check=True,timeout=10).stdout.strip()
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self,*_,**__):return None
    opener=urllib.request.build_opener(NoRedirect())
    def request(path,payload=None):
        req=urllib.request.Request(BASE+path,headers={'x-api-key':key,'Content-Type':'application/json'},
            data=json.dumps(payload).encode() if payload is not None else None)
        with opener.open(req,timeout=35) as response:return json.load(response)
    payload=dict(tool_id=TOOL,tool_version=args.version,history_id=args.history,inputs={})
    selected_services=[name for name,enabled in [('ornl_illumine',args.with_illumine),('bilbao',args.with_bilbao)] if enabled]
    if selected_services:
        query=urllib.parse.urlencode(dict(source_type='tool',source_id=TOOL,source_version=args.version))
        payload['credentials_context']=credential_context(request('/api/users/current/credentials?'+query),selected_services)
    result=request('/api/tools',payload)
    print(json.dumps(dict(history_id=args.history,version=args.version,credential_selected=bool(payload.get('credentials_context')),
        jobs=[j['id'] for j in result['jobs']]),indent=2))


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Android-only Horizon probe lifetime wraps; never rewrites Mac generated output."""
import argparse,copy,hashlib,json,re,struct,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import buildinfo
EXPECTED={'probe_ctor':('0x1006c070','8bc133c989480466894802c3'),
          'probe_dtor':('0x1006c080','8b410485c074068b0850ff5108c3')}
IMAGE_SHA='85398cd8ddb142e8d7435c9e78370e0504218da409d8f4abde0e29615aebffe5'
RETAIL_SHA='bda769e226d71a43335c105fd6f72ed19af0a3d815a079b367356de0731a0d9a'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def android_build(original):
    if not original:raise ValueError('prepare the local game translation first')
    result=copy.deepcopy(original)
    extra=buildinfo.known()[original['build']].get('android_wraps',{})
    if extra:
        if original['build']!='2025-11-12' or original['ffximain_sha']!=RETAIL_SHA:
            raise ValueError('Android lifetime wraps require checked Horizon image')
        if extra!={k:v[0] for k,v in EXPECTED.items()}:
            raise ValueError('Android lifetime anchor metadata differs')
        validate(original)
        result['wraps'].update(extra)
    return result

def validate(original):
    folder=ROOT/'generated/images'/original['build']
    image=folder/'FFXiMain.unpacked.dll';retail=folder/'FFXiMain.retail.dll'
    data=image.read_bytes()
    if hashlib.sha256(data).hexdigest()!=IMAGE_SHA or digest(retail)!=RETAIL_SHA:
        raise ValueError('exact retail and unpacked image hashes required')
    peoff=struct.unpack_from('<I',data,0x3c)[0]
    if data[peoff:peoff+4]!=b'PE\0\0':raise ValueError('PE signature mismatch')
    coff=peoff+4;nsects,optsize=struct.unpack_from('<H',data,coff+2)[0],struct.unpack_from('<H',data,coff+16)[0]
    opt=coff+20
    if struct.unpack_from('<H',data,opt)[0]!=0x10b:raise ValueError('PE32 required')
    base=struct.unpack_from('<I',data,opt+28)[0]
    sections=[]
    for i in range(nsects):
        p=opt+optsize+i*40;virtual_size,rva,size,offset=struct.unpack_from('<IIII',data,p+8)
        sections.append((rva,size,offset))
    functions={f['entry']:f for f in json.loads(Path(original['ffximain_meta']).read_text())['functions']}
    checked=[]
    for name,(address,code) in EXPECTED.items():
        entry=int(address,16);raw=bytes.fromhex(code);f=functions.get(entry)
        if not f or f['thunk'] or f['ranges']!=[[entry,entry+len(raw)]]:
            raise ValueError(name+': exact known function range required')
        locations=[o+entry-base-r for r,s,o in sections if r<=entry-base and entry-base+len(raw)<=r+s]
        if len(locations)!=1 or data[locations[0]:locations[0]+len(raw)]!=raw:
            raise ValueError(name+': exact source body mismatch')
        checked.append({'name':name,'entry':address,'bytes':len(raw),'function_range_verified':True,'body_verified':True})
    return {'retail_sha256':RETAIL_SHA,'unpacked_sha256':IMAGE_SHA,'checks':checked,'android_only':True}

def protected_manifest():
    files=list((ROOT/'generated/all').rglob('*'))+[ROOT/'generated/build.h']
    return {str(p.relative_to(ROOT)):digest(p) for p in files if p.is_file()}
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--generate',action='store_true');ap.add_argument('--evidence',type=Path,required=True);args=ap.parse_args()
    original=buildinfo.current();configuration=android_build(original);report=validate(original)
    before=protected_manifest();report['protected_before']=before
    if args.generate:
        folder=ROOT/'generated/images'/original['build']
        cmd=[sys.executable,str(ROOT/'recomp/recomp.py'),'--meta',original['ffximain_meta'],'--image',str(folder/'FFXiMain.unpacked.dll'),'--retail',str(folder/'FFXiMain.retail.dll'),'--out',str(ROOT/'generated/android-all'),'--wraps',','.join(k+'='+v for k,v in configuration['wraps'].items()),'--all']
        if configuration['hooks']:
            cmd+=['--hooks',','.join(k+'='+v for k,v in configuration['hooks'].items())]
        if configuration['patches']:
            patches=ROOT/'generated/android-patches.json'
            patches.write_text(json.dumps(configuration['patches'],indent=2)+'\n')
            cmd+=['--patches',str(patches)]
        subprocess.run(cmd,cwd=ROOT,check=True)
    table=(ROOT/'generated/android-all/table.c').read_text()
    rows=len(re.findall(r'^    \{ 0x[0-9A-Fa-f]+u, f_',table,re.M))
    if rows<10000 or any('rt_orig_'+name+' = f_'+EXPECTED[name][0][2:]+'_body;' not in table for name in EXPECTED):
        raise ValueError('complete Android function table and nonzero original lifetime bodies required')
    report['function_table_rows']=rows
    report['complete_android_tree_admitted']=True
    after=protected_manifest();report['protected_after']=after;report['protected_unchanged']=before==after
    if before!=after:raise ValueError('Mac generated output changed')
    report['android_wraps']=configuration['wraps'];report['generated']=args.generate
    report['android_files']={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'generated/android-all').glob('*') if p.is_file()}
    args.evidence.parent.mkdir(parents=True,exist_ok=True);args.evidence.write_text(json.dumps(report,indent=2)+'\n')
    print('Android wrap validation/generation passed; Mac generated output unchanged')
if __name__=='__main__':main()

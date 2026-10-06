"""S16 finite GPFS quota provider; no GPU calls and no qualification inheritance."""
from pathlib import Path
import os,time,uuid
from urllib.parse import unquote
from common.suite_io import require,bounded,atomic_json,process_ok,read_json,sha

def quota_rows(text,uid,fileset):
    header=None;selected=[]
    for line in text.splitlines():
        parts=line.rstrip(':').split(':')
        if len(parts)>2 and parts[:2]==['mmlsquota','user']:
            if parts[2]=='HEADER':header=parts
            else:
                require(header is not None and len(parts)==len(header),'mmlsquota field layout')
                row=dict(zip(header[6:],map(unquote,parts[6:])))
                if row['quotaType']=='USR' and row['id']==str(uid) and row['filesystemName']=='gpfs' and row['filesetname']==fileset:selected.append(row)
    require(len(selected)==1,'exact UID/gpfs/fileset quota row required')
    row=selected[0]
    for k in ('blockUsage','blockLimit','blockInDoubt','filesUsage','filesLimit','filesInDoubt'):require(row[k].isdigit(),'mmlsquota integer units')
    require(row['blockGrace'] not in ('expired','none expired') and row['filesGrace'] not in ('expired','none expired'),'quota block/inode grace expired')
    require(int(row['blockLimit'])>0 and int(row['filesLimit'])>0,'finite hard quota required')
    require(int(row['filesUsage'])+int(row['filesInDoubt'])<int(row['filesLimit'])-65536,'quota inode reserve exhausted')
    # IBM -Y returns KB irrespective of --block-size. In-doubt consumes headroom.
    return {'hard_bytes':int(row['blockLimit'])*1024,'used_bytes':(int(row['blockUsage'])+int(row['blockInDoubt']))*1024}


def quota_scope(repo,output):
    require(repo.is_relative_to(Path('/gpfs/scratch')),'only ROMEO scratch quota deployment')
    require(output.is_relative_to(repo),'quota evidence inside deployment')
    cursor=output
    while cursor!=repo:
        require(not cursor.is_symlink(),'quota evidence symlink forbidden');cursor=cursor.parent


def query_quota(repo,output):
    repo=Path(repo).resolve();output=Path(output).absolute();quota_scope(repo,output)
    output.parent.mkdir(parents=True,exist_ok=True)
    # Each refresh owns new immutable process/output evidence. The logical alias
    # is a current receipt copy pointing at that query's immutable stdout.
    queries=output.parent/(output.stem+'.queries')
    require(not queries.is_symlink(),'quota query namespace symlink forbidden')
    queries.mkdir(mode=0o700,exist_ok=True)
    query=queries/(str(time.time_ns())+'-'+str(os.getpid())+'-'+uuid.uuid4().hex)
    query.mkdir(mode=0o700);stdout=query/'stdout';stderr=query/'stderr'
    argv=['/usr/lpp/mmfs/bin/mmlsquota','-e','-u',str(os.geteuid()),'-Y','gpfs']
    receipt=bounded(argv,repo,stdout,stderr,30);atomic_json(query/'process.json',receipt)
    failure={'schema_version':1,'status':'quota_query_failed','queried_unix_ns':time.time_ns(),'execution_uid':os.geteuid(),'process_sha256':sha(query/'process.json'),'query_stdout':{'path':str(stdout.relative_to(repo)),'sha256':sha(stdout)},'query_stderr':{'path':str(stderr.relative_to(repo)),'sha256':sha(stderr)}}
    if not (type(receipt['returncode'])is int and process_ok(receipt)):
        atomic_json(query/'quota.json',failure);raise ValueError('quota query failed/cleanup unconfirmed')
    try:values=quota_rows(stdout.read_text(),os.geteuid(),'scratch')
    except (ValueError,KeyError) as error:
        atomic_json(query/'quota.json',{**failure,'status':'quota_parse_rejected','reason':str(error)});raise
    result={'schema_version':1,'status':'current_quota_query_verified','queried_unix_ns':time.time_ns(),'execution_uid':os.geteuid(),'filesystem_root':'/gpfs/scratch',**values,'reserve_bytes':512*1024**2,'query_stdout':{'path':str(stdout.relative_to(repo)),'sha256':sha(stdout)}}
    atomic_json(query/'quota.json',result);atomic_json(output,result);return result

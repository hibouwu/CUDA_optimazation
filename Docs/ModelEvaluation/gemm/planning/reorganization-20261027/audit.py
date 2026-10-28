"""只读盘点；不跟随目录软链接，不执行实验，不改 results。"""
from pathlib import Path
import concurrent.futures, csv, hashlib, json, os, re, time

ROOT = Path(__file__).resolve().parents[5]
OUT = Path('/data/jianyeshi/cuda-reorganization-audit-20261027')
RESULTS = ROOT / 'results'
ACCESS = RESULTS / 'gh200_resource_campaign/access_rules'

def write_tsv(name, fields, rows):
    with (OUT / name).open('w') as f:
        w = csv.DictWriter(f, fields, delimiter='\t'); w.writeheader(); w.writerows(rows)

def digest(path):
    try:
        before = path.stat()
        h = hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(chunk)
        after = path.stat()
        return dict(path=str(path.relative_to(ROOT)), bytes=after.st_size,
                    sha256=h.hexdigest(), mtime_ns=after.st_mtime_ns,
                    stable=before.st_size == after.st_size and before.st_mtime_ns == after.st_mtime_ns)
    except OSError as e:
        return dict(path=str(path.relative_to(ROOT)), bytes='', sha256='', mtime_ns='', stable=str(e))

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.time(); files = []; links = []; dirs = []
    for base, ds, fs in os.walk(RESULTS, followlinks=False):
        ds.sort(); fs.sort(); dirs.append(str(Path(base).relative_to(ROOT)))
        for name in ds + fs:
            p = Path(base) / name
            if p.is_symlink():
                links.append(dict(path=str(p.relative_to(ROOT)), target=os.readlink(p),
                                  exists=p.exists(), resolved=str(p.resolve())))
            elif p.is_file(): files.append(p)
    write_tsv('symlinks.tsv', ['path','target','exists','resolved'], links)
    (OUT/'directories.json').write_text(json.dumps(dirs, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'physical_files':len(files),'symlinks':len(links),'phase':'hash'},ensure_ascii=False), flush=True)
    total = 0; unstable = []; count = 0
    with (OUT/'results-sha256.tsv').open('w') as f:
        w=csv.DictWriter(f,['path','bytes','sha256','mtime_ns','stable'],delimiter='\t');w.writeheader()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            for row in pool.map(digest,files):
                w.writerow(row); count+=1
                if isinstance(row['bytes'],int): total+=row['bytes']
                if row['stable'] is not True: unstable.append(row)
                if count%20000==0: print(json.dumps({'hashed':count,'bytes':total}),flush=True)
    # Inventory and conservative review flags. Names are discovery hints, not freeze validation.
    flags=[]; broken=[]; readmes=0
    moved={}
    with (ACCESS/'MOVED-PATHS-20261020.tsv').open() as f:
        for r in csv.DictReader(f,delimiter='\t'): moved[r['old']]=r['new']
    for p in files:
        if ACCESS not in p.parents: continue
        rel=p.relative_to(ACCESS); name=p.name.lower()
        if any(k in name for k in ('manifest','freeze','frozen','binding')):
            flags.append(dict(path=str(p.relative_to(ROOT)),kind='freeze_or_identity_candidate',detail='需按文件格式核对引用；不是已通过校验'))
        if p.suffix in ('.py','.sh') and p.stat().st_size<2_000_000:
            s=p.read_text(errors='replace')
            for i,line in enumerate(s.splitlines(),1):
                if re.search(r'parents\s*\[\s*\d+\s*\]|(?:\.\./){3,}|flat_layout_view',line):
                    flags.append(dict(path=str(p.relative_to(ROOT)),kind='path_dependency',detail=f'{i}: {line.strip()[:220]}'))
        if p.suffix.lower()!='.md': continue
        readmes+=1
        s=p.read_text(errors='replace')
        # Inline Markdown links only; reference links, HTML, code paths and fragments are outside this count.
        s=re.sub(r'```.*?```','',s,flags=re.S)
        for m in re.finditer(r'\]\(([^\s)]+)(?:\s+[^)]*)?\)',s):
            target=m.group(1).strip('<>'); target=target.split('#')[0]
            if not target or re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:',target):continue
            from urllib.parse import unquote
            target=unquote(target)
            dest=Path(target) if target.startswith('/') else p.parent/target
            if dest.exists():continue
            candidate=''
            for part in Path(target).parts:
                if part in moved:
                    suffix=target.split(part,1)[1].lstrip('/')
                    guess=ACCESS/moved[part]/suffix
                    if guess.exists(): candidate=str(guess.relative_to(ROOT))
                    break
            broken.append(dict(source=str(p.relative_to(ROOT)),target=target,mapped_existing=candidate))
    write_tsv('path-and-freeze-review.tsv',['path','kind','detail'],flags)
    write_tsv('broken-inline-links.tsv',['source','target','mapped_existing'],broken)
    summary=dict(scope='results physical files; directory symlink targets excluded',files=count,bytes=total,
                 symlinks=len(links),unstable=unstable,markdown_files_scanned=readmes,
                 broken_inline_links=len(broken),mapped_existing=sum(bool(r['mapped_existing']) for r in broken),
                 elapsed_seconds=round(time.time()-start,2),backup_created=False,
                 freeze_validation='not performed; conservative candidates only')
    (OUT/'audit-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':main()

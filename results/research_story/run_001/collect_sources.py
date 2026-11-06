"""Read-only evidence collection pinned to the completed ANN-long commit."""
import concurrent.futures
import hashlib
import json
import subprocess
import urllib.parse
import urllib.request
import zipfile
import shutil
from pathlib import Path

ROOT=Path(__file__).parent
REPO=Path('/workspace/personal')
REV='ad8c537edc6f0084d07a052bf3ab0c3ec9ceb2e8'


def main():
    tree=subprocess.check_output(['git','ls-tree','-r',REV],cwd=REPO,text=True)
    expected={v.split('\t',1)[1]:v.split('\t',1)[0].split()[2] for v in tree.splitlines()}
    requested=json.loads((ROOT/'sources/requested_files.json').read_text())
    def fetch(path):
        candidates=[REPO/path,Path('/workspace/raw_history_audit/run_001/sources')/path]
        raw=None
        for candidate in candidates:
            if candidate.is_file():
                data=candidate.read_bytes()
                blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
                if blob==expected[path]:raw=data;break
        if raw is None:
            url='https://raw.githubusercontent.com/rom-cfd/personal/'+REV+'/'+urllib.parse.quote(path)
            with urllib.request.urlopen(url,timeout=90) as response:raw=response.read()
        blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        if blob!=expected[path]:raise ValueError('Source blob mismatch: '+path)
        out=ROOT/'sources/evidence'/path;out.parent.mkdir(parents=True,exist_ok=True)
        out.write_bytes(raw)
        return {'path':path,'bytes':len(raw),'git_blob_sha':blob,'sha256':hashlib.sha256(raw).hexdigest()}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:items=list(pool.map(fetch,requested))
    (ROOT/'sources/evidence_manifest.json').write_text(json.dumps(items,indent=2))
    print('Evidence verified',len(items),'files',sum(v['bytes'] for v in items),'bytes')
    fonts=[('NanumGothic-Regular.ttf','https://raw.githubusercontent.com/google/fonts/main/ofl/nanumgothic/NanumGothic-Regular.ttf'),
           ('NanumGothic-Bold.ttf','https://raw.githubusercontent.com/google/fonts/main/ofl/nanumgothic/NanumGothic-Bold.ttf'),
           ('OFL.txt','https://raw.githubusercontent.com/google/fonts/main/ofl/nanumgothic/OFL.txt')]
    for name,url in fonts:
        with urllib.request.urlopen(url,timeout=90) as response:(ROOT/'fonts'/name).write_bytes(response.read())
    with zipfile.ZipFile(ROOT/'sources/evidence.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in requested:z.write(ROOT/'sources/evidence'/path,path)
    shutil.rmtree(ROOT/'sources/evidence')
    print('Korean TrueType fonts and license collected')


if __name__=='__main__':main()

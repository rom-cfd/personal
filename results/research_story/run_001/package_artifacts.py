"""Package the presentation without duplicating the large original study archive."""
import hashlib
import json
import zipfile
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT=Path(__file__).parent


def main():
    for license_file in (ROOT/'fonts').glob('*.txt'):
        # Preserve license wording while using LF and removing whitespace-only EOF lines.
        text=license_file.read_text()
        license_file.write_text('\n'.join(line.rstrip() for line in text.splitlines()).rstrip()+'\n')
    verification=json.loads((ROOT/'verification.json').read_text())
    assert verification['status']=='passed'
    files=[ROOT/'slides/hydrogen_research_story.pdf',ROOT/'slides/hydrogen_research_story.pptx',ROOT/'gallery/index.html']
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    complete=dict(status='passed',created_KST=datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),source_revision=verification['source_revision'],
        main_revision=verification['main_revision'],verification_sha256=sha(ROOT/'verification.json'),outputs={str(p.relative_to(ROOT)):sha(p) for p in files})
    (ROOT/'complete.json').write_text(json.dumps(complete,ensure_ascii=False,indent=2))
    request=dict(scope=['results/research_story/run_001'],branch='bw',preserve_remote_tree=True,excluded=['__pycache__','*.pyc'],
                 source_revision=verification['source_revision'],intent='Publish research narrative, editable slides, curated figures, evidence and searchable original figure index; preserve original experiments.')
    (ROOT/'publication_request.json').write_text(json.dumps(request,ensure_ascii=False,indent=2))
    archive=ROOT/'hydrogen_research_package.zip'
    paths=sorted(p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc' and p not in [archive,ROOT/'SHA256SUMS.txt'])
    (ROOT/'SHA256SUMS.txt').write_text(''.join(f'{sha(p)}  {p.relative_to(ROOT)}\n' for p in paths),encoding='utf-8')
    paths.append(ROOT/'SHA256SUMS.txt')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in paths:z.write(path,Path('hydrogen_research_story')/path.relative_to(ROOT))
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    assert archive.stat().st_size<100*1024**2
    print('Packaged files:',len(paths),'ZIP bytes:',archive.stat().st_size,'SHA256:',sha(archive))


if __name__=='__main__':main()

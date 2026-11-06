"""Download only saved evidence, verifying each file against its Git blob SHA."""
import argparse
import concurrent.futures
import hashlib
import json
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent
REPO = Path('/workspace/personal')
REVISION = '53df7a5a54b1373110269a98a8d6b85e93763aac'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--list', default=str(ROOT / 'planned_sources.json'))
    args = parser.parse_args()
    paths = json.loads(Path(args.list).read_text())
    tree = subprocess.check_output(['git', 'ls-tree', '-r', REVISION], cwd=REPO, text=True)
    expected = {line.split('\t', 1)[1]: line.split('\t', 1)[0].split()[2] for line in tree.splitlines()}

    def fetch(path):
        target = ROOT / 'sources' / path
        local = REPO / path
        raw = target.read_bytes() if target.exists() else local.read_bytes() if local.exists() else None
        source = 'cached/local'
        if raw is None:
            url = 'https://raw.githubusercontent.com/rom-cfd/personal/' + REVISION + '/' + urllib.parse.quote(path)
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(url, timeout=90) as response:
                        raw = response.read()
                    source = 'github raw; app cannot return large CSV/binary bodies'
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(1)
        blob = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
        if blob != expected[path]:
            raise ValueError('Git blob mismatch: ' + path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(raw)
        return dict(path=path, bytes=len(raw), git_blob_sha=blob, source=source)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(fetch, paths))
    name = Path(args.list).stem + '_downloaded.json'
    (ROOT / name).write_text(json.dumps(results, indent=2))
    print('Verified files:', len(results), 'bytes:', sum(r['bytes'] for r in results))


if __name__ == '__main__':
    main()

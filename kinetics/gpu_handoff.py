"""Build/restore hashed, split ZIP payloads using the Python standard library."""
import argparse,hashlib,json,os,tempfile,zipfile
from pathlib import Path,PurePosixPath


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def safe_path(root,name):
    relative=PurePosixPath(name)
    if relative.is_absolute() or '..' in relative.parts:raise ValueError('Invalid archive path')
    target=root.joinpath(*relative.parts)
    if not target.resolve().is_relative_to(root.resolve()):raise ValueError('Path escapes destination')
    return target


def build(root,output,paths,chunk_bytes=40*1024*1024):
    root=Path(root).resolve();output=Path(output);output.mkdir(parents=True,exist_ok=False)
    files={str(p):dict(bytes=(root/p).stat().st_size,sha256=digest(root/p)) for p in paths}
    with tempfile.TemporaryDirectory() as tmp:
        archive=Path(tmp)/'training.zip'
        with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
            for name in files:z.write(root/name,name)
        parts=[]
        with archive.open('rb') as f:
            while block:=f.read(chunk_bytes):
                name=f'payload-{len(parts):03d}.zip.part';(output/name).write_bytes(block)
                parts.append(dict(path=name,bytes=len(block),sha256=digest(output/name)))
        manifest=dict(format='split-zip-v1',archive_bytes=archive.stat().st_size,
            archive_sha256=digest(archive),files=files,parts=parts)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def restore(payload,root,verify_only=False):
    payload=Path(payload);root=Path(root).resolve();manifest=json.loads((payload/'manifest.json').read_text())
    assert manifest['format']=='split-zip-v1'
    if verify_only:
        for name,entry in manifest['files'].items():
            path=safe_path(root,name)
            assert path.is_file() and path.stat().st_size==entry['bytes'] and digest(path)==entry['sha256'],name
        return manifest
    with tempfile.TemporaryDirectory() as tmp:
        archive=Path(tmp)/'training.zip'
        with archive.open('wb') as out:
            for part in manifest['parts']:
                path=safe_path(payload,part['path'])
                assert path.stat().st_size==part['bytes'] and digest(path)==part['sha256'],part['path']
                with path.open('rb') as f:
                    for block in iter(lambda:f.read(4*1024*1024),b''):out.write(block)
        assert archive.stat().st_size==manifest['archive_bytes'] and digest(archive)==manifest['archive_sha256']
        with zipfile.ZipFile(archive) as z:
            assert set(z.namelist())==set(manifest['files'])
            for name,entry in manifest['files'].items():
                target=safe_path(root,name)
                if target.exists():
                    assert digest(target)==entry['sha256'],f'Existing file differs; preserve it and use a fresh clone: {name}'
                    continue
                target.parent.mkdir(parents=True,exist_ok=True);temp=target.with_suffix(target.suffix+'.restoring')
                h=hashlib.sha256()
                try:
                    with z.open(name) as source,temp.open('wb') as dest:
                        for block in iter(lambda:source.read(4*1024*1024),b''):dest.write(block);h.update(block)
                    assert temp.stat().st_size==entry['bytes'] and h.hexdigest()==entry['sha256'],name
                    os.replace(temp,target)
                finally:
                    if temp.exists():temp.unlink()
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['restore','verify'])
    parser.add_argument('--payload',type=Path,default=Path('handoff/zhang_1000_gpu'))
    parser.add_argument('--root',type=Path,default=Path('.'))
    args=parser.parse_args();m=restore(args.payload,args.root,args.action=='verify')
    print(f'{args.action.upper()} OK: {len(m["files"])} files, all SHA256 hashes match')


if __name__=='__main__':main()

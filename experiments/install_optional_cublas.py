#!/usr/bin/env python3
"""Install official cuBLAS 12.8.4.1 shared libraries in the local CUDA toolkit."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
import urllib.request

BASE = 'https://developer.download.nvidia.com/compute/cuda/redist/'
ARCHIVE = 'libcublas/linux-x86_64/libcublas-linux-x86_64-12.8.4.1-archive.tar.xz'
SHA256 = '21718957c2cf000bacd69d36c95708a2319199e39e056f8b4f0f68e3b9f323bb'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cuda-root', type=Path,
                        default=Path(__file__).resolve().parents[1]/'.tools/cuda-12.8')
    args = parser.parse_args()
    if not (args.cuda_root/'bin/nvcc').is_file():
        raise RuntimeError('Install the local CUDA 12.8 toolkit first or supply --cuda-root')
    with tempfile.TemporaryDirectory(prefix='5206-cublas-') as temporary:
        package = Path(temporary)/'cublas.tar.xz'
        print('Downloading official cuBLAS archive (~939 MB); shared libraries and headers only will be installed.', flush=True)
        previous = -1
        def progress(block, size, total):
            nonlocal previous
            mib = block*size//(128*1024*1024)
            if mib != previous:
                print(f'Downloaded {block*size//(1024*1024)} MiB', flush=True)
                previous = mib
        urllib.request.urlretrieve(BASE+ARCHIVE, package, progress)
        digest = hashlib.sha256()
        with package.open('rb') as stream:
            for chunk in iter(lambda:stream.read(4*1024*1024), b''):
                digest.update(chunk)
        if digest.hexdigest() != SHA256:
            raise RuntimeError('cuBLAS archive checksum mismatch')
        print('Checksum verified; extracting shared libraries and headers.', flush=True)
        destination=Path(temporary)/'unpacked'
        with tarfile.open(package) as archive:
            members=[m for m in archive.getmembers() if
                     ('/include/' in m.name or '/lib/' in m.name and '.so' in m.name
                      or m.name.endswith('/LICENSE')) and not m.name.endswith('.a')]
            archive.extractall(destination, members=members, filter='data')
        source,=destination.iterdir()
        for name in ('include','lib'):
            shutil.copytree(source/name,args.cuda_root/name,dirs_exist_ok=True,symlinks=True)
        license_dir=args.cuda_root/'licenses/libcublas'
        license_dir.mkdir(parents=True,exist_ok=True)
        if (source/'LICENSE').exists(): shutil.copy2(source/'LICENSE',license_dir/'LICENSE')
        (license_dir/'redistrib.json').write_text(json.dumps({'url':BASE+ARCHIVE,'sha256':SHA256},indent=2)+'\n')
    print('Installed cuBLAS in',args.cuda_root,flush=True)


if __name__=='__main__':
    main()

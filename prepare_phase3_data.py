"""Download exact original training images from official ZIP byte ranges."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import struct
import time
import urllib.request
import uuid
import zipfile
import zlib

from phase3_common import checked_splits, digest, ensure_new_or_equal, safe_run, sha, verify_freeze, write_new
from prepare_phase2_local import check_image

OFFICIAL_URL = 'https://vizwiz.cs.colorado.edu/VizWiz_final/images/train.zip'


class RangeArchive(io.RawIOBase):
    """Read ZIP metadata via validated HTTP ranges, never a full archive."""
    def __init__(self, url=OFFICIAL_URL):
        self.url = url
        with urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=45) as r:
            self.size = int(r.headers['Content-Length'])
            self.etag = r.headers['ETag']
            if r.geturl() != url or r.headers.get('Accept-Ranges') != 'bytes':
                raise ValueError('Official source does not support verified partial downloads')
        self.position = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        if not 0 <= self.position <= self.size:
            raise ValueError('ZIP metadata seek outside archive')
        return self.position

    def read(self, size=-1):
        size = self.size-self.position if size < 0 else min(size, self.size-self.position)
        if not size:
            return b''
        if size > 16*1024*1024:
            raise ValueError('Unexpected full archive read')
        data = self.fetch(self.position, self.position+size-1)
        self.position += len(data)
        return data

    def fetch(self, start, end):
        end = min(end, self.size-1)
        for attempt in range(4):
            try:
                req = urllib.request.Request(self.url, headers={'Range':f'bytes={start}-{end}', 'If-Range':self.etag})
                with urllib.request.urlopen(req, timeout=60) as r:
                    expected = f'bytes {start}-{end}/{self.size}'
                    if r.status != 206 or r.headers.get('Content-Range') != expected or r.headers.get('ETag') != self.etag or r.geturl() != self.url:
                        raise ValueError('Official archive range/identity mismatch; refusing substitute data')
                    data = r.read(end-start+2)
                    if len(data) != end-start+1:
                        raise IOError('Incomplete range')
                    return data
            except (OSError, TimeoutError):
                if attempt == 3:
                    raise
                time.sleep(min(2**attempt, 8))


def member_bytes(remote, info):
    if info.flag_bits & 1 or info.compress_type not in (0, 8) or info.file_size > 40*1024*1024:
        raise ValueError('Unexpected encrypted/compression/size entry')
    # Local extra-field length is at most 65,535. One bounded range includes
    # the full selected entry; following bytes are never decoded as images.
    span = remote.fetch(info.header_offset, info.header_offset+30+len(info.filename.encode('utf-8'))+65535+info.compress_size-1)
    header = struct.unpack('<4s5H3I2H', span[:30])
    if header[0] != b'PK\x03\x04' or header[3] != info.compress_type:
        raise ValueError('Local ZIP header mismatch')
    name_len, extra_len = header[-2:]
    name = span[30:30+name_len].decode('utf-8' if info.flag_bits & 0x800 else 'cp437')
    if name != info.filename:
        raise ValueError('Archive filename mismatch')
    offset = 30+name_len+extra_len
    payload = span[offset:offset+info.compress_size]
    data = payload if info.compress_type == 0 else zlib.decompress(payload, -15)
    if len(data) != info.file_size or zlib.crc32(data) & 0xffffffff != info.CRC:
        raise ValueError('Official ZIP CRC/size mismatch')
    return data


def prepare(run_dir, workers=4):
    root = safe_run(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    verify_freeze()
    _, train, dev, payload = checked_splits()
    inputs = root/'inputs'; inputs.mkdir(exist_ok=True)
    ensure_new_or_equal(inputs/'train-only.json', {'split':'train','rows':train,'rows_sha256':digest(train),'source_manifest_sha256':payload['source_manifest_sha256']})
    ensure_new_or_equal(inputs/'dev-only.json', payload)
    images=root/'data/train-images'; ledger=root/'data/image-integrity'
    images.mkdir(parents=True,exist_ok=True); ledger.mkdir(exist_ok=True)
    remote=RangeArchive()
    with zipfile.ZipFile(remote) as z:
        matches={}
        wanted={r['image'] for r in train}
        for entry in z.infolist():
            parts=PurePosixPath(entry.filename)
            if parts.name in wanted:
                if parts.is_absolute() or '..' in parts.parts or '\\' in entry.filename or parts.name in matches:
                    raise ValueError('Unsafe or duplicate official image member')
                matches[parts.name]=entry
    if set(matches) != wanted:
        raise ValueError('Official archive lacks exact original training IDs')
    total_bytes=sum(e.file_size for e in matches.values())
    existing_bytes=sum(p.stat().st_size for p in images.glob('*.jpg'))
    if shutil.disk_usage(root).free < total_bytes-existing_bytes+2*1024**3:
        raise RuntimeError('Insufficient disk for exact training images plus 2 GiB reserve')
    source={'url':OFFICIAL_URL,'archive_bytes':remote.size,'etag':remote.etag,'selected_images':5000,'selected_image_bytes':total_bytes,
            'rows_sha256':digest(train),'members':{name:{'crc32':e.CRC,'bytes':e.file_size,'compressed_bytes':e.compress_size,'offset':e.header_offset} for name,e in matches.items()}}
    ensure_new_or_equal(root/'data/archive-identity.json',source)
    print(f'Official archive verified: exact 5,000 entries; {total_bytes/1024**3:.2f} GiB selected images; {workers} download workers',flush=True)

    def download(name):
        info=matches[name]; path=images/name; record=ledger/(name+'.json')
        if record.exists():
            saved=json.loads(record.read_text())
            if not path.is_file() or sha(path)!=saved['sha256'] or saved['crc32']!=info.CRC:
                raise ValueError(f'Existing verified image changed: {name}')
            return 'reused'
        if path.exists():
            data=path.read_bytes()
            if len(data)!=info.file_size or zlib.crc32(data)&0xffffffff!=info.CRC:
                raise ValueError(f'Unverified existing file differs: {name}; preserved')
        else:
            data=member_bytes(remote,info)
            temp=path.with_name(name+'.'+uuid.uuid4().hex+'.partial')
            with temp.open('xb') as stream: stream.write(data)
            check_image(temp)
            if path.exists(): raise FileExistsError('Concurrent image writer')
            temp.rename(path)
        check_image(path)
        write_new(record,{'image':name,'sha256':sha(path),'crc32':info.CRC,'bytes':info.file_size,'source':OFFICIAL_URL})
        return 'downloaded'

    start=time.perf_counter(); counts={'downloaded':0,'reused':0}
    names=sorted(wanted,key=lambda n:matches[n].header_offset)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures={executor.submit(download,name):name for name in names}
        for index, future in enumerate(as_completed(futures),1):
            counts[future.result()]+=1
            if index%25==0 or index==5000:
                print(f'Images {index}/5000; downloaded={counts["downloaded"]}; reused={counts["reused"]}; elapsed={(time.perf_counter()-start)/60:.1f} min',flush=True)
    ensure_new_or_equal(root/'data/prepared.json',{'images':5000,'train_rows_sha256':digest(train),'source_manifest_sha256':payload['source_manifest_sha256'],'all_readable_and_crc_verified':True,'holdout_accessed':False})
    print('EXACT_TRAINING_IMAGES_READY',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',required=True); p.add_argument('--workers',type=int,default=4)
    args=p.parse_args(); prepare(args.run_dir,args.workers)

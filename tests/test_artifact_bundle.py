import hashlib
import json
import zipfile

import pytest

from artifact_bundle import ALLOWED_FILES, info, relative, verify


@pytest.mark.parametrize('name',['../escape','/absolute','C:/absolute','models/../escape','x:stream','a//b'])
def test_unsafe_path_rejected(name):
    with pytest.raises(ValueError):relative(name)


def tiny_bundle(tmp_path):
    archive=tmp_path/'bundle.zip';data=b'unit-test bytes, not model weights'
    rows=[{'path':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()} for name in sorted(ALLOWED_FILES)]
    with zipfile.ZipFile(archive,'w') as bundle:
        bundle.writestr('artifact-manifest.json',json.dumps({'schema':1,'files':rows,'source_sha256':{}}))
        for row in rows:bundle.writestr(info('payload/'+row['path']),data)
    return archive,data


def test_install_and_resume_preserve_existing_bytes(tmp_path):
    archive,data=tiny_bundle(tmp_path);project=tmp_path/'project';project.mkdir()
    verify(archive,project,install=True)
    before={p:p.read_bytes() for p in project.rglob('*') if p.is_file()}
    verify(archive,project,install=True)
    assert all(p.read_bytes()==content==data for p,content in before.items())
    assert len(before)==21


def test_conflict_detected_before_any_payload_write(tmp_path):
    archive,_=tiny_bundle(tmp_path);project=tmp_path/'project';project.mkdir()
    existing=project/'models/blip_vizwiz_v2/vocab.txt';existing.parent.mkdir(parents=True);existing.write_bytes(b'preserve me')
    with pytest.raises(FileExistsError):verify(archive,project,install=True)
    assert existing.read_bytes()==b'preserve me'
    assert len(list(p for p in project.rglob('*') if p.is_file()))==1


def test_wrong_zip_hash_rejected(tmp_path):
    archive,_=tiny_bundle(tmp_path)
    with pytest.raises(ValueError,match='SHA256'):verify(archive,tmp_path,expected_sha256='0'*64)


def test_incomplete_or_extra_payload_rejected(tmp_path):
    archive=tmp_path/'bad.zip'
    with zipfile.ZipFile(archive,'w') as bundle:
        bundle.writestr('artifact-manifest.json',json.dumps({'schema':1,'files':[],'source_sha256':{}}))
    with pytest.raises(ValueError,match='21 reviewed'):verify(archive,tmp_path)

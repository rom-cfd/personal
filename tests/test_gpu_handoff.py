import json
import pytest
from kinetics.gpu_handoff import build,restore


def test_split_payload_roundtrip_hashes_and_existing_data_preserved(tmp_path):
    source=tmp_path/'source';source.mkdir();(source/'arrays').mkdir()
    (source/'arrays/a.bin').write_bytes(bytes(range(256))*32)
    (source/'state.json').write_text('{"step":17}')
    payload=tmp_path/'payload';m=build(source,payload,['arrays/a.bin','state.json'],chunk_bytes=100)
    assert len(m['parts'])>1
    target=tmp_path/'target';restore(payload,target);restore(payload,target,verify_only=True)
    (target/'state.json').write_text('different checkpoint')
    with pytest.raises(AssertionError):restore(payload,target)
    assert (target/'state.json').read_text()=='different checkpoint'


def test_corrupt_part_is_rejected_before_extracting(tmp_path):
    source=tmp_path/'source';source.mkdir();(source/'data').write_bytes(b'abc')
    payload=tmp_path/'payload';m=build(source,payload,['data'],chunk_bytes=10)
    part=payload/m['parts'][0]['path'];part.write_bytes(b'corrupted')
    with pytest.raises(AssertionError):restore(payload,tmp_path/'target')
    assert not (tmp_path/'target/data').exists()

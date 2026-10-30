"""Require an NVIDIA TensorFlow GPU, then resume the unchanged training program."""
import argparse,json,sys
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true',help='Check GPU and restored payload without training')
    parser.add_argument('--run-updates',type=int,help='Optional bounded test; checkpoint and pause after N updates')
    args=parser.parse_args()
    from .gpu_handoff import restore,digest,safe_path
    payload=Path('handoff/zhang_1000_gpu');manifest=json.loads((payload/'manifest.json').read_text())
    receipt=Path('results/zhang_5p3m_gelu_1000/g0.1/gpu_handoff_receipt.json')
    if receipt.exists():
        assert json.loads(receipt.read_text())['manifest_sha256']==digest(payload/'manifest.json')
        for name,entry in manifest['files'].items():
            if name.startswith('results/zhang_5p3m_gelu_1000/g0.1/seed11/'):continue
            assert digest(safe_path(Path('.'),name))==entry['sha256'],name
    else:
        try:restore(payload,Path('.'),verify_only=True)
        except AssertionError as error:
            raise SystemExit(f'Handoff payload not restored ({error}). Run first: python -m kinetics.gpu_handoff restore') from None
    import tensorflow as tf
    if tf.__version__!='2.20.0':raise RuntimeError('Install tensorflow[and-cuda]==2.20.0 in a clean environment')
    devices=tf.config.list_physical_devices('GPU')
    if not devices:raise RuntimeError('No TensorFlow GPU detected. Check NVIDIA driver and GPU-enabled TensorFlow; CPU training was not started.')
    for device in devices:tf.config.experimental.set_memory_growth(device,True)
    if not receipt.exists():receipt.write_text(json.dumps(dict(manifest_sha256=digest(payload/'manifest.json'))))
    print('GPU_READY',devices,flush=True)
    progress=json.loads(Path('results/zhang_5p3m_gelu_1000/g0.1/seed11/checkpoint.json').read_text())
    print('RESUME',progress['completed_epochs'],'epochs; target=1000; batch=512',flush=True)
    if args.check:return
    # Restore is verified only at takeover. Later checkpoints legitimately differ.
    sys.argv=['large_zhang_continue','--config','configs/zhang_5p3m_gelu_1000.json','--resume']
    if args.run_updates:sys.argv+=['--run-updates',str(args.run_updates)]
    from .large_zhang_continue import main as train
    train()


if __name__=='__main__':main()

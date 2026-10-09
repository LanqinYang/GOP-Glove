"""Check recording identities, partition isolation and descriptor dimensions."""
import ast
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from reproducibility.splits import partition_indices
from src.training.manuscript_gate import confidence_gated_predict

def main():
    with (ROOT/'reproducibility/dataset_manifest.csv').open(newline='') as stream:
        recordings=list(csv.DictReader(stream))
    assert len(recordings)==660
    for r in recordings:
        assert hashlib.sha256((ROOT/r['relative_path']).read_bytes()).hexdigest()==r['sha256']
    cells=Counter((r['participant_id'],r['class_id']) for r in recordings)
    assert len(cells)==66 and set(cells.values())=={10}
    files=[p.name for p in sorted((ROOT/'datasets/gesture_csv').glob('*.csv'))]
    folder=ROOT/'reproducibility/generated'
    assert tuple(map(len,partition_indices(folder/'iid_seed42.csv',files,'IID')))==(422,106,132)
    for fold in range(1,7):
        assert tuple(map(len,partition_indices(folder/'loso_seed42.csv',files,fold)))==(440,110,110)
    features=np.load(folder/'manuscript_unscaled_features.npy')
    assert features.shape==(660,190) and np.isfinite(features).all()
    a=np.full((4,11),0.025,dtype=np.float32);b=a.copy()
    a[0,1]=b[0,1]=0.75
    a[1,2]=0.75;b[1]=1/11
    a[2]=1/11;b[2,3]=0.75
    a[3]=b[3]=1/11
    assert confidence_gated_predict(a,b).tolist()==[1,2,3,10]
    sources=[p for directory in ('src','experiments','scripts','reproducibility') for p in (ROOT/directory).rglob('*.py')]
    for p in sources: ast.parse(p.read_text(),filename=str(p.relative_to(ROOT)))
    print(json.dumps({'dataset_files':660,'subject_class_cells':66,'loso_folds':6,
                      'feature_shape':[660,190],'python_sources':len(sources),'checks':'passed'},indent=2))

if __name__=='__main__': main()

"""Test external validation independently with injected output errors."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory() as temporary:
    folder=Path(temporary); inp=folder/'input'; out=folder/'out'
    np.array([[-1,0],[1,0],[0,0]],dtype='<f4').tofile(str(inp)+'.X.f32')
    np.array([[0,0]],dtype='<f4').tofile(str(inp)+'.Q.f32')
    np.array([10,20,30],dtype='<i4').tofile(str(inp)+'.labels.i32')
    np.array([1,1,0],dtype='<f4').tofile(folder/'full.f32')
    def run(ids,distances,labels):
        np.array(ids,dtype='<u4').tofile(str(out)+'.ids.u32')
        np.array(distances,dtype='<f4').tofile(str(out)+'.distances.f32')
        np.array(labels,dtype='<i4').tofile(str(out)+'.labels.i32')
        cmd=[sys.executable,str(root/'scripts/verify_external.py'),'--input',str(inp),'--output',str(out),
             '--n','3','--d','2','--q','1','--k','2','--full-distances',str(folder/'full.f32'),
             '--repeat-output',str(out),'--report',str(folder/'report.json')]
        p=subprocess.run(cmd,capture_output=True,text=True)
        return p.returncode,json.loads((folder/'report.json').read_text())
    code,report=run([2,0],[0,1],[30,10]);assert code==0 and report['status']=='strict_agreement'
    code,report=run([2,0],[0,1],[30,20]);assert code!=0 and not report['labels']
    code,report=run([2,2],[0,0],[30,30]);assert code!=0 and not report['structure']
    code,report=run([2,1],[0,1],[30,20]);assert code!=0 and not report['fixed_distance_selection']
    code,report=run([2,0],[-0.0,1],[30,10]);assert code!=0 and not report['structure']
    code,report=run([2,0],[float('nan'),1],[30,10]);assert code!=0 and not report['distance_tolerance']
print('PASS external verifier: valid output and five injected failures')

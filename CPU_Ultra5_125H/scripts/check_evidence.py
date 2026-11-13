#!/usr/bin/env python3
"""Check emitted instructions and compilation settings; save auditable evidence."""
import json,re,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[1]
evidence={}
for name in ('scalar','simd'):
    path=root/'build/CMakeFiles/knn.dir/src'/f'{name}.cpp.o'
    text=subprocess.check_output(['objdump','-d','-C',str(path)],text=True)
    (root/'results'/('scalar_instructions.txt' if name=='scalar' else 'avx2_instructions.txt')).write_text(text)
    ops=re.findall(r'\t([a-z][a-z0-9]+)\s',text)
    assert not any('fmadd' in op or 'fmsub' in op for op in ops)
    if name=='scalar':
        assert all(op in ops for op in ('subss','mulss','addss'))
        assert not any(op in ('addps','subps','mulps','divps','vaddps','vsubps','vmulps','vdivps') for op in ops)
    else:
        assert all(op in ops for op in ('vsubps','vmulps','vaddps'))
    evidence[name]={'no_FMA':True,'checked_instructions':sorted(set(ops))}
commands=json.loads((root/'build/compile_commands.json').read_text())
for item in commands:
    assert '-ffp-contract=off' in item['command'] and '-fno-fast-math' in item['command']
    if item['file'].endswith(('scalar.cpp','simd.cpp','oracle.cpp')):
        assert '-fno-tree-vectorize' in item['command'] and '-fno-tree-slp-vectorize' in item['command']
evidence['controlled_build_flags']=True
(root/'results/instruction_checks.json').write_text(json.dumps(evidence,indent=2)+'\n')
print('PASS controlled floating point flags, scalar instructions, AVX2 and absence of FMA')

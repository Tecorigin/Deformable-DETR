"""Qualify two binaries, then compare warmed full attention forward/backward costs."""
import hashlib
import json
import runpy
import statistics
import sys
import time
from pathlib import Path

import torch
import torch_sdaa
from native_attention import NativeAttention

out=Path(sys.argv[1])
result={'passed':False,'scope':'Isolated complete attention forward/backward; not official training imgs/s or score',
        'warmup':5,'repetitions':15,'numerics':{},'cases':[]}
bindings={}
try:
    for label in ('baseline','candidate'):
        child=out/label;child.mkdir()
        source=(out/(label+'_deform_attention.scpp')).read_bytes()
        (child/'native_deform_attention.scpp').write_bytes(source)
        sys.argv=[sys.argv[0],str(child)]
        runpy.run_path(str(out/'native_attention_probe.py'),run_name='__main__')
        qualification=json.loads((child/'result.json').read_text())
        result['numerics'][label]=qualification
        if not qualification['passed']:raise RuntimeError('Numerical gate failed: '+label)
        bindings[label]=NativeAttention(child/'libdetr_attention.so')
    torch.manual_seed(20261008)
    for name,b,sizes,h,c,q,p in (
            ('encoder',1,[(23,29)],8,32,667,4),
            ('decoder',2,[(23,29)],8,32,300,4),
            ('larger_encoder',1,[(30,40)],8,32,1200,4)):
        spatial=sum(y*x for y,x in sizes);l=len(sizes)
        v=torch.randn(b,spatial,h,c,device='sdaa:0',requires_grad=True)
        loc=torch.rand(b,q,h,l,p,2,device='sdaa:0',requires_grad=True)
        w=torch.rand(b,q,h,l,p,device='sdaa:0',requires_grad=True)
        grad=torch.randn(b,q,h*c,device='sdaa:0')
        shape=torch.tensor(sizes);starts=torch.tensor([0])
        row={'name':name,'batch':b,'spatial':spatial,'queries':q,'timings':{}}
        result['cases'].append(row)
        samples={label:[] for label in bindings}
        for repetition in range(result['warmup']+result['repetitions']):
            order=('baseline','candidate') if repetition%2==0 else ('candidate','baseline')
            for label in order:
                torch.sdaa.synchronize();begin=time.perf_counter()
                output=bindings[label].apply(v,shape,starts,loc,w)
                gradients=torch.autograd.grad(output,(v,loc,w),grad)
                torch.sdaa.synchronize();elapsed=time.perf_counter()-begin
                if repetition>=result['warmup']:samples[label].append(elapsed)
        for label,values in samples.items():
            row['timings'][label]={'seconds':values,'median':statistics.median(values),'min':min(values),'max':max(values)}
        row['speedup']=row['timings']['baseline']['median']/row['timings']['candidate']['median']
        # Both implementations must agree on new benchmark input, not merely the gate inputs.
        outputs={};grads={}
        for label,binding in bindings.items():
            y=binding.apply(v,shape,starts,loc,w)
            grads[label]=[g.detach().cpu() for g in torch.autograd.grad(y,(v,loc,w),grad)]
            outputs[label]=y.detach().cpu()
        row['candidate_to_baseline_numerics']={}
        for label,a,bv in zip(('output','value_gradient','location_gradient','weight_gradient'),
                              [outputs['candidate'],*grads['candidate']],[outputs['baseline'],*grads['baseline']]):
            error=(a-bv).abs();limit=2e-5+2e-5*bv.abs()
            row['candidate_to_baseline_numerics'][label]={'passed':bool(torch.isfinite(a).all() and (error<=limit).all()),
                                                         'max_abs_error':float(error.max())}
        if not all(c['passed'] for c in row['candidate_to_baseline_numerics'].values()):
            raise RuntimeError('Benchmark input numerical mismatch')
        (out/'result.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(row),flush=True)
    result['passed']=True
except BaseException as error:
    result['error']=type(error).__name__+': '+str(error)
    raise
finally:
    (out/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)

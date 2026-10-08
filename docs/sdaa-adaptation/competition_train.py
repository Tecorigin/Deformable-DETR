"""Isolated official-model VOC fine-tuning with fixed 20 epochs and native attention.

Preparation is CPU-only. Training requires a complete native numerical record
bound to the exact shared library. Resume preserves the fine-tuned class head.
"""
import argparse
import hashlib
import json
import math
import random
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, BatchSampler

PRETRAINED_SHA256='d442fb2365d6e9640347b2b38686089d13cd55a5c3790d63e3602d079791fed1'
CASE_NAMES={'single_scale','multi_scale_channel_tail','many_levels','encoder_shape','decoder_shape'}


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda:file.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def write_json(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(path)


def qualified_library(library,qualification):
    record=json.loads(Path(qualification).read_text())
    if not record.get('passed') or {c['name'] for c in record.get('cases',[])}!=CASE_NAMES:
        raise RuntimeError('Complete five-case native numerical record required')
    for case in record['cases']:
        if not case.get('passed') or case.get('sampling_cpu_fallback',True):
            raise RuntimeError('Native numerical or execution-path gate failed')
        comparisons=case.get('comparisons',{})
        if set(comparisons)!={'output','value_gradient','location_gradient','weight_gradient'} or not all(
                c.get('passed') for c in comparisons.values()):
            raise RuntimeError('Complete forward and gradient comparisons required')
    if not 0<record.get('atol',float('inf'))<=2e-5 or not 0<record.get('rtol',float('inf'))<=2e-5:
        raise RuntimeError('Native numerical tolerance exceeds the frozen gate')
    if record.get('binary_sha256')!=digest(library):
        raise RuntimeError('Native binary differs from its numerical qualification')
    return record


def seed_worker(worker_id):
    seed=torch.initial_seed()%2**32
    random.seed(seed);np.random.seed(seed);torch.set_num_threads(1)


def forbidden_extension(*args):
    raise RuntimeError('Unported CUDA attention extension cannot execute')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=('prepare','train','evaluate'),default='prepare')
    parser.add_argument('--native-library',type=Path)
    parser.add_argument('--qualification',type=Path)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--batch-size',type=int,default=2)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--seed',type=int,default=20261007)
    parser.add_argument('--lr',type=float,default=2e-5)
    parser.add_argument('--lr-backbone',type=float,default=2e-6)
    parser.add_argument('--evaluate-every',type=int,default=5)
    options=parser.parse_args()
    if options.batch_size<1 or options.workers<0 or options.evaluate_every<1:
        parser.error('Invalid batch, worker or evaluation interval')
    if options.mode!='prepare' and (not options.native_library or not options.qualification):
        parser.error('Device work requires a library and complete numerical qualification')
    if options.mode=='evaluate' and not options.resume:
        parser.error('Evaluation requires the trained checkpoint')
    project=options.project.resolve();output=options.output.resolve();output.mkdir(parents=True,exist_ok=True)
    if options.mode=='train' and not options.resume and (output/'checkpoint.pth').exists():
        raise RuntimeError('Existing training checkpoint requires explicit resume or a new output directory')
    dataset_path=project/'data/voc_coco'
    pretrained_path=project/'r50_deformable_detr_single_scale-checkpoint.pth'
    if digest(pretrained_path)!=PRETRAINED_SHA256:
        raise RuntimeError('Specified pretrained checkpoint identity differs')
    annotation_hashes={split:digest(dataset_path/split/(split+'.json')) for split in ('train','val')}
    random.seed(options.seed);np.random.seed(options.seed);torch.manual_seed(options.seed);torch.set_num_threads(4)
    device='cpu' if options.mode=='prepare' else 'sdaa:0'
    qualification=None;binding=None
    if options.mode!='prepare':
        qualification=qualified_library(options.native_library,options.qualification)
        import torch_sdaa
        from native_attention import NativeAttention
        torch.sdaa.manual_seed_all(options.seed)
        binding=NativeAttention(options.native_library)
    sys.path.insert(0,str(project))
    extension=types.ModuleType('MultiScaleDeformableAttention')
    extension.ms_deform_attn_forward=forbidden_extension;extension.ms_deform_attn_backward=forbidden_extension
    sys.modules[extension.__name__]=extension
    import models.ops.modules.ms_deform_attn as module
    if binding:
        module.MSDeformAttnFunction=binding
    from main import get_args_parser
    from models import build_model
    from datasets import build_dataset,get_coco_api_from_dataset
    import util.misc as utils
    args=get_args_parser().parse_args(['--device',device,'--dataset_file','coco','--coco_path',str(dataset_path),
        '--num_classes','21','--num_feature_levels','1','--epochs','20','--batch_size',str(options.batch_size),
        '--lr',str(options.lr),'--lr_backbone',str(options.lr_backbone),'--lr_drop','14',
        '--clip_max_norm','0.1','--weight_decay','1e-4','--seed',str(options.seed)])
    model,criterion,postprocessors=build_model(args)
    if options.mode!='prepare':
        model.to(device);criterion.to(device)
    train_dataset=build_dataset('train',args);validation_dataset=build_dataset('val',args)
    if len(train_dataset)!=5011 or len(validation_dataset)!=4952:
        raise RuntimeError('Official complete dataset lengths differ')
    parameters=list(model.named_parameters())
    groups=[{'params':[p for n,p in parameters if p.requires_grad and not any(
                word in n for word in args.lr_backbone_names+args.lr_linear_proj_names)],'lr':args.lr},
            {'params':[p for n,p in parameters if p.requires_grad and any(word in n for word in args.lr_backbone_names)],'lr':args.lr_backbone},
            {'params':[p for n,p in parameters if p.requires_grad and any(word in n for word in args.lr_linear_proj_names)],'lr':args.lr*args.lr_linear_proj_mult}]
    grouped=[id(p) for group in groups for p in group['params']]
    trainable=[id(p) for _,p in parameters if p.requires_grad]
    if len(set(grouped))!=len(grouped) or set(grouped)!=set(trainable):
        raise RuntimeError('Optimizer groups must cover every trainable parameter exactly once')
    trained_checkpoint=None;first_epoch=0
    identity={'pretrained_sha256':PRETRAINED_SHA256,'annotation_sha256':annotation_hashes,
              'seed':options.seed,'batch_size':options.batch_size,'workers':options.workers,
              'lr':options.lr,'lr_backbone':options.lr_backbone,
              'epochs':20,'parameters':sum(p.numel() for p in model.parameters()),
              'native_binary_sha256':qualification['binary_sha256'] if qualification else None}
    if options.resume:
        trained_checkpoint=torch.load(options.resume,map_location='cpu',weights_only=False)
        if trained_checkpoint.get('identity')!=identity:
            raise RuntimeError('Resume training identity differs; do not mix builds or settings')
        model.load_state_dict(trained_checkpoint['model'],strict=True)
        first_epoch=trained_checkpoint['completed_epochs']
        if not 0<=first_epoch<=20:
            raise RuntimeError('Invalid completed epoch count')
    else:
        checkpoint=torch.load(pretrained_path,map_location='cpu',weights_only=False)
        original=checkpoint['model'];removed={key for key in original if 'class_embed' in key}
        missing,unexpected=model.load_state_dict({k:v for k,v in original.items() if k not in removed},strict=False)
        if set(missing)!=removed or unexpected:
            raise RuntimeError('Unexpected initial fine-tuning parameter mismatch')
        del checkpoint,original
    manifest={**identity,'mode':options.mode,'project':str(project),'output':str(output),
              'workers':options.workers,'train_images':len(train_dataset),'validation_images':len(validation_dataset),
              'optimizer_group_sizes':[sum(p.numel() for p in group['params']) for group in groups],
              'trainable_parameters':sum(p.numel() for _,p in parameters if p.requires_grad),
              'optimizer_parameter_coverage_verified':True,
              'drop_last':True,'completed_epochs':first_epoch,'official_accuracy_measured':False,
              'device_work_submitted':options.mode!='prepare','formal_score_awarded':False}
    manifest['class_head_sha256']={name:hashlib.sha256(parameter.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        for name,parameter in model.named_parameters() if 'class_embed' in name}
    write_json(output/'manifest.json',manifest)
    if options.mode=='prepare':
        print(json.dumps(manifest,indent=2),flush=True);return
    optimizer=torch.optim.AdamW(groups,lr=args.lr,weight_decay=args.weight_decay)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,args.lr_drop)
    if trained_checkpoint and options.mode=='train':
        optimizer.load_state_dict(trained_checkpoint['optimizer']);scheduler.load_state_dict(trained_checkpoint['scheduler'])
        random.setstate(trained_checkpoint['rng_python']);np.random.set_state(trained_checkpoint['rng_numpy'])
        torch.set_rng_state(trained_checkpoint['rng_torch']);torch.sdaa.set_rng_state_all(trained_checkpoint['rng_sdaa'])
    from engine import evaluate
    validation_loader=DataLoader(validation_dataset,batch_size=options.batch_size,sampler=SequentialSampler(validation_dataset),
        drop_last=False,collate_fn=utils.collate_fn,num_workers=options.workers,worker_init_fn=seed_worker,pin_memory=False)
    base_ds=get_coco_api_from_dataset(validation_dataset)
    manifest['device_work_submitted']=True;write_json(output/'manifest.json',manifest)
    if options.mode=='train':
        for epoch in range(first_epoch,20):
            generator=torch.Generator().manual_seed(options.seed+epoch)
            sampler=RandomSampler(train_dataset,generator=generator)
            loader=DataLoader(train_dataset,batch_sampler=BatchSampler(sampler,options.batch_size,drop_last=True),
                collate_fn=utils.collate_fn,num_workers=options.workers,worker_init_fn=seed_worker,
                generator=torch.Generator().manual_seed(options.seed+epoch+100000),pin_memory=False)
            model.train();criterion.train();torch.sdaa.synchronize();begin=time.perf_counter();images=0
            for step,(samples,targets) in enumerate(loader):
                samples=samples.to(device);targets=[{k:v.to(device) for k,v in target.items()} for target in targets]
                optimizer.zero_grad(set_to_none=True)
                losses=criterion(model(samples),targets)
                loss=sum(losses[k]*criterion.weight_dict[k] for k in losses if k in criterion.weight_dict)
                value=float(loss.detach().cpu())
                if not math.isfinite(value):raise RuntimeError('Nonfinite loss during official fine-tuning')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),args.clip_max_norm)
                if not bool(torch.isfinite(norm).cpu()):raise RuntimeError('Nonfinite training gradient norm')
                optimizer.step();images+=len(targets)
                if step%50==0:print(json.dumps({'epoch':epoch+1,'step':step,'images':images,'loss':value}),flush=True)
            torch.sdaa.synchronize();elapsed=time.perf_counter()-begin;scheduler.step()
            record={'epoch':epoch+1,'images':images,'end_to_end_seconds':elapsed,'images_per_second':images/elapsed}
            with (output/'epochs.jsonl').open('a') as file:file.write(json.dumps(record)+'\n')
            temporary=output/'checkpoint.tmp'
            torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                'identity':identity,'completed_epochs':epoch+1,'rng_python':random.getstate(),'rng_numpy':np.random.get_state(),
                'rng_torch':torch.get_rng_state(),'rng_sdaa':torch.sdaa.get_rng_state_all()},temporary)
            temporary.replace(output/'checkpoint.pth')
            manifest['completed_epochs']=epoch+1;write_json(output/'manifest.json',manifest)
            if (epoch+1)%options.evaluate_every==0 and epoch+1<20:
                stats,_=evaluate(model,criterion,postprocessors,validation_loader,base_ds,torch.device(device),str(output))
                write_json(output/f'evaluation_epoch_{epoch+1:02}.json',stats)
    stats,_=evaluate(model,criterion,postprocessors,validation_loader,base_ds,torch.device(device),str(output))
    write_json(output/'evaluation_final.json',stats)
    manifest['official_accuracy_measured']=manifest['completed_epochs']==20
    manifest['ap_50_95']=stats['coco_eval_bbox'][0]
    manifest['reaches_documented_baseline']=manifest['ap_50_95']>=0.570
    manifest['native_calls']={'forward':binding.forward_calls,'backward':binding.backward_calls}
    write_json(output/'manifest.json',manifest)
    print(json.dumps(manifest,indent=2),flush=True)


if __name__=='__main__':
    main()

"""용도: 검증한 C# 소스·기존 안전 보강 모델·평가 결과를 새 전달 ZIP으로 묶는다.
생성일: 2026-10-10
"""
import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'filtering_training/outputs/runs/2026-10-10-base-safety-continue-01'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    dest=a.output.resolve()
    if not dest.is_relative_to(ROOT):raise ValueError('Keep generated handoff inside workspace')
    if dest.exists() or dest.with_suffix('.zip').exists():raise ValueError('Preserve existing handoff; choose a new directory')
    source=ROOT/'integration/csharp'
    result=json.loads((source/'verification-result.json').read_text(encoding='utf-8-sig'))
    if not result.get('input_token_score_parity') or not result.get('client_contracts_compiled') or not result.get('default_applies_filter'):raise ValueError('Verified client contracts, parity and default filtering required')
    exports=json.loads((RUN/'export.json').read_text(encoding='utf-8'))
    expected=exports['files']['model-fp32.onnx']['sha256']
    if sha(RUN/'model-fp32.onnx')!=expected:raise ValueError('Source model changed')
    code=dest/'integration/csharp';code.mkdir(parents=True)
    for file in source.rglob('*'):
        rel=file.relative_to(source)
        if not file.is_file() or any(s in {'bin','obj','__pycache__','models'} for s in rel.parts):continue
        if file.name=='create_handoff.py':continue
        target=code/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(file,target)
    model=dest/'models/filtering/klue-base-safety-v1';model.mkdir(parents=True)
    for filename,src in [('model-fp32.onnx',RUN/'model-fp32.onnx'),('tokenizer.json',RUN/'tokenizer/tokenizer.json'),('tokenizer_config.json',RUN/'tokenizer/tokenizer_config.json')]:
        shutil.copy2(src,model/filename)
    manifest={'purpose':'Verified native C# filtering handoff; training candidate rejected','created_date':'2026-10-10','version':'klue-base-safety-v1','model_run':RUN.name,'model_sha256':expected,'tokenizer_sha256':sha(model/'tokenizer.json'),'model_bytes':(model/'model-fp32.onnx').stat().st_size,'onnxruntime':'1.31.0','inputs':{'input_ids':'Int64 [batch,sequence]','attention_mask':'Int64 [batch,sequence]'},'output':'logits float [batch,2,5]','heads':['urgency','relevance'],'max_tokens':512,'policy':'urgency >= 4 OR relevance >= 4','default_shadow':False,'client_commit':'4fb6b75022309e3019b31c0b5d339142c635173b','original_policy_correct':'68/74','original_urgent_policy_misses':'2/17','new_training_candidate_adopted':False,'real_notification_accuracy_measured':False}
    (model/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    reports=dest/'reports';reports.mkdir()
    for name,src in [('safety-model-summary.md',RUN/'summary.md'),('boundary-training-baseline.json',ROOT/'filtering_training/outputs/runs/2026-10-10-base-boundary-continue-01/baseline.json'),('boundary-training-epoch-1.json',ROOT/'filtering_training/outputs/runs/2026-10-10-base-boundary-continue-01/epoch-1.json'),('new-holdout-report.json',ROOT/'filtering_training/outputs/runs/2026-10-10-base-boundary-holdout-03/report.json'),('new-holdout-review.md',ROOT/'filtering_training/outputs/runs/2026-10-10-base-boundary-holdout-03/review.md')]:shutil.copy2(src,reports/name)
    shutil.copy2(source/'README.md',dest/'README.md')
    checks={str(f.relative_to(dest)).replace('\\','/'):sha(f) for f in dest.rglob('*') if f.is_file()}
    (dest/'checksums.json').write_text(json.dumps(checks,indent=2)+'\n',encoding='utf-8')
    archive=dest.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for f in dest.rglob('*'):
            if f.is_file():z.write(f,str(f.relative_to(dest)))
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise ValueError('ZIP CRC failure')
    info={'package':str(dest),'zip':str(archive),'zip_bytes':archive.stat().st_size,'zip_sha256':sha(archive),'files':len(checks)+1,'model_sha256':expected}
    (dest.parent/(dest.name+'.manifest.json')).write_text(json.dumps(info,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(info,ensure_ascii=False))

if __name__=='__main__':main()

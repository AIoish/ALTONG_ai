"""용도: 긴급도 출력 추출·관련도 ONNX 변환을 검증하고 앱 코드 없는 전달 ZIP을 만든다.
생성일: 2026-11-17
"""

import argparse
import gc
from pathlib import Path
import shutil
import zipfile

import numpy as np
import onnx
import onnxruntime as ort
import torch
from transformers import AutoTokenizer

from filtering_training.common.artifact_io import ROOT, created_date, digest, load_lines, read, write
from filtering_training.common.encoder_input import INPUT_SCHEMA, encode_checked, input_text
from filtering_training.common.encoder_model import build_encoder


def session(path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    options.add_session_config_entry('session.intra_op.allow_spinning', '0')
    options.add_session_config_entry('session.inter_op.allow_spinning', '0')
    return ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])


def fixtures():
    bodies = ['ㅇ', 'ㅁ', '네', '아니', 'ㅋㅋ', 'ㅠㅠ', '수정한 합계표 공유했습니다.',
              'API 응답 오류를 재현했습니다.', '회의는 내일입니다.', '사진을 보냈습니다.',
              'Excel', '프로젝트 진행은 어떤가요?', '', '123', '확인\n두 번째 줄',
              '"인용"과 \\경로', '이모지 👩‍💻🙂', '中文 日本語 English',
              '공백\t탭\r\n줄바꿈', 'Cafe\u0301와 café', '알림\u00a0공백', '주말 모임 안내',
              '계정 접근이 유출됐습니다. 지금 차단해 주세요.', '문제가 해결되었습니다.']
    rows = []
    for i, body in enumerate(bodies):
        current = {'app_name':'EXCEL.EXE', 'window_title':'분기 합계 검토.xlsx - Excel'}
        rows.append({'id':f'fixture-{i+1:03d}', 'state':{
            'notification':{'app_name':['Slack','KakaoTalk.exe','Discord'][i % 3], 'sender':'검증 담당',
                            'title':'검증용 합성 알림', 'body':body, 'timestamp':'2026-11-17T00:00:00Z'},
            'context':{'active_process':current['app_name'], 'window_title':current['window_title'],
                       'recent_windows':([current, {'app_name':'chrome.exe','window_title':'관련 자료'},
                           {'app_name':'Code.exe','window_title':'다른 작업'},
                           {'app_name':'notepad.exe','window_title':'메모'},
                           {'app_name':'ignored.exe','window_title':'네 번째 이전 창'}] if i % 2 else [])}}})
    return rows


def export_task(run, task, output, rows, public_rows, *, legacy_onnx=None, expected=None):
    run, output = Path(run), Path(output)
    config = read(run/'run_config.json')
    if config['status'] != 'completed' or digest(run/'model.pt') != config['checkpoint_sha256']:
        raise ValueError('completed unchanged checkpoint required')
    protected = {run/'model.pt':digest(run/'model.pt'), run/'run_config.json':digest(run/'run_config.json')}
    sizes = (5,5) if task == 'urgency' else (3,)
    model = build_encoder(base=run/'encoder', pretrained=False, head_sizes=sizes,
                          attention='eager', required_model_type='roberta').eval()
    model.load_state_dict(torch.load(run/'model.pt', map_location='cpu', weights_only=True))
    # Physically remove the unused old relevance head before ONNX tracing.
    if task == 'urgency':
        model.heads = torch.nn.ModuleList([model.heads[0]])

    class SingleOutput(torch.nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.model = encoder

        def forward(self, input_ids, attention_mask):
            return self.model(input_ids, attention_mask)[0]

    wrapper = SingleOutput(model).eval()
    tokenizer = AutoTokenizer.from_pretrained(str(run/'tokenizer'), local_files_only=True)
    output.mkdir(parents=True)
    tokenizer.save_pretrained(output/'tokenizer')
    tokenizer_config = output/'tokenizer/tokenizer_config.json'
    settings = read(tokenizer_config)
    for name in ('name_or_path','tokenizer_file','vocab_file','merges_file',
                 'special_tokens_map_file','added_tokens_file'):
        settings.pop(name, None)
    write(tokenizer_config,settings)
    reloaded = AutoTokenizer.from_pretrained(str(output/'tokenizer'),local_files_only=True)
    texts = [input_text(r['state'], task) for r in rows + public_rows]
    encoded, lengths = encode_checked(tokenizer, texts)
    if encode_checked(reloaded,texts)[0] != encoded:
        raise ValueError('packaged tokenizer changed tokenization')
    sample = tokenizer.pad(encoded[:2], padding=True, return_tensors='pt')
    path = output/'model-fp32.onnx'
    torch.onnx.export(wrapper, (sample['input_ids'], sample['attention_mask']), str(path),
        input_names=['input_ids','attention_mask'], output_names=['logits'],
        dynamic_axes={'input_ids':{0:'batch',1:'sequence'}, 'attention_mask':{0:'batch',1:'sequence'},
                      'logits':{0:'batch'}}, opset_version=17, dynamo=False, external_data=False)
    onnx.checker.check_model(str(path))
    runtime = session(path)
    original = session(legacy_onnx) if legacy_onnx else None
    differences = []
    scores = []
    maximum_error = 0.
    classes = 5 if task == 'urgency' else 3
    for start in range(0, len(encoded), 8):
        batch = tokenizer.pad(encoded[start:start+8], padding=True, return_tensors='pt')
        feed = {k:v.numpy().astype(np.int64) for k,v in batch.items()}
        with torch.inference_mode():
            reference = wrapper(**batch).numpy()
        actual = runtime.run(['logits'], feed)[0]
        if actual.shape != (len(reference), classes):
            raise ValueError('unexpected single-task output shape')
        maximum_error = max(maximum_error, float(np.max(np.abs(reference-actual))))
        np.testing.assert_allclose(actual, reference, rtol=1e-4, atol=2e-4)
        if not np.array_equal(actual.argmax(-1), reference.argmax(-1)):
            raise ValueError('PyTorch/ONNX score mismatch')
        if original:
            old = original.run(['logits'],feed)[0][:,0,:]
            np.testing.assert_allclose(actual,old,rtol=1e-4,atol=2e-4)
            if not np.array_equal(actual.argmax(-1),old.argmax(-1)):
                raise ValueError('existing urgency ONNX score changed')
        scores.extend((actual.argmax(-1)+1).tolist())
        if start % 128 == 0:
            print({'task':task,'verified':min(start+8,len(encoded)),'total':len(encoded)},flush=True)
    if expected:
        differences = [r['id'] for r,s in zip(rows,scores) if expected[r['id']] != s]
        if differences:
            raise ValueError('saved selected-checkpoint predictions changed')
    # Public fixtures contain authored synthetic inputs only, never reviewed user messages.
    for i, row in enumerate(public_rows, len(rows)):
        row.setdefault('expected',{})[task] = {'input_text':texts[i], 'input_ids':encoded[i]['input_ids'],
            'attention_mask':encoded[i]['attention_mask'], 'score':scores[i]}
    report = {'purpose':'FP32 conversion parity, not an independent accuracy evaluation',
        'created_date':created_date(), 'task':task, 'rows':len(encoded), 'private_development_rows':len(rows),
        'public_synthetic_rows':len(public_rows), 'maximum_tokens':max(lengths), 'truncated_rows':0,
        'pytorch_onnx_score_parity':True, 'existing_urgency_onnx_parity':True if original else None,
        'saved_validation_predictions_parity':True if expected else None,
        'maximum_absolute_logit_error':maximum_error, 'mismatches':len(differences),
        'csharp_parity_tested':False, 'full_app_tested':False}
    write(output/'verification.json',report)
    manifest = {'purpose':'Single-task FP32 ONNX integration candidate', 'created_date':created_date(),
        'task':task, 'source_run':run.name, 'source_checkpoint_sha256':config['checkpoint_sha256'],
        'selected_epoch':config['selected_epoch'], 'score_range':[1,classes],
        'input_schema':'notification-only-v1' if task == 'urgency' else INPUT_SCHEMA,
        'inputs':{'input_ids':'Int64 [batch,sequence]','attention_mask':'Int64 [batch,sequence]'},
        'output':{'name':'logits','dtype':'float32','shape':['batch',classes]},
        'decoding':'argmax(logits[0,:]) + 1; first maximum on ties', 'max_tokens':512,
        'onnxruntime_verified':ort.__version__, 'opset':17, 'precision':'FP32',
        'model_sha256':digest(path), 'model_bytes':path.stat().st_size,
        'tokenizer_sha256':digest(output/'tokenizer/tokenizer.json'),
        'training_performed':False, 'unused_relevance_head_removed':task == 'urgency',
        'deployment_approved':False}
    write(output/'manifest.json',manifest)
    for p,h in protected.items():
        if digest(p)!=h:
            raise ValueError('source artifact changed')
    del original, runtime, model, wrapper
    gc.collect()
    return report


def package(output, urgency_run, relevance_run, guide):
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT) or output.exists() or output.with_suffix('.zip').exists():
        raise ValueError('choose a new output directory within workspace')
    torch.set_num_threads(4)
    output.mkdir(parents=True)
    public = fixtures()
    old_rows = read(Path(urgency_run)/'epoch-1.predictions.json')
    urgency_rows = [{'id':str(i),'state':{'notification':r['notification']}} for i,r in enumerate(old_rows)]
    config = read(Path(relevance_run)/'run_config.json')
    rows = load_lines(Path(config['dataset_dir'])/'relevance/validation.jsonl')
    expected = {r['id']:r['predicted'] for r in load_lines(Path(relevance_run)/'validation.predictions.jsonl')}
    reports = [export_task(urgency_run,'urgency',output/'models/filtering/urgency-v1',urgency_rows,public,
                          legacy_onnx=Path(urgency_run)/'model-fp32.onnx'),
               export_task(relevance_run,'relevance',output/'models/filtering/relevance-v1',rows,public,expected=expected)]
    write(output/'verification-fixtures.json',{'purpose':'Authored synthetic token/score fixtures; scores are model predictions, not gold',
          'created_date':created_date(),'fixtures':public})
    write(output/'verification-report.json',{'purpose':'Single-task exports verification; app integration remains untested',
          'created_date':created_date(),'tasks':reports,'csharp_parity_tested':False,'full_app_tested':False})
    shutil.copy2(guide,output/'APPLY_GUIDE.md')
    manifest = {'purpose':'Models-only urgency5/relevance3 integration candidate handoff; no app source',
        'created_date':created_date(),'version':'filtering-u5-r3-2026-11-17',
        'relevance_candidate':'teacher share5%, seed42, selected epoch2',
        'development_accuracy':{'rows':1032,'correct':1012,'real_rows':47,'real_correct':40},
        'policy_proposal':'urgency >= 4 OR relevance == 3; app team must explicitly adopt',
        'deployment_approved':False,'files':{str(p.relative_to(output)).replace('\\','/'):digest(p)
                                             for p in output.rglob('*') if p.is_file()}}
    write(output/'manifest.json',manifest)
    write(output/'checksums.json',{str(p.relative_to(output)).replace('\\','/'):digest(p)
                                 for p in output.rglob('*') if p.is_file()})
    archive = output.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as zipped:
        for path in sorted(output.rglob('*')):
            if path.is_file():
                zipped.write(path,str(path.relative_to(output)))
    with zipfile.ZipFile(archive) as zipped:
        if zipped.testzip() is not None:
            raise ValueError('corrupt ZIP')
    write(archive.with_suffix('.zip.sha256.json'),{'purpose':'Release asset integrity','created_date':created_date(),
          'name':archive.name,'bytes':archive.stat().st_size,'sha256':digest(archive)})
    print({'archive':str(archive),'bytes':archive.stat().st_size,'tasks':reports},flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--urgency-run',type=Path,required=True)
    parser.add_argument('--relevance-run',type=Path,required=True)
    parser.add_argument('--guide',type=Path,required=True)
    args = parser.parse_args()
    package(args.output,args.urgency_run,args.relevance_run,args.guide)

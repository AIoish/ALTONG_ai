"""용도: 앱 develop 계약 공동 모델을 앱의 models/filtering/klue-base-safety-v1 폴더에 그대로 넣을 수 있는 전달 ZIP으로 만들고 CPU ONNX로 검증한다.
생성일: 2026-11-19
"""

import argparse
from pathlib import Path
import shutil
import statistics
import time
import zipfile

import numpy as np
import onnxruntime as ort

from filtering_training.common.artifact_io import ROOT, created_date, digest, load_lines, read, write
from filtering_training.common.csharp_tokenizer import encode as app_encode, load_rules
from filtering_training.common.encoder_data import DEFAULT_DATASET, load_evaluation_inputs
from filtering_training.common.encoder_input import APP_INPUT_SCHEMA, input_text
from filtering_training.common.validation_data import evaluation_sets
from filtering_training.modeling.export_filtering_handoff import fixtures, session
from filtering_training.modeling.joint_app_contract import APP_RELEVANCE_SCORE, app_passed

APP_FOLDER = 'models/filtering/klue-base-safety-v1'
# tokenizer.json currently tracked in ALTONG_client develop (LF bytes); shipping it unchanged keeps tokenizer_sha256.
APP_TOKENIZER_SHA256 = 'cc02cedb56df2e4d3bb6b5478cef6aa0daa98930c96b46976c8acdd2432459a8'
APP_TOKENIZER = ROOT / 'filtering_training/outputs/runs/2026-10-10-base-safety-continue-01/tokenizer/tokenizer.json'


def app_run(sess, rules, text, head):
    """Same steps as KlueFilter.Run: app token ids, batch 1, all-ones mask, finite check, first maximum."""
    ids = np.asarray([app_encode(rules, text)], dtype=np.int64)
    logits = sess.run(['logits'], {'input_ids': ids, 'attention_mask': np.ones_like(ids)})[0]
    if logits.shape != (1, 2, 5) or not np.isfinite(logits).all():
        raise ValueError('app contract violated: logits shape or non-finite values')
    return ids[0].tolist(), int(np.argmax(logits[0, head])) + 1


def verify(model_path, tokenizer_path, run, config):
    sess, rules = session(model_path), load_rules(tokenizer_path)
    names = ([i.name for i in sess.get_inputs()], [(o.name, o.shape) for o in sess.get_outputs()])
    expected = config['selected_scores']
    evals, _ = load_evaluation_inputs(DEFAULT_DATASET, evaluation_sets())
    mismatches, times, passed = [], [], {}
    for name, rows in evals.items():
        passed[name] = 0
        for i, row in enumerate(rows):
            begin = time.perf_counter()
            _, u = app_run(sess, rules, input_text(row, 'urgency', schema=APP_INPUT_SCHEMA), 0)
            _, r = app_run(sess, rules, input_text(row, 'relevance', schema=APP_INPUT_SCHEMA), 1)
            times.append(time.perf_counter() - begin)
            want = (expected[name]['urgency'][i], APP_RELEVANCE_SCORE[expected[name]['relevance'][i]])
            passed[name] += app_passed(u, r)
            if (u, r) != want:
                mismatches.append({'set': name, 'index': i, 'onnx': [u, r], 'torch': list(want)})
    selected = read(run / f"epoch-{config['selected_epoch']}.predictions.json")['relevance_validation']
    validation = load_lines(Path(config['dataset_dir']) / 'relevance/validation.jsonl')
    relevance_mismatch = sum(app_run(sess, rules, input_text(row['state'], 'relevance', schema=APP_INPUT_SCHEMA), 1)[1]
                             != APP_RELEVANCE_SCORE[score] for row, score in zip(validation, selected))
    public = []
    for row in fixtures():
        item = {'id': row['id'], 'state': row['state']}
        for head, task in enumerate(('urgency', 'relevance')):
            text = input_text(row['state'], task, schema=APP_INPUT_SCHEMA)
            ids, score = app_run(sess, rules, text, head)
            item[task] = {'input_text': text, 'input_ids': ids, 'app_score': score}
        public.append(item)
    report = {'created_date': created_date(), 'onnxruntime': ort.__version__, 'onnx_inputs_outputs': names,
              'evaluation_rows': len(times), 'evaluation_score_mismatches': mismatches, 'app_passed_counts': passed,
              'relevance_validation_rows': len(validation), 'relevance_validation_mismatches': relevance_mismatch,
              'cpu_mean_ms_two_calls': statistics.mean(times) * 1000,
              'cpu_p95_ms_two_calls': float(np.percentile(times, 95)) * 1000,
              'tokenization': 'Python port of app KlueTokenizer (common/csharp_tokenizer.py) on the shipped tokenizer.json',
              'csharp_executed': False}
    if mismatches or relevance_mismatch:
        raise ValueError('packaged ONNX differs from the selected checkpoint predictions')
    return report, public


def package(run, output, guide, comparison):
    run, output = Path(run).resolve(), Path(output).resolve()
    if not output.is_relative_to(ROOT) or output.exists() or output.with_suffix('.zip').exists():
        raise ValueError('choose a new output directory within workspace')
    config = read(run / 'run_config.json')
    source = run / 'app-package/model-fp32.onnx'
    if config['status'] != 'completed' or digest(source) != config['app_package']['manifest']['model_sha256']:
        raise ValueError('completed run with unchanged app ONNX required')
    if digest(APP_TOKENIZER) != APP_TOKENIZER_SHA256:
        raise ValueError('app tokenizer.json bytes changed')
    folder = output / APP_FOLDER
    folder.mkdir(parents=True)
    shutil.copy2(source, folder / 'model-fp32.onnx')
    shutil.copyfile(APP_TOKENIZER, folder / 'tokenizer.json')
    report, public = verify(folder / 'model-fp32.onnx', folder / 'tokenizer.json', run, config)
    relevance, urgency = config['selected_relevance'], config['selected_urgency']
    real = relevance['by_origin']['real_development']['confusion_matrix']
    write(folder / 'manifest.json', {
        'purpose': 'Drop-in replacement model for this folder; app code and tokenizer.json unchanged',
        'created_date': created_date(), 'version': 'klue-joint-app-2026-11-18-t00-s42',
        'folder_name_kept_for_app_path': True, 'model_run': run.name,
        'model_sha256': digest(folder / 'model-fp32.onnx'), 'tokenizer_sha256': digest(folder / 'tokenizer.json'),
        'model_bytes': (folder / 'model-fp32.onnx').stat().st_size, 'onnxruntime': ort.__version__,
        'inputs': {'input_ids': 'Int64 [batch,sequence]', 'attention_mask': 'Int64 [batch,sequence]'},
        'output': 'logits float [batch,2,5]', 'heads': ['urgency', 'relevance'], 'max_tokens': 512,
        'input_schema': APP_INPUT_SCHEMA, 'relevance_classes_trained': 3,
        'relevance_score_to_app': {str(k): v for k, v in APP_RELEVANCE_SCORE.items()},
        'policy': 'urgency >= 4 OR relevance >= 4',
        'development_relevance': {'rows': relevance['rows'], 'correct': relevance['exact_correct'],
                                  'real_rows': sum(map(sum, real)), 'real_correct': sum(real[i][i] for i in range(3))},
        'urgency_validation_correct': f"{urgency['validation']['exact_correct']}/{urgency['validation']['rows']}",
        'urgency_threshold_misses': {k: len(urgency[k]['threshold_miss_ids']) for k in ('validation', 'safety_development')},
        'real_notification_final_test_measured': False, 'deployment_approved': False})
    write(output / 'verification-report.json', dict(report, purpose='CPU ONNX app-contract verification; C# not executed',
                                                     model_comparison=read(comparison) if comparison else None))
    write(output / 'verification-fixtures.json', {'purpose': 'Authored synthetic inputs with app token ids and model scores (predictions, not gold)',
                                                  'created_date': created_date(), 'fixtures': public})
    shutil.copy2(guide, output / 'APPLY_GUIDE.md')
    write(output / 'checksums.json', {str(p.relative_to(output)).replace('\\', '/'): digest(p)
                                      for p in sorted(output.rglob('*')) if p.is_file()})
    archive = output.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=3) as zipped:
        for path in sorted(output.rglob('*')):
            if path.is_file():
                zipped.write(path, str(path.relative_to(output)))
    with zipfile.ZipFile(archive) as zipped:
        if zipped.testzip() is not None:
            raise ValueError('corrupt ZIP')
    write(archive.with_suffix('.zip.sha256.json'), {'purpose': 'Release asset integrity', 'created_date': created_date(),
                                                    'name': archive.name, 'bytes': archive.stat().st_size, 'sha256': digest(archive)})
    return {'archive': str(archive), 'bytes': archive.stat().st_size, 'sha256': digest(archive),
            'model_sha256': digest(folder / 'model-fp32.onnx'), 'verification': {k: v for k, v in report.items() if k != 'evaluation_score_mismatches'}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--guide', type=Path, required=True)
    parser.add_argument('--comparison', type=Path)
    args = parser.parse_args()
    print(package(args.run, args.output, args.guide, args.comparison), flush=True)

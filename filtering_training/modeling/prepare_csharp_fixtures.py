"""용도: Python ONNX의 입력·토큰·점수를 C# 비교용 고정 자료로 내보낸다.
생성일: 2026-10-10
"""
import argparse
import json
import random
from pathlib import Path
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from filtering_training.modeling.encoder_score_experiment import input_text

ROOT=Path(__file__).resolve().parents[2]
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    tok=Tokenizer.from_file(str(a.run/'tokenizer/tokenizer.json'))
    opts=ort.SessionOptions();opts.intra_op_num_threads=4;opts.inter_op_num_threads=1
    opts.add_session_config_entry('session.intra_op.allow_spinning','0');opts.add_session_config_entry('session.inter_op.allow_spinning','0')
    session=ort.InferenceSession(str(a.run/'model-fp32.onnx'),opts,providers=['CPUExecutionProvider'])
    cfg=json.loads((a.run/'run_config.json').read_text(encoding='utf-8'))
    original=json.loads((a.run/f"epoch-{cfg['selected_epoch']}.predictions.json").read_text(encoding='utf-8'))
    states=[{'notification':x['notification'],'context':x['context']} for x in original]
    for name in ('v9_personal_use_eval_01','v11_boundary_supplement_01'):
        states.extend(json.loads(s) for s in (ROOT/f'filtering_training/outputs/{name}/dataset.jsonl').read_text(encoding='utf-8').splitlines())
    cases=[]
    for state in states:
        f={k:state[k] for k in ('notification','context')};pred={}
        for head,task in enumerate(('urgency','relevance')):
            text=input_text(state,task);e=tok.encode(text)
            if len(e.ids)>512:raise ValueError('Fixture exceeds context')
            y=session.run(None,{'input_ids':np.array([e.ids],dtype=np.int64),'attention_mask':np.array([e.attention_mask],dtype=np.int64)})[0][0,head]
            f[task+'_text']=text;f[task+'_ids']=e.ids;pred[task+'_score']=int(y.argmax())+1
        f['prediction']=pred;cases.append(f)
    texts=['','집에 불났어','[CLS] [PAD] [SEP] [UNK] [MASK]','foo[MASK]bar','ＡＢＣ café CAFÉ e\u0301','한글\t中文\n日本語 😀🚨','"\\\b\f\r\n\t\x00\x01\x7f','x\u200by\u2028z\u2029q\ufeff끝','a'*101,'1,000원 > 900원 < 5% & https://abc.co/a?x=1']
    rng=random.Random(19);chars='가나다불났어긴급확인ABCxyz123 !?.[]_-\t\n😀中é\u0301\u00a0\u200b\u2028'
    texts += [''.join(rng.choice(chars) for _ in range(rng.randrange(1,140))) for _ in range(250)]
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps({'purpose':'Synthetic Python/C# input-token-score parity fixtures','created_date':'2026-10-10','model_sha256':json.loads((a.run/'export.json').read_text(encoding='utf-8'))['files']['model-fp32.onnx']['sha256'],'cases':cases,'tokenizer_cases':[{'text':s,'ids':tok.encode(s).ids} for s in texts]},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'cases':len(cases),'tokenizer_cases':len(texts)}))
if __name__=='__main__':main()

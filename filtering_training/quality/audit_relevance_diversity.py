"""용도: 실제 생성 틀과 라벨별 앱·창 개수 분포를 점검해 반복을 의미 다양성으로 오인하지 않게 한다.
생성일: 2026-11-08
"""

import argparse
import ast
from collections import Counter, defaultdict
import json
from pathlib import Path
import re

from filtering_training.common.artifact_io import ROOT, created_date, digest, load_lines, write
from filtering_training.preparation.curate_relevance import OUTPUT, core_body, profile

GENERATOR = ROOT / 'filtering_training/generation/generate_score_review_batch.py'
TEMPLATE_NAMES = {'BULK_EVENTS', 'TRAINING_EVENT_PHRASINGS', 'unrelated_templates',
                  'decorative_templates', 'topic_templates', 'resource_templates'}


def authored_templates(path=GENERATOR):
    """Read literal templates with AST; never import old Qwen modules or execute their code."""
    tree = ast.parse(Path(path).read_text(encoding='utf-8'))
    result = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        names = {t.id for t in node.targets if isinstance(t, ast.Name)}
        if not names.intersection(TEMPLATE_NAMES):
            continue
        for literal in ast.walk(node.value):
            if isinstance(literal, ast.Constant) and isinstance(literal.value, str):
                template = literal.value
                if not re.search(r'\{(?:goal|item|field|resource|motif)\}', template):
                    continue
                # Names and objects may vary; retain the whole fixed sentence, including negation.
                parts, previous = [], 0
                for slot in re.finditer(r'\{(?:goal|item|field|resource|motif)\}([을를은는이가과와]?)', template):
                    parts.append(re.escape(template[previous:slot.start()]))
                    parts.append('.+?')
                    if slot[1]:
                        pairs = {'을': '을를', '를': '을를', '은': '은는', '는': '은는',
                                 '이': '이가', '가': '이가', '과': '과와', '와': '과와'}
                        parts.append('[' + pairs[slot[1]] + ']')
                    previous = slot.end()
                parts.append(re.escape(template[previous:]))
                ending = r'(?: .+ 쓸 수 있어\.)?' if 'resource_templates' in names else ''
                result[template] = re.compile('^' + ''.join(parts) + ending + '$')
    return sorted(result.items(), key=lambda pair: (-len(pair[0]), pair[0]))


def template_report(records, templates=None):
    templates = authored_templates() if templates is None else templates
    groups, unmatched = defaultdict(list), []
    for row in records:
        body = core_body(row['state']['notification']['body'])
        matched = next((text for text, pattern in templates if pattern.fullmatch(body)), None)
        (groups[matched] if matched is not None else unmatched).append(row)
    findings = []
    for template, rows in groups.items():
        findings.append({'template': template, 'rows': len(rows),
                         'distinct_bodies': len({r['state']['notification']['body'] for r in rows}),
                         'scores': dict(Counter(r['score'] for r in rows)),
                         'examples': [{'id': r['id'], 'body': r['state']['notification']['body'],
                                       'window': r['state']['context']['window_title'], 'score': r['score']}
                                      for r in rows[:3]]})
    return {'matched_rows': sum(len(v) for v in groups.values()), 'unmatched_rows': len(unmatched),
            'matched_template_count': len(groups), 'templates': sorted(findings, key=lambda r: (-r['rows'], r['template'])),
            'interpretation': 'Template repetition is a review signal, not proof that task/context variants are redundant.'}


def audit(folder=OUTPUT):
    folder = Path(folder)
    source = folder / 'relevance/retained.jsonl'
    rows = load_lines(source)
    source_hash = digest(source)
    result = {'purpose': 'Relevance diversity review after conservative wrapper curation',
              'created_date': created_date(), 'source_sha256': source_hash,
              'generator_sha256': digest(GENERATOR), 'profile': profile(rows),
              'repeated_expression_families': template_report(rows),
              'training_ready': False, 'labels_changed': False}
    if digest(source) != source_hash:
        raise ValueError('source changed during audit')
    write(folder / 'review/diversity-audit.json', result)
    print(json.dumps({'retained_rows': len(rows),
                      'matched_template_count': result['repeated_expression_families']['matched_template_count'],
                      'matched_rows': result['repeated_expression_families']['matched_rows']}, ensure_ascii=False))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, default=OUTPUT)
    audit(parser.parse_args().folder)

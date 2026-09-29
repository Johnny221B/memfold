"""Train-only, full-conversation memory banks; no Persona MCQ or gold retrieval."""
import hashlib
import json
from pathlib import Path


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def source_identity(row):
    identity=row.get('source_id') or row.get('source_adapter')
    if not isinstance(identity,str) or not identity:
        raise ValueError('missing explicit memory source identity')
    return identity


def memory_bank(rows):
    if not rows or any(r['split']!='train' for r in rows):
        raise ValueError('compressor training accepts train memories only')
    if len({r['id'] for r in rows})!=len(rows):
        raise ValueError('duplicate memory ID')
    if len({source_identity(r) for r in rows})!=1:
        raise ValueError('mixed memory sources')
    bank={}
    for row in rows:
        text=row['memory_text']
        obj=json.loads(text)
        if not isinstance(obj,dict) or set(obj)!={'memories'} or not isinstance(obj['memories'],list):
            raise ValueError('native v3 memory required; run prepare_compressor_reconstruction first')
        if row['state_id']!=sha((source_identity(row)+'\0'+row['id']+'\0'+text).encode()):
            raise ValueError('memory provenance hash mismatch')
        bank.setdefault(row['context_id'],[]).append(row)
    return bank


def qa_rows(rows, bank):
    if not rows or len({r['question_id'] for r in rows})!=len(rows):
        raise ValueError('empty or duplicate QA rows')
    for r in rows:
        if r['split']!='train' or r['context_id'] not in bank:
            raise ValueError('QA split/context mismatch')
        expected={m['id'] for m in bank[r['context_id']]}
        ids=r['source_session_ids']
        if len(ids)!=len(set(ids)) or set(ids)!=expected:
            raise ValueError('QA must use complete conversation, not gold evidence sessions')
        if not isinstance(r['question'],str) or not isinstance(r['answer'],str):
            raise ValueError('LoCoMo open-answer strings required')
    return rows


def reasoning_rows(rows, questions, memories_sha256):
    """Caller supplies audited explanations; never synthesize them from gold answers.

    A separate field must contain reasoning only, without the final-answer clause.
    Provenance declarations are checked but cannot prove semantic factual support.
    """
    lookup={r['question_id']:r for r in rows}
    if len(lookup)!=len(rows) or set(lookup)!={q['question_id'] for q in questions}:
        raise ValueError('reasoning targets must cover all train questions exactly')
    for q in questions:
        r=lookup[q['question_id']]
        if r['split']!='train' or r['context_id']!=q['context_id']:
            raise ValueError('reasoning split/context mismatch')
        if r.get('memory_file_sha256')!=memories_sha256:
            raise ValueError('reasoning targets refer to different memory bank')
        if r.get('gold_in_prompt') is not False or r.get('answer_tokens_supervised') is not False:
            raise ValueError('reasoning provenance must exclude gold prompt/final-answer supervision')
        if not r.get('teacher_checkpoint') or not r.get('prompt_sha256') or not r.get('reasoning','').strip():
            raise ValueError('missing reasoning text/teacher/prompt provenance')
    return lookup


def negative_context(context, bank):
    contexts=sorted(bank)
    if len(contexts)<2:
        raise ValueError('ranking/separation needs at least two train conversations')
    return contexts[(contexts.index(context)+1)%len(contexts)]

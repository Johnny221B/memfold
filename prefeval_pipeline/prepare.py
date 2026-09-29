"""Build the HF persona evaluation using pinned official retrieval scores/prompts."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / 'vendor/PrefEval'

def read(path):
    return json.loads(path.read_text())

def official_function(path, name):
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    scope = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), scope)
    return scope[name]

def messages(conversation):
    return [{'role': role, 'content': turn[role]} for _, turn in
            sorted(conversation.items(), key=lambda x: int(x[0])) if turn is not None
            for role in ('user', 'assistant')]

def main():
    import pyarrow.parquet as pq
    rows = pq.read_table(ROOT / 'data/data/train-00000-of-00001.parquet').to_pylist()
    base = VENDOR / 'benchmark_dataset'
    inter = [m for s in read(base / 'filtered_inter_turns.json') for m in s['conversation']][:4]
    assert [m['role'] for m in inter] == ['user', 'assistant'] * 2
    prefix = official_function(VENDOR / 'utils/implicit_utils.py', 'convert_top_k_sentences_to_msg')
    cache = {}
    for topic in sorted({r['topic'] for r in rows}):
        p = read(base / f'rag_retrieval/simcse_implicit_persona/{topic}_overall300_topk_history_persona.json')
        q = read(base / f'rag_retrieval/simcse_question_inter_conversation_similarities/{topic}_300_inter_similarities.json')
        cache[topic] = (p, q)
    prepared, issues = [], []
    encoder = tokenizer = None
    fallback_path = ROOT / 'data/recomputed_scores.json'
    fallback = read(fallback_path) if fallback_path.exists() else {}
    for i, row in enumerate(rows):
        history = messages(row['conversation'])
        p, q = cache[row['topic']]
        matches = [r for r in p if r['question'] == row['question'] and
                   messages(r['conversation']) == history]
        distractors = [r for r in q if r['question'] == row['question'] and r['preference'] == row['preference']]
        if len(matches) != 1 or len(distractors) != 1:
            issues.append({'id': i, 'topic': row['topic'], 'persona_matches': len(matches), 'inter_matches': len(distractors)})
            if str(i) not in fallback:
                import torch
                from transformers import AutoModel, AutoTokenizer
                if encoder is None:
                    model = str(ROOT.parent / 'models/sup-simcse-roberta-large')
                    tokenizer = AutoTokenizer.from_pretrained(model)
                    encoder = AutoModel.from_pretrained(model).to('cuda').eval()
                texts = [row['question']] + [m['content'] for m in history + inter]
                inputs = tokenizer(texts,padding=True,truncation=True,max_length=512,return_tensors='pt').to('cuda')
                with torch.no_grad():
                    emb = torch.nn.functional.normalize(encoder(**inputs).pooler_output,dim=1)
                    values = (emb[1:] @ emb[0]).cpu().tolist()
                fallback[str(i)] = {'scores':values,'model':'princeton-nlp/sup-simcse-roberta-large',
                                    'pooling':'pooler_output','max_length':512,'text_format':'content_without_role'}
                fallback_path.write_text(json.dumps(fallback,indent=2))
            vals = fallback[str(i)]['scores']
            assert len(vals) == len(history)+4
            scores = list(enumerate(vals[:len(history)]))
            dscores = list(enumerate(vals[len(history):]))
            retrieval_source = 'recomputed_simcse'
        else:
            scores = matches[0]['sentence_scores']
            dscores = distractors[0]['inter_sentence_scores'][:4]
            retrieval_source = 'official_cache'
        assert [x[0] for x in scores] == list(range(len(history)))
        assert [x[0] for x in dscores] == list(range(4))
        candidates = [(j, s[1]) for j, s in enumerate(scores)] + [(len(history)+j, s[1]) for j,s in enumerate(dscores)]
        top = sorted(candidates, key=lambda x: x[1], reverse=True)[:5]
        all_history = history + inter
        retrieved = [all_history[j]['role'] + ': ' + all_history[j]['content'] for j,_ in top]
        common = [{'role':'system','content':'You are a helpful assistant.'}] + all_history
        for method in ('full_text', 'rag'):
            question = (prefix(retrieved) if method == 'rag' else '') + row['question'] + ' (Please respond within 300 words.)'
            prepared.append({'id':i, 'method':method, 'topic':row['topic'], 'question':row['question'],
                             'preference':row['preference'], 'messages': common + [{'role':'user','content':question}],
                             'retrieval':top if method == 'rag' else [], 'retrieval_source':retrieval_source if method == 'rag' else None})
    out = ROOT / 'prepared'; out.mkdir(exist_ok=True)
    (out/'alignment_issues.json').write_text(json.dumps(issues, indent=2))
    print(json.dumps({'hf_rows':len(rows),'prepared':len(prepared),'issues':len(issues)}))
    assert len(rows) == 1000 and len(prepared) == 2000
    with (out/'inputs.jsonl').open('w') as f:
        for row in prepared: f.write(json.dumps(row, ensure_ascii=False)+'\n')
    files = [ROOT/'data/data/train-00000-of-00001.parquet', base/'filtered_inter_turns.json', VENDOR/'utils/implicit_utils.py']
    files += list((base/'rag_retrieval/simcse_implicit_persona').glob('*.json'))
    files += list((base/'rag_retrieval/simcse_question_inter_conversation_similarities').glob('*.json'))
    if fallback_path.exists(): files.append(fallback_path)
    (out/'manifest.json').write_text(json.dumps({'hf':read(ROOT/'data/hf_revision.json'),
        'official_commit':'50795054b5ff5f418d2b768a331d71e480f93331','inter_turns':2,'topk_messages':5,
        'rag_keeps_full_history':True, 'rows':1000,'sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}},indent=2))

if __name__ == '__main__': main()

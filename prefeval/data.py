"""PrefEval input validation and file hashes."""
import hashlib
import json

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read_jsonl(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]

def inputs(source,limit):
    rows=[r for r in read_jsonl(source) if r['method']=='full_text']
    if len(rows)!=1000 or len({r['id'] for r in rows})!=1000:raise ValueError('Expected 1000 unique full-history HF records')
    for row in rows:
        if row['messages'][0]!={'role':'system','content':'You are a helpful assistant.'}:raise ValueError('Unexpected system prompt')
        if row['messages'][-1]['role']!='user':raise ValueError('Missing final query')
    return rows[:limit] if limit else rows

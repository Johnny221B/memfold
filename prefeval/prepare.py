"""Prepare full-history PrefEval inputs for MemFold."""
import argparse
import hashlib
import json
from pathlib import Path


def messages(conversation):
    return [{'role': role, 'content': turn[role]} for _, turn in
            sorted(conversation.items(), key=lambda x: int(x[0])) if turn is not None
            for role in ('user', 'assistant')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True, help='Official persona parquet file')
    parser.add_argument('--inter-turns', type=Path, required=True, help='Official filtered_inter_turns.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import pyarrow.parquet as pq
    rows = pq.read_table(args.data).to_pylist()
    inter = [m for s in json.loads(args.inter_turns.read_text()) for m in s['conversation']][:4]
    if [m['role'] for m in inter] != ['user', 'assistant'] * 2:
        raise ValueError('Expected two interleaved user/assistant turns')
    if len(rows) != 1000:
        raise ValueError('Expected 1000 PrefEval records')
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / 'inputs.jsonl').open('w') as out:
        for index, row in enumerate(rows):
            history = messages(row['conversation']) + inter
            question = row['question'] + ' (Please respond within 300 words.)'
            record = dict(id=index, method='full_text', topic=row['topic'], question=row['question'],
                          preference=row['preference'], messages=[{'role':'system','content':'You are a helpful assistant.'}]
                          + history + [{'role':'user','content':question}], retrieval=[], retrieval_source=None)
            out.write(json.dumps(record, ensure_ascii=False) + '\n')
    manifest = dict(rows=len(rows), inter_turns=2, sha256={
        str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [args.data, args.inter_turns]})
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()

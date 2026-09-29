"""Build memory reconstruction pairs from memory-writer initialization checkpoint self memories."""
import argparse
import json
from pathlib import Path
from .prepare import compact,jsonl,sha

def build(writer_rows,self_rows,checkpoint):
    expected={r['task_id']:r for r in writer_rows}
    if len(expected)!=len(writer_rows): raise ValueError('duplicate writer ID')
    bank={r['task_id']:r for r in self_rows}
    if len(bank)!=len(self_rows) or set(bank)!=set(expected): raise ValueError('self-memory ID coverage mismatch')
    out={'train':[],'validation':[]}
    for jid,src in expected.items():
        row=bank[jid]
        if row.get('origin')!='self_generated' or row.get('source_adapter')!=checkpoint:
            raise ValueError('compressor reconstruction requires selected memory-writer initialization checkpoint self memory')
        if row.get('context_id')!=src['context_id'] or row.get('split')!=src['split']:
            raise ValueError('self-memory split/context mismatch')
        if row.get('input_sha256')!=src['input_sha256']: raise ValueError('self-memory source hash mismatch')
        if row.get('finish_reason')!='stop': raise ValueError('non-stop self memory')
        mem=json.loads(row['memory_text'])
        if not isinstance(mem,dict) or set(mem)!={'memories'} or not isinstance(mem['memories'],list):
            raise ValueError('self memory must use native v3 wrapper')
        if not all(isinstance(m,dict) and isinstance(m.get('statement'),str) and m['statement'].strip() for m in mem['memories']):
            raise ValueError('invalid memory proposition')
        # Preserve schema and content rather than truncate to PersonaMem evidence-v1.
        text=compact(mem)
        out[src['split']].append(dict(id=jid,context_id=src['context_id'],split=src['split'],
            memory_text=text,target_text=text,state_id=sha((checkpoint+'\0'+jid+'\0'+text).encode()),
            source_adapter=checkpoint,input_sha256=src['input_sha256'],
            objective='memory_reconstruction',trainable_component='compressor'))
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--self-memories',type=Path,required=True,help='train+validation checkpoint outputs; no API targets')
    p.add_argument('--writer-checkpoint',required=True,help='exact identifier recorded as source_adapter')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError('refusing to overwrite output')
    manifest=json.loads((a.prepared/'manifest.json').read_text())
    writers=[r for split in ['train','validation'] for r in jsonl(a.prepared/split/'writer_inputs.jsonl')]
    rows=build(writers,jsonl(a.self_memories),a.writer_checkpoint)
    a.output.mkdir(parents=True)
    for split,rr in rows.items(): (a.output/(split+'.jsonl')).write_text(''.join(compact(r)+'\n' for r in rr))
    (a.output/'manifest.json').write_text(json.dumps(dict(procedure='compressor_reconstruction',objective='memory_reconstruction',
        writer_checkpoint=a.writer_checkpoint,trainable=['compressor'],
        backbone_base_frozen=True,backbone_lora_frozen=True,
        complete_data=manifest['complete_data'],self_memory_sha256=sha(a.self_memories.read_bytes()),
        counts={s:len(r) for s,r in rows.items()},training_launched=False),indent=2)+'\n')
if __name__=='__main__': main()

"""Prepare LoCoMo writer SFT and open-answer QA; fail closed on incomplete targets."""
import argparse
import ast
import hashlib
import json
import re
from pathlib import Path

WORKSPACE=Path(__file__).resolve().parent.parent
CATEGORY={1:'multi_hop',2:'temporal',3:'open_domain',4:'single_hop',5:'adversarial'}
FIELDS={'memory_type','subject','facet','attribute','value','statement','polarity','temporal','state_status','explicitness','evidence_ids','confidence'}

def sha(x): return hashlib.sha256(x).hexdigest()
def jsonl(path): return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
def compact(x): return json.dumps(x,ensure_ascii=False,separators=(',',':'))

def writer_prompt(script):
    tree=ast.parse(script.read_text())
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SYSTEM_PROMPT' for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError('extractor SYSTEM_PROMPT not found')

def sessions(conversations):
    jobs={}
    for c in conversations:
        ctx=c['sample_id']; body=c['conversation']
        for sk in sorted([k for k,v in body.items() if re.fullmatch(r'session_\d+',k) and isinstance(v,list)],key=lambda k:int(k[8:])):
            ids=[]; lines=[f"[SESSION_TIME] {str(body.get(sk+'_date_time','')).strip()}"]
            for t in body[sk]:
                eid=str(t['dia_id'])
                if eid in ids: raise ValueError('duplicate turn ID')
                ids.append(eid)
                line=f"[{eid}] {str(t['speaker']).strip()}: {str(t['text']).strip()}"
                cap=str(t.get('blip_caption','')).strip()
                if cap: line+=f" [Shared image: {cap}]"
                lines.append(line)
            jid=f'locomo-{ctx}-{sk}'
            if jid in jobs: raise ValueError('duplicate session ID')
            jobs[jid]=dict(job_id=jid,context_id=ctx,segment_id=sk,input='\n'.join(lines),evidence_ids=ids)
    return jobs

def split_map(splits,contexts):
    result={}
    if set(splits)!={'train','validation'}: raise ValueError('require train and validation splits')
    for name,ids in splits.items():
        if not ids: raise ValueError('empty split')
        for ctx in ids:
            if ctx in result: raise ValueError('conversation split overlap')
            result[ctx]=name
    if set(result)!=set(contexts): raise ValueError('split must cover source conversations exactly')
    return result

def validate_target(row,job,prompt):
    if row.get('schema_version')!='ood_atomic_memory_v3': raise ValueError('not native v3')
    if row.get('input_sha256')!=sha(job['input'].encode()): raise ValueError('source input hash mismatch')
    if row.get('prompt_sha256')!=sha(prompt.encode()): raise ValueError('extraction prompt mismatch')
    if row.get('finish_reason')!='stop': raise ValueError('truncated/non-stop extraction')
    if row.get('context_id')!=job['context_id'] or row.get('segment_id')!=job['segment_id']: raise ValueError('source identity mismatch')
    if not isinstance(row.get('memories'),list): raise ValueError('invalid memories')
    target=[]
    for m in row['memories']:
        if set(m)!=FIELDS|{'memory_key'}: raise ValueError('invalid v3 fields')
        if not isinstance(m['evidence_ids'],list) or not m['evidence_ids'] or not all(isinstance(e,str) for e in m['evidence_ids']): raise ValueError('invalid memory evidence')
        if not set(m['evidence_ids'])<=set(job['evidence_ids']): raise ValueError('invented memory evidence')
        if not all(isinstance(m[k],str) and m[k].strip() for k in ['subject','facet','attribute','value','statement']): raise ValueError('invalid memory text')
        if m['explicitness'] not in ['direct','strict_entailment']: raise ValueError('invalid explicitness')
        if m['polarity'] not in ['positive','negative']: raise ValueError('invalid polarity')
        if not isinstance(m['temporal'],dict) or set(m['temporal'])!={'source_expression','start','end','granularity','certainty'}: raise ValueError('invalid temporal schema')
        # memory_key is client-generated metadata, not part of the API target schema.
        target.append({k:v for k,v in m.items() if k!='memory_key'})
    return {'memories':target}

def evidence_ids(values):
    ids=[]
    for value in values:
        for part in re.split(r'[;,]',value):
            part=part.strip()
            if not re.fullmatch(r'D\d+:\d+',part): raise ValueError('malformed QA evidence')
            if part not in ids: ids.append(part)
    return ids

def prepare(source,memory_path,prompt_path,config,allow_partial=False):
    data=json.loads(source.read_text()); jobs=sessions(data)
    owners=split_map(config['splits'],[c['sample_id'] for c in data])
    prompt=writer_prompt(prompt_path); targets={}
    for row in jsonl(memory_path):
        jid=row['job_id']
        if jid in targets: raise ValueError('duplicate memory job')
        if jid not in jobs: raise ValueError('unknown memory job')
        targets[jid]=validate_target(row,jobs[jid],prompt)
    missing=sorted(set(jobs)-set(targets))
    if missing and not allow_partial: raise ValueError(f'missing {len(missing)}/{len(jobs)} session targets; use --allow-partial for pilot only')
    files={f'{s}/{kind}.jsonl':[] for s in owners.values() for kind in ['writer_inputs','writer_sft','qa']}
    files['source_sessions.jsonl']=[dict(j,split=owners[j['context_id']]) for j in jobs.values()]
    exclusions=[]
    for jid,target in targets.items():
        j=jobs[jid]; split=owners[j['context_id']]
        messages=[dict(role='system',content=prompt),dict(role='user',content=j['input'])]
        common=dict(id=jid,task_id=jid,context_id=j['context_id'],shared_context_id=j['context_id'],segment_id=j['segment_id'],split=split,input_sha256=sha(j['input'].encode()))
        files[f'{split}/writer_inputs.jsonl'].append(dict(common,writer_messages=messages))
        files[f'{split}/writer_sft.jsonl'].append(dict(common,messages=messages+[dict(role='assistant',content=compact(target))],
            metadata=dict(memory_variant='locomo-native-v3',assistant_loss_weights=[1.0],question_conditioned=False,memory_key_client_generated=True)))
    available={ctx:{e for j in jobs.values() if j['context_id']==ctx and j['job_id'] in targets for e in j['evidence_ids']} for ctx in owners}
    allids={ctx:{e for j in jobs.values() if j['context_id']==ctx for e in j['evidence_ids']} for ctx in owners}
    for c in data:
        ctx=c['sample_id']; split=owners[ctx]
        for index,q in enumerate(c['qa']):
            qid=f'{ctx}-q{index}'; reason=None
            if q['category'] not in config['qa_categories']: reason='category_excluded'
            elif 'answer' not in q or not str(q['answer']).strip(): reason='missing_answer'
            elif not q.get('evidence'): reason='no_gold_evidence'
            else:
                try: ev=evidence_ids(q['evidence'])
                except ValueError: reason='malformed_gold_evidence'
                if reason is None and not set(ev)<=allids[ctx]: reason='missing_gold_turn'
                if reason is None and not set(ev)<=available[ctx]: reason='target_coverage_incomplete'
            if reason:
                exclusions.append(dict(question_id=qid,split=split,reason=reason)); continue
            # Gold is confined to QA/reward records, never writer inputs.
            files[f'{split}/qa.jsonl'].append(dict(question_id=qid,context_id=ctx,shared_context_id=ctx,split=split,
                question=q['question'],answer=str(q['answer']),category=q['category'],question_type=CATEGORY[q['category']],
                evidence_ids=ev,source_session_ids=[jid for jid,j in jobs.items() if j['context_id']==ctx and jid in targets],
                coverage_mode='partial_development' if missing else 'complete',
                answer_format='short_answer',memory_origin_required='current_writer_checkpoint'))
    files['excluded_qa.jsonl']=exclusions
    manifest=dict(schema='locomo-v3-data-v1',source_sha256=sha(source.read_bytes()),memory_sha256=sha(memory_path.read_bytes()),
        extractor_script_sha256=sha(prompt_path.read_bytes()),prompt_sha256=sha(prompt.encode()),config=config,
        total_sessions=len(jobs),available_sessions=len(targets),missing_session_ids=missing,
        complete_data=not missing,training_ready=False,
        training_blockers=['legacy MCQ trainer requires explicit open-answer integration','reader_initialization compressor-only and on_policy_optimization LoRA scopes must be wired into trainer']+(['memory bank incomplete'] if missing else []),
        counts={k:len(v) for k,v in files.items()},api_targets_for_test=False,
        note='Dev pilot conversations already inspected; not an untouched evaluation split. direct is a model label, not verified truth.')
    return files,manifest

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',type=Path,default=Path(__file__).with_name('config.json'))
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--allow-partial',action='store_true')
    args=ap.parse_args(); config=json.loads(args.config.read_text())
    if args.output.exists(): raise FileExistsError('refusing to overwrite output')
    files,manifest=prepare(WORKSPACE/config['source'],WORKSPACE/config['memories'],WORKSPACE/config['extractor_script'],config,args.allow_partial)
    args.output.mkdir(parents=True)
    for name,rows in files.items():
        p=args.output/name; p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(''.join(compact(r)+'\n' for r in rows))
    (args.output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:manifest[k] for k in ['total_sessions','available_sessions','complete_data','training_ready','counts']},indent=2))
if __name__=='__main__': main()

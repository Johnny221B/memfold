"""Apply evaluation_token_accounting_standard without inference or judge calls."""
import hashlib
import json
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'runs/qwen3_4b_inter2'
STANDARD=ROOT.parent/'docs/token_accounting.md'

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def stats(values):
    return {'mean':float(np.mean(values)),'median':float(np.median(values)),
            'p95':float(np.percentile(values,95)),'total':int(sum(values))}

def main():
    reader_path=ROOT.parent/'models/Qwen3-4B'
    retriever_path=ROOT.parent/'models/sup-simcse-roberta-large'
    reader=AutoTokenizer.from_pretrained(reader_path,local_files_only=True)
    retriever=AutoTokenizer.from_pretrained(retriever_path,local_files_only=True)
    source=ROOT/'prepared/inputs.jsonl'
    inputs={(r['id'],r['method']):r for r in map(json.loads,source.read_text().splitlines())}
    answers=[json.loads(s) for s in (RUN/'responses.jsonl').read_text().splitlines()]
    assert len(inputs)==len(answers)==2000
    assert len({(r['id'],r['method']) for r in answers})==2000
    rows=[]
    for answer in answers:
        r=inputs[(answer['id'],answer['method'])]
        prompt=reader.apply_chat_template(r['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        assert hashlib.sha256(prompt.encode()).hexdigest()==answer['prompt_sha256']
        count=len(reader.encode(prompt,add_special_tokens=False))
        assert count==answer['input_tokens']
        output=answer['output_tokens'] # Captured from len(vLLM CompletionOutput.token_ids), not retokenized text.
        usage={k:0 for k in ['builder_input','builder_output','document_or_session_embed_input','query_embed_input',
                'compressor_text_input','compressor_effective_positions','weaver_positions','reasoner_positions','soft_input_positions']}
        usage.update(reader_prompt=count,reader_output=output,reader_only_total=count+output,strict_end_to_end_total=count+output)
        record={'dataset':'prefeval_implicit_persona','split':'train (evaluation-only)','backbone':'Qwen3-4B',
                'method':r['method'],'trial':0,'question_id':r['id'],'topic':r['topic'],
                'tokenizer_by_component':{'reader':str(reader_path),'retriever':str(retriever_path) if r['method']=='rag' else None},
                'token_usage':usage,'status':'exact','retrieval_source':r['retrieval_source']}
        if r['method']=='rag':
            # All original messages are charged, not just top-5. Shared 2-turn context is charged every time.
            texts=[m['content'] for m in r['messages'][1:-1]]
            doc_lengths=[len(retriever(t,truncation=True,max_length=512).input_ids) for t in texts]
            query_length=len(retriever(r['question'],truncation=True,max_length=512).input_ids)
            usage.update(document_or_session_embed_input=sum(doc_lengths),query_embed_input=query_length,
                         strict_end_to_end_total=sum(doc_lengths)+query_length+count+output)
            record['retriever_message_token_lengths']=doc_lengths
            record['retriever_query_token_length']=query_length
            record['status']='reconstructed_known_encoder_protocol'
            if r['retrieval_source']=='official_cache':
                # Upstream cache provides similarities but no encoding config/IDs. Never label those exact.
                record['estimated_token_usage']=usage.copy()
                for k in ['document_or_session_embed_input','query_embed_input','strict_end_to_end_total']:
                    usage[k]=None
                record['status']='estimated_encoder_protocol_for_official_cache'
                record['estimate_reason']='Original official encoder token IDs/config not published; use the local completion protocol: raw content, SimCSE tokenizer, special tokens included, max_length=512.'
        numeric=record.get('estimated_token_usage',usage)
        assert all(isinstance(v,int) and v>=0 for v in numeric.values())
        assert numeric['strict_end_to_end_total']==numeric['reader_only_total']+numeric['document_or_session_embed_input']+numeric['query_embed_input']
        rows.append(record)
    with (RUN/'token_usage.jsonl').open('w') as f:
        for r in rows:f.write(json.dumps(r)+'\n')
    summary={}
    for method in ['full_text','rag']:
        rr=[r for r in rows if r['method']==method]
        numeric=[r.get('estimated_token_usage',r['token_usage']) for r in rr]
        uncertain=any(r['status'].startswith('estimated') for r in rr)
        summary[method]={'n':len(rr),'reader_prompt':stats([u['reader_prompt'] for u in numeric]),
            'reader_output':stats([u['reader_output'] for u in numeric]),
            'reader_only_total':stats([u['reader_only_total'] for u in numeric]),
            'strict_end_to_end_total':None if uncertain else stats([u['strict_end_to_end_total'] for u in numeric]),
            'strict_end_to_end_estimate':stats([u['strict_end_to_end_total'] for u in numeric]) if uncertain else None,
            'encoder_document_input_estimate' if uncertain else 'encoder_document_input':stats([u['document_or_session_embed_input'] for u in numeric]),
            'encoder_query_input_estimate' if uncertain else 'encoder_query_input':stats([u['query_embed_input'] for u in numeric]),
            'estimated_rows':sum(r['status'].startswith('estimated') for r in rr)}
    (RUN/'token_usage_summary.json').write_text(json.dumps(summary,indent=2))
    hashes={str(p):sha(p) for p in [Path(__file__),STANDARD,source,RUN/'responses.jsonl',ROOT/'prepare.py',ROOT/'generate.py']}
    for directory in [reader_path,retriever_path]:
        for name in ['tokenizer.json','tokenizer_config.json','special_tokens_map.json','vocab.json','merges.txt','chat_template.jinja']:
            p=directory/name
            if p.exists():hashes[str(p)]=sha(p)
    manifest={'standard':str(STANDARD),'sha256':hashes,'no_cross_question_amortization':True,
              'reader_output_source':'stored runtime len(CompletionOutput.token_ids); includes generated EOS/stop token when emitted, per vLLM 0.11 detokenizer',
              'encoder_protocol':{'raw_message_content':True,'add_special_tokens':True,'max_length':512,'count_padding':False,'query': 'original question without RAG prefix'},
              'encoder_config_original_cache_verified':False,'excluded':['judge calls','training','smoke/debug/retries','CPU cosine similarity (not a model token operation)'],
              'reader_prompt_cache_discount':False,'note':'Full Text exact; RAG full-cohort strict total only in estimate field because 973 official-cache encoding configs are unavailable.'}
    (RUN/'token_usage_manifest.json').write_text(json.dumps(manifest,indent=2))
    full=summary['full_text'];rag=summary['rag'];strict=rag['strict_end_to_end_estimate']
    lines=['# PrefEval Qwen3-4B token accounting','',f'Standard: {STANDARD}','',
           'Single evaluation trial, 1,000 questions per method. No judge or model calls performed for this audit.','',
           '| Method | Reader input / question | Output / question | Reader-only / question | Strict E2E / question | E2E estimate / question |',
           '|---|---:|---:|---:|---:|---:|',
           f'| Full text | {full["reader_prompt"]["mean"]:.3f} | {full["reader_output"]["mean"]:.3f} | {full["reader_only_total"]["mean"]:.3f} | {full["strict_end_to_end_total"]["mean"]:.3f} | — |',
           f'| RAG | {rag["reader_prompt"]["mean"]:.3f} | {rag["reader_output"]["mean"]:.3f} | {rag["reader_only_total"]["mean"]:.3f} | Not fully recoverable | {strict["mean"]:.3f} |','',
           f'RAG encoding estimate per question: documents {rag["encoder_document_input_estimate"]["mean"]:.3f} + query {rag["encoder_query_input_estimate"]["mean"]:.3f}.',
           '', 'Official RAG keeps full history and repeats top-5 retrieved messages in the final query. All source embeddings and query encoding are charged per question, even when served from cache. Shared context and prefix cache are not amortized.',
           '', 'Reader counts are verified against the runtime prompt hash and tokenizer. Outputs use recorded runtime token-ID lengths, including emitted stop/EOS tokens. The original official embedding cache lacks token IDs and encoding configuration for 973 rows; its encoder cost is reconstructed under our known SimCSE completion protocol (raw content, special tokens, max_length=512). Those costs remain explicitly estimates, not exact strict E2E measurements.',
           '', 'Median/p95/totals and per-question component provenance are in token_usage_summary.json and token_usage.jsonl. This is model token-equivalent, not API billing or FLOPs.']
    (RUN/'TOKEN_USAGE.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()

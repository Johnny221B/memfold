"""Official Gemini 3.8 client following secrets/API_USAGE.md section 7.

All users of this module share a file-locked index and global request interval.
No keys, authentication headers, or raw HTTP errors are logged.
"""
import argparse
import email.utils
import fcntl
import json
import os
import random
import time
from pathlib import Path

import httpx

ROOT=Path(__file__).resolve().parent.parent
MODEL='gemini-3.8-flash'
ENDPOINT=f'https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent'
STATE=ROOT/'secrets/gemini_dispatch_state.json'


def retry_delay(response, attempt):
    delays=[min(60,2**(attempt+1))+random.uniform(0,1)]
    value=response.headers.get('Retry-After')
    if value:
        try:
            delays.append(float(value))
        except ValueError:
            try:
                delays.append(email.utils.parsedate_to_datetime(value).timestamp()-time.time())
            except (ValueError,TypeError):
                pass
    try:
        for detail in response.json().get('error',{}).get('details',[]):
            if detail.get('@type','').endswith('RetryInfo'):
                delays.append(float(detail['retryDelay'].removesuffix('s')))
    except (ValueError,KeyError,TypeError):
        pass
    return max(delays)


def daily_quota(response):
    try:
        error=json.dumps(response.json().get('error',{})).lower()
    except ValueError:
        return False
    return any(x in error for x in ['perday','per_day','per day','daily'])


def body_text(raw):
    if raw.get('promptFeedback',{}).get('blockReason'):
        raise ValueError('Gemini prompt blocked')
    candidate=raw.get('candidates',[{}])[0]
    if candidate.get('finishReason')!='STOP':
        raise ValueError('Gemini non-STOP response')
    text=''.join(p.get('text','') for p in candidate.get('content',{}).get('parts',[]) if not p.get('thought',False))
    if not text.strip():
        raise ValueError('Gemini empty non-thought response')
    if not raw.get('modelVersion','').startswith(MODEL):
        raise ValueError('unexpected Gemini modelVersion; no fallback allowed')
    return text


class GeminiClient:
    def __init__(self):
        self.keys=[x.strip() for x in (ROOT/'secrets/gemini_key').read_text().splitlines() if x.strip()]
        if not self.keys or len(self.keys)!=len(set(self.keys)):
            raise ValueError('Gemini key file must contain distinct nonempty lines')

    def generate(self, text, *, system=None, max_tokens=8192, json_mode=True, validator=None):
        body={'contents':[{'role':'user','parts':[{'text':text}]}],
              'generationConfig':{'temperature':0,'maxOutputTokens':max_tokens,
                                  'thinkingConfig':{'thinkingLevel':'LOW'}}}
        if system:
            body['systemInstruction']={'parts':[{'text':system}]}
        if json_mode:
            body['generationConfig']['responseMimeType']='application/json'
        # Hold the shared lock throughout an API call and any quota backoff.
        # This intentionally serializes callers even across processes/threads.
        fd=os.open(STATE,os.O_CREAT|os.O_RDWR,0o600)
        with os.fdopen(fd,'r+') as handle:
            fcntl.flock(handle,fcntl.LOCK_EX)
            data=handle.read()
            state=json.loads(data) if data.strip() else {'next_index':0,'next_allowed':0,'paused_keys':[]}

            def save():
                handle.seek(0)
                json.dump(state,handle)
                handle.truncate()
                handle.flush()
                os.fsync(handle.fileno())

            if state.get('daily_quota_blocked'):
                raise RuntimeError('Gemini daily quota blocked; wait for quota recovery and explicit reset')
            with httpx.Client(timeout=120) as client:
                for attempt in range(6):
                    time.sleep(max(0,state.get('next_allowed',0)-time.time()))
                    available=[(state['next_index']+i)%len(self.keys) for i in range(len(self.keys))
                               if (state['next_index']+i)%len(self.keys) not in state['paused_keys']]
                    if not available:
                        raise RuntimeError('all Gemini keys paused; check authentication/configuration')
                    index=available[0]
                    state['next_index']=(index+1)%len(self.keys)
                    # Persist advancement before the request, including failed attempts.
                    state['next_allowed']=time.time()+6
                    save()
                    try:
                        response=client.post(ENDPOINT,headers={'x-goog-api-key':self.keys[index]},json=body)
                    except httpx.RequestError:
                        delay=min(60,2**(attempt+1))+random.uniform(0,1)
                        state['next_allowed']=time.time()+max(6,delay)
                        save()
                        print(json.dumps(dict(key_index=index+1,model=MODEL,status='network_error',attempt=attempt+1)),flush=True)
                        continue
                    state['next_allowed']=time.time()+6
                    status=response.status_code
                    print(json.dumps(dict(key_index=index+1,model=MODEL,http_status=status,attempt=attempt+1)),flush=True)
                    if status in [401,403]:
                        state['paused_keys'].append(index)
                        save()
                        raise RuntimeError(f'Gemini HTTP {status}; key index {index+1} paused, check permissions')
                    if status==429:
                        if daily_quota(response):
                            state['daily_quota_blocked']=True
                            save()
                            raise RuntimeError('Gemini daily quota exhausted; stopped without scanning keys')
                        state['next_allowed']=time.time()+max(6,retry_delay(response,attempt))
                        save()
                        continue
                    if status in [500,502,503,504]:
                        state['next_allowed']=time.time()+max(6,retry_delay(response,attempt))
                        save()
                        continue
                    if status!=200:
                        save()
                        raise RuntimeError(f'Gemini HTTP {status}; check model/endpoint/parameters; no fallback')
                    try:
                        raw=response.json()
                        content=body_text(raw)
                        if validator:
                            validator(content)
                    except (ValueError,KeyError,TypeError,IndexError):
                        state['next_allowed']=time.time()+max(6,min(60,2**(attempt+1)))
                        save()
                        continue
                    save()
                    print(json.dumps(dict(key_index=index+1,model=raw['modelVersion'],attempt=attempt+1,
                                          usage=raw.get('usageMetadata',{}))),flush=True)
                    return content,raw,dict(key_index=index+1,attempt=attempt+1,http_status=status)
        raise RuntimeError('Gemini request failed after 6 attempts; no model fallback')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preflight',action='store_true',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():
        raise FileExistsError('refusing overwrite preflight report')
    def check(text):
        if text.strip()!='OK':
            raise ValueError('preflight did not return OK')
    content,raw,meta=GeminiClient().generate('Reply with exactly OK.',max_tokens=256,json_mode=False,validator=check)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(dict(model=MODEL,thinking='LOW',body=content,
        model_version=raw['modelVersion'],finish_reason=raw['candidates'][0]['finishReason'],
        usage=raw.get('usageMetadata'),**meta),indent=2)+'\n')


if __name__=='__main__':
    main()

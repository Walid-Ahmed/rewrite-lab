"""Paired detector evaluation of actual rewrites, with explicit missing scores."""
import argparse
import hashlib
import json
import math
import statistics
import subprocess
from pathlib import Path

NOTE = ('AI-content percentages are detector scores, not measured fractions of AI-authored text. '
        'A lower score does not establish factual accuracy or readability.')

def text_id(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def checked_score(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError('Detector scores must be numeric percentages.')
    if not math.isfinite(value) or not 0 <= value <= 100:
        raise ValueError('Detector scores must be finite and in [0, 100].')
    return float(value)

class CommandDetector:
    """An argv command receives {texts:[...]} on stdin, returns {scores:[0..100]}.

    Configure provider-specific API code separately; keys remain in environment.
    Commands are executed without a shell. No provider or label mapping is guessed.
    """
    def __init__(self, argv, detector_id, cache_path):
        if not isinstance(argv, list) or not argv or not all(isinstance(x,str) for x in argv):
            raise ValueError('Detector command must be a nonempty JSON array of argv strings.')
        if not detector_id:
            raise ValueError('Set a detector ID including provider/model version.')
        self.argv, self.detector_id, self.path = argv, detector_id, Path(cache_path)
        self.cache = json.loads(self.path.read_text()) if self.path.exists() else {}
    def score(self, text):
        key = text_id(self.detector_id+'\0'+text)
        if key not in self.cache:
            result = subprocess.run(self.argv, input=json.dumps({'texts':[text]}), text=True,
                                    capture_output=True, timeout=180, check=False)
            if result.returncode:
                # Do not copy stderr: provider logs could contain credentials.
                raise RuntimeError(f'Detector command failed with exit code {result.returncode}.')
            data=json.loads(result.stdout)
            if not isinstance(data.get('scores'),list) or len(data['scores']) != 1:
                raise ValueError('Detector must return exactly one score for each supplied text.')
            self.cache[key]=checked_score(data['scores'][0])
            self.path.parent.mkdir(parents=True,exist_ok=True)
            temp=self.path.with_suffix('.tmp')
            temp.write_text(json.dumps(self.cache,indent=2))
            temp.replace(self.path)
        return checked_score(self.cache[key])

class FileDetector:
    """Import exact-text scores recorded through the signed-in detector UI."""
    def __init__(self,path,detector_id=None):
        data=json.loads(Path(path).read_text())
        self.detector_id=data['detector_id']
        if detector_id and self.detector_id!=detector_id:
            raise ValueError('Score-file detector ID does not match requested detector.')
        self.minimum_words=data.get('minimum_words',0)
        self.scores={text_id(r['text']):checked_score(r['score_pct']) for r in data['scores']}
    def score(self,text):
        return self.scores[text_id(text)]

def validation_rows(data_dir):
    path=Path(data_dir)/'valid.jsonl'
    manifest_path=Path(data_dir)/'split-manifest.json'
    manifest=json.loads(manifest_path.read_text()).get('valid',[]) if manifest_path.exists() else []
    rows=[]
    for i,line in enumerate(path.read_text().splitlines()):
        r=json.loads(line)
        messages=r['messages']
        user=next(m['content'] for m in messages if m['role']=='user')
        target=next(m['content'] for m in messages if m['role']=='assistant')
        system=next((m['content'] for m in messages if m['role']=='system'),None)
        rows.append({'id':text_id((system or '')+'\0'+user),'source_id':text_id(user),'index':i,'before':user,'reference':target,
                     'system_prompt':next((m['content'] for m in messages if m['role']=='system'),None),
                     'source_reference':manifest[i] if i<len(manifest) else None})
    return rows

def generate_rewrite(model, tokenizer, text, max_tokens, system_prompt=None):
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler
    from train_qwen3_humanizer import SYSTEM
    prompt=tokenizer.apply_chat_template([{'role':'system','content':system_prompt or SYSTEM},{'role':'user','content':text}],
              tokenize=False,add_generation_prompt=True,enable_thinking=False)
    output=[]
    finish=None
    for chunk in stream_generate(model,tokenizer,prompt=prompt,max_tokens=max_tokens,sampler=make_sampler(temp=0)):
        output.append(chunk.text)
        finish=chunk.finish_reason
    return ''.join(output).strip(),finish

def summarize(rows):
    complete=[r for r in rows if r.get('status')=='scored']
    summary={'requested_samples':len(rows),'scored_pairs':len(complete),
             'truncated_outputs':sum(r.get('finish_reason')=='length' for r in rows),
             'missing_or_failed_pairs':len(rows)-len(complete),
             'below_detector_word_minimum':sum(r.get('status')=='below_detector_word_minimum' for r in rows),
             'score_coverage_fraction':len(complete)/len(rows) if rows else 0}
    if complete:
        before=[r['ai_content_before_pct'] for r in complete]
        after=[r['ai_content_after_pct'] for r in complete]
        drops=[b-a for b,a in zip(before,after)]
        summary.update(ai_content_before_pct=statistics.mean(before),
                       ai_content_after_pct=statistics.mean(after),
                       ai_content_reduction_pp=statistics.mean(drops),
                       median_reduction_pp=statistics.median(drops),
                       improved_fraction=sum(d>0 for d in drops)/len(drops),
                       worsened_fraction=sum(d<0 for d in drops)/len(drops),
                       unchanged_fraction=sum(d==0 for d in drops)/len(drops))
    return summary

def score_rows(rows,detector):
    for r in rows:
        r.update(ai_content_before_pct=None,ai_content_after_pct=None,ai_content_reduction_pp=None)
        if r.get('finish_reason')!='stop' or not r.get('after','').strip():
            r['status']='generation_incomplete'
            continue
        if detector is None:
            r['status']='detector_pending'
            continue
        if any(len(r[k].split())<getattr(detector,'minimum_words',0) for k in ['before','after']):
            r['status']='below_detector_word_minimum'
            continue
        try:
            before,after=detector.score(r['before']),detector.score(r['after'])
            r.update(ai_content_before_pct=before,ai_content_after_pct=after,
                     ai_content_reduction_pp=before-after,status='scored')
            r.pop('error',None)
        except KeyError:
            r['status']='detector_pending'
        except Exception as exc:
            r.update(status='detector_failed',error=type(exc).__name__)
    return rows

def write_report(rows,folder,iteration,detector_id,scope,run=None):
    folder=Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    summary=summarize(rows)
    report={'iteration':iteration,'scope':scope,'detector_id':detector_id,'note':NOTE,
            'summary':summary,'samples':rows}
    (folder/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    with (folder/'samples.jsonl').open('w') as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False)+'\n')
    def fmt(x): return 'pending' if x is None else f'{x:.2f}'
    lines=['# Validation rewrite evaluation','',NOTE,'',f'Detector: {detector_id or "pending selection"}',
           f'Scope: {scope}; iteration: {iteration}',f'Paired scores: {summary["scored_pairs"]}/{len(rows)}','',
           '| Mean before (%) | Mean after (%) | Reduction (percentage points) |',
           '| ---: | ---: | ---: |',
           f'| {fmt(summary.get("ai_content_before_pct"))} | {fmt(summary.get("ai_content_after_pct"))} | {fmt(summary.get("ai_content_reduction_pp"))} |','']
    # Include all examples so failures and regressions remain visible.
    for r in rows:
        lines += [f'## Sample {r["index"]+1}', '',f'Status: {r["status"]}',
                  f'Before: {fmt(r["ai_content_before_pct"])}%; after: {fmt(r["ai_content_after_pct"])}%; reduction: {fmt(r["ai_content_reduction_pp"])} pp','',
                  '**Original**','',r['before'],'','**Model rewrite**','',r['after'],'',
                  '**Dataset reference (not the model output)**','',r['reference'],'']
    (folder/'report.md').write_text('\n'.join(lines))
    if run:
        import wandb
        metrics={f'validation_ai/{scope}/{k}':v for k,v in summary.items()}
        metrics['iteration']=iteration
        columns=['id','index','before','after','reference','ai_content_before_pct','ai_content_after_pct','ai_content_reduction_pp','status']
        metrics[f'validation_ai/{scope}/samples']=wandb.Table(columns=columns,data=[[r.get(k) for k in columns] for r in rows])
        run.log(metrics)
        run.summary[f'validation_ai/{scope}/detector_id']=detector_id or 'pending'
    return report

class ValidationAIEvaluator:
    def __init__(self,model,tokenizer,data_dir,output_dir,detector=None,max_tokens=768):
        self.model,self.tokenizer=model,tokenizer
        self.rows=validation_rows(data_dir)
        self.out=Path(output_dir)
        self.detector,self.max_tokens=detector,max_tokens
    def evaluate(self,iteration,limit=None,scope='final',run=None):
        import mlx.core as mx
        import numpy as np
        # Evaluation must not change training mode or random streams.
        rng=np.random.get_state()
        mx.eval(mx.random.state)
        metal_rng=list(mx.random.state)
        training=self.model.training
        self.model.eval()
        folder=self.out/f'{scope}-{iteration:06d}'
        folder.mkdir(parents=True,exist_ok=True)
        selected=self.rows if limit is None else self.rows[:limit]
        rows=[]
        try:
            with (folder/'generations.jsonl').open('w') as f:
                for source in selected:
                    after,finish=generate_rewrite(self.model,self.tokenizer,source['before'],self.max_tokens,system_prompt=source.get('system_prompt'))
                    row=dict(source,after=after,finish_reason=finish)
                    rows.append(row)
                    f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush()
                    print(f'Validation rewrite {len(rows)}/{len(selected)} ({finish})',flush=True)
            score_rows(rows,self.detector)
            return write_report(rows,folder,iteration,getattr(self.detector,'detector_id',None),scope,run)
        finally:
            self.model.train(training)
            np.random.set_state(rng)
            mx.random.state=metal_rng
            mx.clear_cache()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',default='mlx-data')
    p.add_argument('--adapter',default='mlx-qwen3-humanizer-epoch1')
    p.add_argument('--model',default='mlx-community/Qwen3-1.7B-4bit')
    p.add_argument('--output',default='evaluation/epoch1')
    p.add_argument('--iteration',type=int,default=0)
    p.add_argument('--limit',type=int)
    p.add_argument('--max-tokens',type=int,default=768)
    p.add_argument('--detector-command',help='JSON argv array; command consumes {texts:[...]} and returns {scores:[percent,...]}')
    p.add_argument('--detector-id')
    p.add_argument('--detector-scores',type=Path,help='Import browser scores by exact text from a saved JSON file')
    p.add_argument('--score-existing',type=Path,help='Score a saved generations.jsonl without regenerating rewrites')
    p.add_argument('--wandb',action='store_true')
    p.add_argument('--offline',action='store_true')
    p.add_argument('--entity')
    p.add_argument('--wandb-run-id',help='Append evaluation metrics to a specified W&B run (including a training run)')
    p.add_argument('--scope',choices=['periodic','final'],default='final')
    a=p.parse_args()
    if a.limit is not None and a.limit<1: p.error('--limit must be positive')
    if a.max_tokens<1: p.error('--max-tokens must be positive')
    if a.detector_command and not a.detector_id: p.error('Set --detector-id with --detector-command')
    if a.detector_command and a.detector_scores: p.error('Select command scoring or a score file, not both')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    detector=CommandDetector(json.loads(a.detector_command),a.detector_id,out/'detector-cache.json') if a.detector_command else (FileDetector(a.detector_scores,a.detector_id) if a.detector_scores else None)
    if detector: a.detector_id=detector.detector_id
    run=None
    if a.wandb:
        import wandb
        run=wandb.init(project='qwen3-humanizer',entity=a.entity,id=a.wandb_run_id,resume='must' if a.wandb_run_id else None,name=None if a.wandb_run_id else out.name,job_type='validation-ai-evaluation',
            dir=str(out.resolve()),mode='offline' if a.offline else 'online',
            config={'adapter':a.adapter,'model':a.model,'data':a.data,'detector_id':a.detector_id,'max_tokens':a.max_tokens})
        run.define_metric('iteration');run.define_metric('validation_ai/*',step_metric='iteration')
    try:
        if a.score_existing:
            rows=[json.loads(x) for x in a.score_existing.read_text().splitlines()]
            report=write_report(score_rows(rows,detector),out,a.iteration,a.detector_id,a.scope,run)
        else:
            from mlx_lm import load
            model,tokenizer=load(a.model,adapter_path=a.adapter)
            evaluator=ValidationAIEvaluator(model,tokenizer,a.data,out,detector,a.max_tokens)
            report=evaluator.evaluate(a.iteration,a.limit,scope=a.scope,run=run)
        print(json.dumps(report['summary'],indent=2))
        if run: print('W&B run:',run.url)
    finally:
        if run: run.finish()

if __name__=='__main__':
    main()

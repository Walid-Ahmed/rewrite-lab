"""Local rewrite metrics and separately attributed, exact-text semantic reviews.

Literal preservation flags are screening signals, never a factual accuracy score.
No language model, paid API, detector, or network request is used by this script.
"""
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
from difflib import SequenceMatcher
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import statistics

VERSION='rewrite_quality_v1'
NOTE=('English readability formulas estimate linguistic complexity, not quality. Repeated technical phrases can be appropriate. '
      'Literal preservation checks cannot establish semantic fidelity or real-world truth. Semantic reviews are separately attributed judgments, not automatic facts.')
WORD=re.compile(r"[A-Za-z0-9]+(?:[’'_-][A-Za-z0-9]+)*")
NUMBER=re.compile(r'(?<![\w.])[-+]?\d+(?:,\d{3})*(?:\.\d+)*(?:\s*%|\b)')
IDENTIFIER=re.compile(r'`[^`\n]+`|https?://[^\s)]+|\b\w+_\w+\b|\b[A-Za-z]+(?:\.[A-Za-z_]\w*)+\b|\b(?:[A-Z]{2,}[A-Za-z0-9]*|[A-Z][a-z]+(?:[A-Z][A-Za-z0-9]*)+)\b|\[(?:\d+(?:\s*[,–-]\s*\d+)*)\]')
NEGATION={'not','no','never','cannot',"can't","don't","doesn't","isn't","aren't","won't",'without','neither','nor'}
QUALIFIERS={'possibility':{'may','might','could','can'},'necessity':{'must','required','requires','mandatory'},'scope':{'only','always','all','every','usually','often','sometimes','some'}}


def digest(text):return hashlib.sha256(text.encode()).hexdigest()
def words(text):return WORD.findall(text.lower().replace('’',"'"))
def sentences(text):return [x.strip() for x in re.split(r'(?<=[.!?])\s+(?=[A-Z0-9`])|\n+',text) if x.strip()]
def average(values):return statistics.mean(values) if values else None

def repetition(text,n=4):
    tokens=words(text);grams=[tuple(tokens[i:i+n]) for i in range(max(0,len(tokens)-n+1))];counts=Counter(grams)
    repeated=[{'phrase':' '.join(g),'count':c} for g,c in counts.most_common() if c>1]
    sentence_counts=Counter(' '.join(words(s)) for s in sentences(text) if words(s))
    return {'ngram_size':n,'ngram_windows':len(grams),'repeated_ngram_excess_fraction':sum(c-1 for c in counts.values())/len(grams) if grams else None,
            'repeated_phrases':repeated,'duplicate_sentence_excess_count':sum(c-1 for c in sentence_counts.values())}


def readability(text):
    if not words(text):return {'status':'empty','flesch_reading_ease':None,'flesch_kincaid_grade':None,'words_per_sentence':None}
    import textstat
    textstat.set_lang('en_US')
    return {'status':'measured','flesch_reading_ease':textstat.flesch_reading_ease(text),
            'flesch_kincaid_grade':textstat.flesch_kincaid_grade(text),
            'words_per_sentence':textstat.lexicon_count(text)/max(1,textstat.sentence_count(text))}


def literal_checks(before,after):
    source_numbers={m.group().replace(',','').replace(' ','') for m in NUMBER.finditer(before)}
    output_numbers={m.group().replace(',','').replace(' ','') for m in NUMBER.finditer(after)}
    identifiers={m.group() for m in IDENTIFIER.finditer(before)};output_identifiers={m.group() for m in IDENTIFIER.finditer(after)}
    # Require exact token/span preservation; an occurrence inside another identifier does not suffice.
    missing_identifiers=sorted(identifiers-output_identifiers)
    source_words=Counter(words(before));output_words=Counter(words(after))
    before_neg=sum(source_words[w] for w in NEGATION);after_neg=sum(output_words[w] for w in NEGATION)
    changed_qualifiers={kind:{'before':sum(source_words[w] for w in terms),'after':sum(output_words[w] for w in terms)}
                        for kind,terms in QUALIFIERS.items() if sum(source_words[w] for w in terms)!=sum(output_words[w] for w in terms)}
    flags=[]
    for label,items in [('missing_numbers',sorted(source_numbers-output_numbers)),('added_numbers',sorted(output_numbers-source_numbers)),
                        ('missing_technical_spans',missing_identifiers),('added_technical_spans',sorted(output_identifiers-identifiers))]:
        if items:flags.append({'kind':label,'values':items})
    if before_neg!=after_neg:flags.append({'kind':'negation_marker_count_changed','before':before_neg,'after':after_neg})
    if changed_qualifiers:flags.append({'kind':'qualifier_marker_counts_changed','groups':changed_qualifiers})
    return {'status':'screening_only_semantic_review_required','flags':flags,
            'source_numbers':sorted(source_numbers),'output_numbers':sorted(output_numbers),
            'number_token_retention':len(source_numbers&output_numbers)/len(source_numbers) if source_numbers else None,
            'source_technical_spans':sorted(identifiers),'missing_technical_spans':missing_identifiers,
            'technical_span_retention':(len(identifiers)-len(missing_identifiers))/len(identifiers) if identifiers else None,
            'factual_accuracy_pct':None}


def score_pair(row):
    before,after=row['before'],row['after']
    if not isinstance(before,str) or not before.strip() or not isinstance(after,str):raise ValueError('Expected nonempty before text and string after text')
    before_words,after_words=words(before),words(after)
    if not before_words:raise ValueError('Original needs lexical tokens')
    result={**row,'source_id':digest(before),'before_sha256':digest(before),'after_sha256':digest(after),
            'generation_complete':row.get('finish_reason','unknown')=='stop' and bool(after_words),
            'before_metrics':{'readability':readability(before),'repetition':repetition(before),'word_count':len(before_words)},
            'after_metrics':{'readability':readability(after),'repetition':repetition(after),'word_count':len(after_words)},
            'length_ratio':len(after_words)/len(before_words),
            'word_sequence_edit_fraction':1-SequenceMatcher(None,before_words,after_words,autojunk=False).ratio(),
            'literal_preservation':literal_checks(before,after),'semantic_review':{'status':'pending','factual_accuracy_pct':None}}
    for metric in ('flesch_reading_ease','flesch_kincaid_grade','words_per_sentence'):
        a=result['after_metrics']['readability'][metric];b=result['before_metrics']['readability'][metric]
        result[metric+'_change']=a-b if a is not None and b is not None else None
    return result


def attach_reviews(rows,reviews):
    index={(r['source_id'],r['model_label']):r for r in rows};seen=set()
    for review in reviews:
        key=(review['source_id'],review['model_label'])
        if key in seen:raise ValueError('Duplicate semantic review')
        seen.add(key)
        if key not in index:raise ValueError('Review has no corresponding generated output')
        row=index[key]
        if review.get('before_sha256')!=row['before_sha256'] or review.get('after_sha256')!=row['after_sha256']:raise ValueError('Semantic review exact-text hash mismatch')
        if not review.get('reviewer') or review.get('reviewer_type') not in ('agent_llm','human','independent_llm'):raise ValueError('Attribute the semantic reviewer explicitly')
        if review.get('verdict') not in ('preserved','minor_loss','material_error','uncertain'):raise ValueError('Unknown semantic-review verdict')
        if not review.get('claim_checks'):raise ValueError('Semantic review needs explicit claim checks')
        for claim in review['claim_checks']:
            if claim.get('status') not in ('preserved','omitted','changed','uncertain'):raise ValueError('Unknown claim-check status')
            quote=claim.get('source_quote');output_quote=claim.get('output_quote')
            if not quote or quote not in row['before']:raise ValueError('Claim needs a verbatim source evidence span')
            if output_quote and output_quote not in row['after']:raise ValueError('Claim output evidence is not in the exact rewrite')
            if claim['status']=='preserved' and not output_quote:raise ValueError('Preserved claim needs output evidence')
        row['semantic_review']={**review,'status':'reviewed','scope':'Consistency with supplied paragraph; independent real-world fact checking not performed','factual_accuracy_pct':None}
    return rows


def summarize(rows):
    labels=sorted({r['model_label'] for r in rows});source_sets={label:{r['source_id'] for r in rows if r['model_label']==label and r['generation_complete']} for label in labels}
    shared=set.intersection(*source_sets.values()) if source_sets else set();summary={}
    for label in labels:
        all_rows=[r for r in rows if r['model_label']==label];complete=[r for r in all_rows if r['generation_complete'] and r['source_id'] in shared]
        summary[label]={'requested':len(all_rows),'completed':sum(r['generation_complete'] for r in all_rows),'matched_comparison_samples':len(complete),
            'mean_flesch_reading_ease_before':average([r['before_metrics']['readability']['flesch_reading_ease'] for r in complete]),
            'mean_flesch_reading_ease_after':average([r['after_metrics']['readability']['flesch_reading_ease'] for r in complete]),
            'mean_grade_after':average([r['after_metrics']['readability']['flesch_kincaid_grade'] for r in complete]),
            'mean_length_ratio':average([r['length_ratio'] for r in complete]),
            'mean_word_sequence_edit_fraction':average([r['word_sequence_edit_fraction'] for r in complete]),
            'mean_repeated_4gram_excess_fraction':average([r['after_metrics']['repetition']['repeated_ngram_excess_fraction'] for r in complete if r['after_metrics']['repetition']['repeated_ngram_excess_fraction'] is not None]),
            'literal_flagged_samples':sum(bool(r['literal_preservation']['flags']) for r in complete),
            'semantic_reviews':dict(Counter(r['semantic_review'].get('verdict','pending') for r in complete)),
            'factual_accuracy_pct':None}
    return summary


def write_report(rows,out,metadata):
    out.mkdir(parents=True,exist_ok=True)
    report={'metric_version':VERSION,'created_at':datetime.now(timezone.utc).isoformat(),'note':NOTE,'provenance':metadata,
            'summary':summarize(rows),'samples':rows,'sources':['https://github.com/textstat/textstat','https://aclanthology.org/2022.naacl-main.187/']}
    temp=out/'report.tmp';temp.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n');temp.replace(out/'report.json')
    def fmt(value):return 'n/a' if value is None else f'{value:.2f}'
    lines=['# Rewrite quality comparison','',NOTE,'','| Model | Matched samples | Reading ease ↑ | Grade estimate | Length ratio | Repeated 4-grams ↓ | Literal flags |',
           '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for label,s in report['summary'].items():lines.append(f"| {label} | {s['matched_comparison_samples']} | {fmt(s['mean_flesch_reading_ease_after'])} | {fmt(s['mean_grade_after'])} | {fmt(s['mean_length_ratio'])} | {fmt(s['mean_repeated_4gram_excess_fraction'])} | {s['literal_flagged_samples']} |")
    lines+=['','Grade estimates describe complexity; lower is not universally better for textbook prose. Literal flags are review requests, not factual errors. The pilot does not measure official Detector A scores.','']
    pending=[]
    for i,r in enumerate(rows):
        lines+=[f"## {i+1}. {r['model_label']} · {r.get('topic','sample')}",'','**Original**',r['before'],'','**Rewrite**',r['after'],'',
                f"Generation complete: {r['generation_complete']}. Literal flags: {json.dumps(r['literal_preservation']['flags'],ensure_ascii=False)}.",
                f"Semantic review: {json.dumps(r['semantic_review'],ensure_ascii=False)}",'']
        if r['semantic_review']['status']=='pending':pending.append({k:r[k] for k in ('source_id','model_label','before_sha256','after_sha256','before','after')})
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    (out/'semantic-review-pending.json').write_text(json.dumps(pending,indent=2,ensure_ascii=False)+'\n')
    return report


def plot_report(report,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=list(report['summary']);summary=report['summary'];fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,key,title in zip(axes,('mean_flesch_reading_ease_after','mean_length_ratio','mean_repeated_4gram_excess_fraction'),('Reading ease (higher = easier)','Rewrite/original word count','Repeated 4-gram excess (%)')):
        values=[summary[label][key] for label in labels]
        if key=='mean_repeated_4gram_excess_fraction':values=[v*100 if v is not None else None for v in values]
        if all(v is not None for v in values):
            ax.bar(labels,values,color=['#7660b5','#2b8e7b'][:len(labels)]);ax.bar_label(ax.containers[0],fmt='%.2f',padding=3)
        ax.margins(y=.15);ax.set_title(title);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Matched rewrite pilot — complexity and repetition, not factual accuracy');fig.tight_layout();fig.savefig(out/'quality-comparison.png',dpi=180);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True,help='Generated snapshot with samples: before, after, model_label, finish_reason');p.add_argument('--output',type=Path,required=True);p.add_argument('--reviews',type=Path);p.add_argument('--plot',action='store_true');a=p.parse_args()
    snapshot=json.loads(a.input.read_text());samples=snapshot['samples'];keys=[(digest(r['before']),r['model_label']) for r in samples]
    if not samples or len(keys)!=len(set(keys)):p.error('Use nonempty, unique source/model pairs')
    rows=[score_pair(r) for r in samples]
    if a.reviews:attach_reviews(rows,json.loads(a.reviews.read_text()))
    metadata={**snapshot.get('provenance',{}),'input':str(a.input.resolve()),'input_sha256':digest(a.input.read_text()),'readability_backend':'textstat','textstat_version':importlib.metadata.version('textstat'),'semantic_judge_called_by_script':False}
    report=write_report(rows,a.output,metadata)
    if a.plot:plot_report(report,a.output)
    print(json.dumps(report['summary'],indent=2));print('Report: '+str(a.output/'report.md'))


if __name__=='__main__':main()

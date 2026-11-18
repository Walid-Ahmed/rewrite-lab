"""Fine-tune BERT + sigmoid regression head on Detector A scores on Apple MPS."""
import argparse, json, math, random, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel

class ScoreBert(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder=AutoModel.from_pretrained(model_name)
        self.head=nn.Sequential(nn.Dropout(.1),nn.Linear(self.encoder.config.hidden_size,1))
    def forward(self,**inputs):
        return torch.sigmoid(self.head(self.encoder(**inputs).pooler_output)).squeeze(-1)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,help='Managed Model 2 export; defaults to active registry');p.add_argument('--output',type=Path)
    p.add_argument('--dataset-registry',type=Path);p.add_argument('--check-data-only',action='store_true')
    p.add_argument('--model',default='google-bert/bert-base-uncased')
    p.add_argument('--epochs',type=int,default=1);p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--lr',type=float,default=2e-5);p.add_argument('--max-length',type=int,default=512)
    p.add_argument('--seed',type=int,default=3407);p.add_argument('--resume',type=Path)
    p.add_argument('--latest-checkpoint-only',action='store_true',help='Overwrite checkpoint.pt after each epoch; skip step snapshots')
    p.add_argument('--no-wandb',action='store_true');a=p.parse_args()
    from training_data import load_training_data,save_dataset_provenance
    bundle=load_training_data('model2',a.data,a.dataset_registry);a.data=bundle.path
    if a.check_data_only:
        print(json.dumps(bundle.metadata,indent=2));return
    if a.output is None:p.error('--output is required for training')
    if a.output.exists():raise ValueError('Use a new output directory')
    if a.epochs<1 or a.batch_size<1 or not 1<=a.max_length<=512:raise ValueError('Invalid training options')
    a.output.mkdir(parents=True);random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    torch.set_num_threads(4)
    device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    tokenizer=AutoTokenizer.from_pretrained(a.model)
    train=bundle.splits['train'];valid=bundle.splits['validation']
    assert not {r['group_id'] for r in train}&{r['group_id'] for r in valid}
    def encode(rows):
        return [{'inputs':tokenizer(r['text'],truncation=True,max_length=a.max_length),'label':r['score_pct']/100} for r in rows]
    def collate(rows):
        batch=tokenizer.pad([x['inputs'] for x in rows],return_tensors='pt')
        return {k:v.to(device) for k,v in batch.items()},torch.tensor([x['label'] for x in rows],dtype=torch.float32,device=device)
    tr=DataLoader(encode(train),batch_size=a.batch_size,shuffle=True,collate_fn=collate)
    va=DataLoader(encode(valid),batch_size=a.batch_size,shuffle=False,collate_fn=collate)
    model=ScoreBert(a.model).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=.01)
    start_epoch=0;step=0
    if a.resume:
        state=torch.load(a.resume,map_location='cpu',weights_only=False)
        if not state.get('epoch_complete'):raise ValueError('Exact continuation supported from completed-epoch checkpoint.pt only; step snapshots are recovery weights.')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);start_epoch=state['epoch'];step=state['step']
        for st in optimizer.state.values():
            for k,v in st.items():
                if torch.is_tensor(v) and k!='step':st[k]=v.to(device)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()};config['dataset_provenance']=bundle.metadata;config['device']=str(device);config['architecture']='BERT pooled output → dropout → linear(1) → sigmoid; prediction ×100';config['loss']='MSE on score/100'
    config['truncated_texts']=sum(len(tokenizer(r['text'],truncation=False)['input_ids'])>a.max_length for r in train+valid)
    save_dataset_provenance(a.output,bundle.metadata)
    (a.output/'config.json').write_text(json.dumps(config,indent=2));tokenizer.save_pretrained(a.output/'tokenizer')
    run=None
    if not a.no_wandb:
        import wandb
        run=wandb.init(project='detector-a-score-bert',name=a.output.name,dir=str(a.output.resolve()),config=config)
        (a.output/'wandb-run.json').write_text(json.dumps({'url':run.url,'id':run.id},indent=2));print('W&B:',run.url,flush=True)
    history=[];started=time.time()
    baseline=float(np.mean([r['score_pct'] for r in train]));gold=np.array([r['score_pct'] for r in valid])
    def record(row):
        history.append(row)
        with (a.output/'loss-history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        if run:run.log(row,step=step)
        print(json.dumps(row),flush=True)
    def evaluate():
        model.eval();pred=[]
        with torch.no_grad():
            for inputs,y in va:pred.extend((model(**inputs)*100).cpu().tolist())
        pred=np.array(pred);err=pred-gold
        metrics={'validation_mse':float(np.mean((err/100)**2)),'validation_mae_pp':float(np.mean(abs(err))),'validation_rmse_pp':float(np.sqrt(np.mean(err**2)))}
        return metrics,pred
    try:
        initial,_=evaluate();record({'step':step,'epoch':start_epoch,**initial})
        for epoch in range(start_epoch,start_epoch+a.epochs):
            model.train();total=0.;seen=0
            for inputs,y in tr:
                optimizer.zero_grad(set_to_none=True);pred=model(**inputs);loss=nn.functional.mse_loss(pred,y)
                loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();step+=1
                total+=loss.item()*len(y);seen+=len(y)
                if step%10==0:record({'step':step,'epoch':epoch+seen/len(train),'train_mse':total/seen})
                if step%100==0 and not a.latest_checkpoint_only:
                    torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,'step':step,'config':config,'epoch_complete':False},a.output/f'step-{step}.pt')
            metrics,pred=evaluate();record({'step':step,'epoch':epoch+1,'train_mse':total/seen,**metrics})
            torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch+1,'step':step,'config':config,'epoch_complete':True},a.output/'checkpoint.pt')
        with (a.output/'validation-predictions.jsonl').open('w') as f:
            for row,score in zip(valid,pred):f.write(json.dumps({'text_sha256':row['text_sha256'],'target_score_pct':row['score_pct'],'predicted_score_pct':float(score),'error_pp':float(score-row['score_pct'])})+'\n')
        summary={'dataset_version':bundle.metadata['dataset_version'],'dataset_fingerprint':bundle.metadata['dataset_fingerprint'],'test_examples_reserved':len(bundle.splits['test']),'completed_epochs':a.epochs,'steps':step,'train_examples':len(train),'validation_examples':len(valid),'elapsed_seconds':time.time()-started,**metrics,'constant_baseline_mae_pp':float(np.mean(abs(gold-baseline))),'constant_baseline_rmse_pp':float(np.sqrt(np.mean((gold-baseline)**2))),'baseline_train_mean_pct':baseline}
        (a.output/'training-summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
        if run:run.summary.update(summary)
    finally:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots()
        for metric in ['train_mse','validation_mse']:
            pts=[(x['step'],x[metric]) for x in history if metric in x]
            if pts:ax.plot(*zip(*pts),label=metric)
        ax.set(xlabel='Optimizer step',ylabel='MSE (score normalized to 0–1)');ax.legend();fig.tight_layout();fig.savefig(a.output/'loss-curves.png');plt.close(fig)
        if run:run.finish()

if __name__=='__main__':main()

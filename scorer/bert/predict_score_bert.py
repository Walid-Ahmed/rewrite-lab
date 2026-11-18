"""Predict a Detector A-like score using a trained local BERT checkpoint."""
import argparse,json,torch
from transformers import AutoTokenizer
from train_score_bert import ScoreBert

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('text');p.add_argument('--checkpoint',required=True);a=p.parse_args()
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    c=state['config'];m=ScoreBert(c['model']);m.load_state_dict(state['model']);m.eval()
    t=AutoTokenizer.from_pretrained(c['model']);inputs=t(a.text,return_tensors='pt',truncation=True,max_length=c['max_length'])
    with torch.no_grad(): score=float(m(**inputs).item()*100)
    print(json.dumps({'predicted_detector_a_score_pct':score}))

if __name__=='__main__':main()

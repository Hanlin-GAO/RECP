"""Run reviewed-model inference on its fixed test interval with frozen training statistics."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'code/forecasting'))
import numpy as np
import pandas as pd
import torch
import reviewed_protocol as p

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    output=args.output.resolve()
    if output.exists() or output.with_suffix('.metadata.json').exists(): parser.error('Choose unused output paths')
    if any(output.is_relative_to(ROOT/folder) for folder in ['data','models','code','scripts','provenance','results_reference']):
        parser.error('Output cannot overwrite distributed files')
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    if checkpoint['protocol']['version']!='reviewed_v2': parser.error('Expected a reviewed_v2 checkpoint, not archived weights')
    settings=dict(checkpoint['protocol']); settings['device']='cpu'
    config=p.Protocol(**settings)
    stats=checkpoint['statistics']
    domain,station,tag=checkpoint['domain'],checkpoint['source'],checkpoint['model_tag']
    frame,features,sensors,audit=p.load_raw(domain,station)
    if features!=stats['feature_names']: raise ValueError('Checkpoint feature schema differs from raw preprocessing')
    provenance=checkpoint.get('provenance')
    if provenance is None: raise ValueError('Checkpoint lacks input/code provenance; retrain using the current reviewed entry point')
    if audit['inputs']!=provenance['inputs']: raise ValueError('Raw inputs differ from the training snapshot')
    for name,expected in provenance['code_sha256'].items():
        if p.digest(ROOT/'code/forecasting'/name)!=expected: raise ValueError(f'Forecasting code changed since training: {name}')
    test=frame.loc[frame.Time>=pd.Timestamp(stats['cutoffs'][1])].reset_index(drop=True)
    count=len(test)-config.window-config.horizon+1
    if count<=0: raise ValueError('Test interval has no complete input windows')
    starts=np.arange(count)
    indices=starts+config.window+config.horizon-1
    raw=test[features].to_numpy(np.float32)
    windows=np.lib.stride_tricks.sliding_window_view(raw,config.window,axis=0).transpose(0,2,1)[starts].copy()
    model=p.build_model(tag,domain,config,stats)
    model.load_state_dict(checkpoint['model_state_dict'])
    torch.set_num_threads(4)
    pred=p.predict(model,tag,domain,{'X':windows},stats,config,0.0)
    if not np.isfinite(pred).all(): raise ValueError('Nonfinite reviewed prediction')
    output.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({'Time':test.Time.iloc[indices].to_numpy(),'input_start':test.Time.iloc[starts].to_numpy(),
                  'input_end':test.Time.iloc[starts+config.window-1].to_numpy(),'Pred_Raw':pred,
                  'Pred_Clipped':np.clip(pred,0,stats['training_power_bound'])}).to_csv(output,index=False)
    p.write_json(output.with_suffix('.metadata.json'),{'protocol':'reviewed_v2','checkpoint_sha256':p.digest(args.checkpoint),
                 'statistics':'frozen checkpoint statistics; no refitting','domain':domain,'source':station,
                 'interval_start':stats['cutoffs'][1],'rows':len(pred),'current_inputs':audit['inputs'],
                 'weather_input':'last available input row; no target-time weather','development':config.development,
                 'code_and_input_hashes_verified':True})
    print(f'Saved {len(pred)} predictions to {output}')

if __name__=='__main__': main()

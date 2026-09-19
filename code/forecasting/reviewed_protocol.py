"""Auditable forecasting protocol, separate from the archived paper experiments.

All preprocessing estimates use training rows. Validation alone selects checkpoints.
Test data are evaluated only after fitting, and every requested source/model/seed is reported.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import argparse
import copy
import hashlib
import importlib.metadata
import json
import random
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from model import DNNModel, BaselineModel, GRUModel, TCNModel, TransformerModel, PINNModel
from wind_model import WindPINNModel

ROOT = Path(__file__).resolve().parents[2]
TAGS = ('dnn','lstm','gru','tcn','trans','pinn')

@dataclass(frozen=True)
class Protocol:
    version: str = 'reviewed_v2'
    window: int = 24
    horizon: int = 1
    epochs: int = 150
    patience: int = 30
    hidden: int = 64
    layers: int = 2
    dropout: float = 0.2
    learning_rate: float = 0.0005
    batch_size: int = 256
    seeds: tuple = (42,43,44)
    models: tuple = TAGS
    missing_fractions: tuple = (0.0,0.1,0.3,0.5)
    masking_seed: int = 20260913
    device: str = 'cpu'
    development: bool = False

def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()

def write_json(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')

def read_csv(path):
    for encoding in ('utf-8-sig','gb18030','latin1'):
        try: return pd.read_csv(path,encoding=encoding)
        except UnicodeDecodeError: continue
    raise ValueError(f'Cannot decode {path}')

def clean_time(frame,time_col,year,audit):
    frame = frame.copy()
    frame['Time'] = pd.to_datetime(frame[time_col],format='mixed',errors='coerce')
    audit['input_rows'] = len(frame)
    audit['invalid_time_rows'] = int(frame.Time.isna().sum())
    audit['outside_fixed_year_rows'] = int((frame.Time.notna() & (frame.Time.dt.year!=year)).sum())
    return frame.loc[frame.Time.dt.year==year].copy()

def numeric(frame,columns):
    for col in columns: frame[col] = pd.to_numeric(frame[col],errors='coerce').replace([np.inf,-np.inf],np.nan)
    return frame

def causal_weather_bins(weather,column,cadence):
    """A bin labelled t contains only observations at or before t."""
    return weather[['Time',column]].set_index('Time').resample(cadence,closed='right',label='right').mean().reset_index()

def load_raw(domain,station):
    """Use fixed years and right-closed weather bins; never match future observations."""
    year = 2023 if domain=='pv' else 2020
    cadence = '5min' if domain=='pv' else '15min'
    audit = {'source_id':station,'domain':domain,'fixed_year':year,'cadence':cadence,'inputs':[]}
    def record(path):
        audit['inputs'].append({'path':path.relative_to(ROOT).as_posix(),'sha256':digest(path)})
    if domain=='pv':
        folder = ROOT/'data/forecasting/raw/pv'
        path = folder/f'Library_{station}.csv'
        record(path)
        frame = read_csv(path)
        frame.columns = frame.columns.str.strip()
        frame = clean_time(frame,frame.columns[0],year,audit)
        power = next((c for c in ('totalActivePower(W)','totalActivePower') if c in frame),None)
        if power is None: raise ValueError(f'Missing active-power column in {path}')
        frame = numeric(frame.rename(columns={power:'Power'}),['Power'])
        audit['duplicate_power_rows'] = int(frame.Time.duplicated().sum())
        frame = frame.groupby('Time',as_index=False).Power.mean()
        for file,col in [('Irradiance_2023.csv','Irradiance'),('Temperature_2023.csv','Temperature'),('Wind_2023.csv','WindSpeed')]:
            path = folder/file
            record(path)
            weather = read_csv(path)
            weather.columns = weather.columns.str.strip()
            time_col,value_col = weather.columns[:2]
            detail = {}
            weather = clean_time(weather,time_col,year,detail)
            weather = numeric(weather.rename(columns={value_col:col}),[col])
            detail['duplicate_time_rows'] = int(weather.Time.duplicated().sum())
            weather = causal_weather_bins(weather,col,cadence)
            frame = pd.merge_asof(frame.sort_values('Time'),weather.sort_values('Time'),on='Time',direction='backward',tolerance=pd.Timedelta(cadence))
            audit[file] = detail
        sensors = ['Irradiance','Temperature','WindSpeed']
    else:
        number = int(station.split('_')[-1])
        path = ROOT/f'data/forecasting/raw/wind/Wind {number}.xlsx'
        record(path)
        frame = pd.read_excel(path)
        frame.columns = [' '.join(str(c).split()) for c in frame.columns]
        def find(fragment):
            matches = [c for c in frame if fragment in c]
            if not matches: raise ValueError(f'Missing column matching {fragment}')
            return matches[0]
        rename = {find('Wind speed - at the height of wheel hub (m/s)'):'WindSpeed',
                  find('Air temperature'):'Temperature',find('Power (MW)'):'Power'}
        direction = next((c for c in frame if 'at the height of wheel hub' in c and ('direction' in c or '(˚)' in c)),None)
        if direction is None: raise ValueError('Missing hub-height wind direction')
        rename[direction] = 'WindDirection'
        frame = clean_time(frame,find('Time(year-month-day'),year,audit).rename(columns=rename)
        frame = numeric(frame,['Power','WindSpeed','Temperature','WindDirection'])
        audit['duplicate_time_rows'] = int(frame.Time.duplicated().sum())
        # Aggregate duplicate directions on the circle, without wrapping through 180 degrees.
        radians = np.deg2rad(frame.WindDirection)
        frame['WindDir_sin'],frame['WindDir_cos'] = np.sin(radians),np.cos(radians)
        sensors = ['WindSpeed','Temperature','WindDir_sin','WindDir_cos']
        frame = frame[['Time','Power',*sensors]].groupby('Time',as_index=False).mean()
    # Reindex instead of compressing missing periods or concatenating daytime records.
    grid = pd.date_range(f'{year}-01-01',f'{year+1}-01-01',freq=cadence,inclusive='left')
    audit['off_grid_rows'] = int((~frame.Time.isin(grid)).sum())
    frame = frame.set_index('Time').reindex(grid).rename_axis('Time').reset_index()
    audit['negative_target_rows'] = int((frame.Power<0).sum())
    frame.loc[frame.Power<0,'Power'] = np.nan
    for col in sensors:
        invalid = frame[col]<0 if col in {'WindSpeed','Irradiance'} else frame[col]<=-273.15 if col=='Temperature' else pd.Series(False,index=frame.index)
        audit[f'invalid_{col}_rows'] = int(invalid.sum())
        frame.loc[invalid,col] = np.nan
    hours = frame.Time.dt.hour+frame.Time.dt.minute/60
    day = frame.Time.dt.dayofyear
    for name,values in [('hour',hours/24),('year',day/(366 if year%4==0 else 365))]:
        frame[f'{name}_sin'] = np.sin(2*np.pi*values)
        frame[f'{name}_cos'] = np.cos(2*np.pi*values)
    features = sensors+['hour_sin','hour_cos','year_sin','year_cos']
    audit['grid_rows'] = len(frame)
    audit['missing_target_rows'] = int(frame.Power.isna().sum())
    audit['missing_input_rows_by_feature'] = {c:int(frame[c].isna().sum()) for c in features}
    audit['daytime_filter'] = False
    audit['outlier_filter'] = 'No performance-based filtering; only invalid time/nonfinite and negative generation are excluded.'
    return frame,features,sensors,audit

def prepare(frame,features,sensors,config):
    """Fit on January-August only, then build disjoint interval-local windows."""
    year = int(frame.Time.dt.year.iloc[0])
    dates = (pd.Timestamp(f'{year}-09-01'),pd.Timestamp(f'{year}-11-01'))
    partition = np.where(frame.Time<dates[0],'train',np.where(frame.Time<dates[1],'val','test'))
    train = frame.loc[partition=='train',features].to_numpy(float)
    median = np.nanmedian(train,axis=0)
    if not np.isfinite(median).all(): raise ValueError('A training feature is entirely missing')
    filled = np.where(np.isfinite(train),train,median)
    mean,std = filled.mean(0),filled.std(0)
    std = np.where(std<1e-8,1.0,std)
    targets = frame.loc[partition=='train','Power'].dropna().to_numpy(float)
    if not len(targets): raise ValueError('Training targets are empty')
    y_mean,y_std = float(targets.mean()),max(float(targets.std()),1e-8)
    bound = float(targets.max()*1.05)
    if bound<=0: raise ValueError('Training power must contain positive values')
    quantiles = {col:np.nanquantile(frame.loc[partition=='train',col],[0.1,0.9]).tolist() for col in sensors[:2]}
    samples,counts = {},{}
    for name in ('train','val','test'):
        rows = np.flatnonzero(partition==name)
        local = frame.iloc[rows]
        raw = local[features].to_numpy(np.float32)
        if len(raw)<config.window+config.horizon: raise ValueError(f'{name} interval is too short')
        starts = np.arange(len(raw)-config.window-config.horizon+1)
        indices = starts+config.window+config.horizon-1
        valid = np.isfinite(local.Power.to_numpy()[indices])
        starts,indices = starts[valid],indices[valid]
        X = np.lib.stride_tricks.sliding_window_view(raw,config.window,axis=0).transpose(0,2,1)[starts].copy()
        samples[name] = {'X':X,'y':local.Power.to_numpy(np.float32)[indices],
                         'times':local.Time.iloc[indices].to_numpy(),
                         'input_start':local.Time.iloc[starts].to_numpy(),
                         'input_end':local.Time.iloc[starts+config.window-1].to_numpy(),
                         'target_weather':local[sensors[:2]].to_numpy(np.float32)[indices]}
        counts[name] = {'interval_rows':len(local),'usable_targets':len(indices),
                        'warmup_rows_excluded':config.window+config.horizon-1,
                        'invalid_target_windows':int((~valid).sum())}
        if not len(indices): raise ValueError(f'No usable {name} targets')
    stats = {'feature_names':features,'sensor_names':sensors,'median':median.tolist(),
             'mean':mean.tolist(),'std':std.tolist(),'y_mean':y_mean,'y_std':y_std,
             'training_power_bound':bound,'training_weather_quantiles':quantiles,
             'cutoffs':[str(t) for t in dates],'counts':counts,
             'physics_weather':'last input row after the same masking/imputation as the temporal branch'}
    return samples,stats

def encode(raw,stats,fraction=0.0,masking_seed=20260913):
    """Mask the same sensor cells for every model, including physics inputs."""
    values = raw.copy()
    if not 0<=fraction<=1: raise ValueError('Missing fraction must be in [0,1]')
    sensor_count = len(stats['sensor_names'])
    groups = [[0],[1],[2]] if sensor_count==3 else [[0],[1],[2,3]]
    uniform = np.random.default_rng(masking_seed).random((*values.shape[:2],len(groups)))
    for idx,cols in enumerate(groups):
        for col in cols: values[:,:,col] = np.where(uniform[:,:,idx]<fraction,np.nan,values[:,:,col])
    missing = ~np.isfinite(values)
    physical = np.where(missing, np.array(stats['median'],np.float32),values)
    normalized = (physical-np.array(stats['mean'],np.float32))/np.array(stats['std'],np.float32)
    x = np.concatenate([normalized,missing.astype(np.float32)],axis=2).astype(np.float32)
    return x,physical[:,-1,:2].astype(np.float32)

def build_model(tag,domain,config,stats):
    dims = 2*len(stats['feature_names'])
    if tag=='dnn': return DNNModel(dims,config.hidden,config.window,config.dropout)
    cls = {'lstm':BaselineModel,'gru':GRUModel,'tcn':TCNModel,'trans':TransformerModel}.get(tag)
    if cls: return cls(dims,config.hidden,config.layers,config.dropout)
    if tag!='pinn': raise ValueError(tag)
    model = (PINNModel(dims,config.hidden,config.layers,config.dropout) if domain=='pv' else
             WindPINNModel(dims,config.hidden,config.layers,config.dropout,rated_power_init=stats['training_power_bound']))
    model.set_output_scale(stats['y_mean'],stats['y_std'])
    return model

def forward(model,tag,domain,x,weather):
    if tag!='pinn': return model(x)
    return model(x,**({'G':weather[:,0],'T_amb':weather[:,1]} if domain=='pv' else {'v':weather[:,0],'T_amb':weather[:,1]}))

def make_loader(sample,stats,config,shuffle=False):
    x,weather = encode(sample['X'],stats)
    y = (sample['y']-stats['y_mean'])/stats['y_std']
    data = TensorDataset(torch.from_numpy(x),torch.from_numpy(weather),torch.from_numpy(y.astype(np.float32)))
    return DataLoader(data,batch_size=config.batch_size,shuffle=shuffle,num_workers=0)

def fit_model(model,tag,domain,train,validation,stats,config):
    """Select by validation MSE; the function has no test-data argument."""
    model.to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=config.learning_rate,weight_decay=1e-4)
    train_loader = make_loader(train,stats,config,True)
    val_loader = make_loader(validation,stats,config)
    best,best_state,wait,history = float('inf'),None,0,[]
    for epoch in range(1,config.epochs+1):
        model.train()
        train_sum,n = 0.0,0
        for x,w,y in train_loader:
            x,w,y = x.to(config.device),w.to(config.device),y.to(config.device)
            optimizer.zero_grad()
            pred = forward(model,tag,domain,x,w)
            loss = (pred-y).square().mean()
            if tag=='pinn':
                physical = pred*stats['y_std']+stats['y_mean']
                loss = loss+0.1*((torch.relu(-physical).square()+torch.relu(physical-stats['training_power_bound']).square()).mean()/stats['training_power_bound']**2)
            if not torch.isfinite(loss): raise ValueError('Nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step()
            train_sum+=loss.item()*len(y); n+=len(y)
        model.eval()
        val_sum,n_val = 0.0,0
        with torch.no_grad():
            for x,w,y in val_loader:
                pred=forward(model,tag,domain,x.to(config.device),w.to(config.device))
                val_sum+=(pred-y.to(config.device)).square().sum().item(); n_val+=len(y)
        val_loss=val_sum/n_val
        history.append({'epoch':epoch,'training_loss':train_sum/n,'validation_mse_scaled':val_loss})
        if val_loss<best:
            best,best_state,wait=val_loss,copy.deepcopy(model.state_dict()),0
        else: wait+=1
        if wait>=config.patience: break
    if best_state is None: raise ValueError('No finite validation checkpoint')
    model.load_state_dict(best_state)
    return history

def predict(model,tag,domain,sample,stats,config,fraction):
    x,w=encode(sample['X'],stats,fraction,config.masking_seed)
    values=[]
    model.eval()
    with torch.no_grad():
        for start in range(0,len(x),config.batch_size):
            pred=forward(model,tag,domain,torch.from_numpy(x[start:start+config.batch_size]).to(config.device),torch.from_numpy(w[start:start+config.batch_size]).to(config.device))
            values.append(pred.cpu().numpy()*stats['y_std']+stats['y_mean'])
    return np.concatenate(values)

def metrics(y,pred):
    if len(y)==0: return {'n':0,'mae':None,'rmse':None,'r2':None}
    error=pred-y
    denominator=float(((y-y.mean())**2).sum())
    return {'n':len(y),'mae':float(abs(error).mean()),'rmse':float(np.sqrt((error**2).mean())),
            'r2':None if denominator<=0 else float(1-float((error**2).sum())/denominator)}

def evaluate(model,tag,domain,test,stats,config,out):
    """Evaluate all test rows first; fixed train-threshold subsets are supplementary."""
    weather=test['target_weather']
    driver,temp=stats['sensor_names'][:2]
    dq,tq=stats['training_weather_quantiles'][driver],stats['training_weather_quantiles'][temp]
    extreme=(weather[:,0]>dq[1])|(weather[:,1]<tq[0])|(weather[:,1]>tq[1])
    if domain=='wind': extreme |= weather[:,0]<dq[0]
    records=[]
    for fraction in config.missing_fractions:
        raw=predict(model,tag,domain,test,stats,config,fraction)
        if not np.isfinite(raw).all(): raise ValueError('Nonfinite test prediction')
        clipped=np.clip(raw,0,stats['training_power_bound'])
        for scale,pred in [('raw',raw),('clipped',clipped)]:
            for subset,mask in [('all',np.ones(len(raw),bool)),('extreme_train_threshold',extreme)]:
                records.append({'missing_fraction':fraction,'output':scale,'subset':subset,**metrics(test['y'][mask],pred[mask])})
        pd.DataFrame({'Time':test['times'],'input_start':test['input_start'],'input_end':test['input_end'],
                      'True_Power':test['y'],'Pred_Raw':raw,'Pred_Clipped':clipped,
                      'extreme_train_threshold':extreme}).to_csv(out/f'test_missing_{fraction:g}.csv',index=False)
    return records

def run(domain,stations,config,out,audit_only=False):
    out=Path(out).resolve()
    if any(out==ROOT/p or out.is_relative_to(ROOT/p) for p in ['data','models','code','scripts','provenance','results_reference']):
        raise ValueError('Output must not overwrite distributed inputs or code')
    if out==ROOT or ROOT.is_relative_to(out): raise ValueError('Output cannot be the package root or its ancestor')
    if out.exists() and any(out.iterdir()): raise FileExistsError('Choose a new or empty output directory')
    out.mkdir(parents=True,exist_ok=True)
    protocol={'config':asdict(config),'domain':domain,'sources':stations,'training_rule':'validation-only selection; report every source/model/seed',
              'source_scope':'all raw sources' if len(stations)==(3 if domain=='pv' else 6) else 'explicit subset; not an all-source benchmark',
              'code_sha256':{p.name:digest(p) for p in [Path(__file__),ROOT/'code/forecasting/model.py',ROOT/'code/forecasting/wind_model.py']},
              'environment':{'python':sys.version,'packages':{name:importlib.metadata.version(name) for name in ['numpy','pandas','torch','openpyxl']},
                             'torch_threads':torch.get_num_threads(),'device':config.device}}
    write_json(out/'protocol.json',protocol)
    results=[]
    for station in stations:
        print(f'Preparing {domain}/{station}',flush=True)
        folder=out/station; folder.mkdir()
        frame,features,sensors,audit=load_raw(domain,station)
        samples,stats=prepare(frame,features,sensors,config)
        write_json(folder/'preprocessing_audit.json',audit)
        write_json(folder/'training_statistics.json',stats)
        for split,sample in samples.items():
            pd.DataFrame({k:sample[k] for k in ['times','input_start','input_end']}).to_csv(folder/f'{split}_membership.csv',index=False)
        if audit_only: continue
        for seed in config.seeds:
            for tag in config.models:
                run_dir=folder/f'seed_{seed}'/tag; run_dir.mkdir(parents=True)
                try:
                    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
                    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
                    torch.backends.cudnn.benchmark=False
                    torch.use_deterministic_algorithms(True)
                    model=build_model(tag,domain,config,stats)
                    history=fit_model(model,tag,domain,samples['train'],samples['val'],stats,config)
                    pd.DataFrame(history).to_csv(run_dir/'history.csv',index=False)
                    torch.save({'protocol':asdict(config),'domain':domain,'source':station,'model_tag':tag,'seed':seed,
                                'statistics':stats,'provenance':{'inputs':audit['inputs'],'code_sha256':protocol['code_sha256']},
                                'model_state_dict':{k:v.cpu() for k,v in model.state_dict().items()}},run_dir/'checkpoint.pt')
                    entries=evaluate(model,tag,domain,samples['test'],stats,config,run_dir)
                    write_json(run_dir/'metrics.json',entries)
                    for entry in entries: results.append({'source':station,'seed':seed,'model':tag,'status':'ok',**entry})
                except Exception as exc:
                    results.append({'source':station,'seed':seed,'model':tag,'status':'failed','reason':str(exc)})
                    write_json(run_dir/'failure.json',{'error_type':type(exc).__name__,'message':str(exc)})
                write_json(out/'all_runs.json',results)
                print(f'Finished {station}, seed {seed}, {tag}: {results[-1]["status"]}',flush=True)
    if results:
        table=pd.DataFrame(results); table.to_csv(out/'all_runs.csv',index=False)
        failures=table.loc[table.status!='ok']
        if len(failures): raise RuntimeError('One or more runs failed; see all_runs.json. Do not report an aggregate over only successful runs.')
        summary=table.groupby(['source','model','missing_fraction','output','subset'],dropna=False).agg(
            seed_count=('seed','nunique'),mae_mean=('mae','mean'),mae_std=('mae','std'),rmse_mean=('rmse','mean'),rmse_std=('rmse','std'),r2_mean=('r2','mean'),r2_std=('r2','std')).reset_index()
        summary.to_csv(out/'summary_all_seeds.csv',index=False)
    write_json(out/'completion.json',{'status':'audit_complete' if audit_only else 'complete','source_count':len(stations),'development':config.development})

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--domain',choices=['pv','wind'],required=True)
    parser.add_argument('--station',help='Raw source ID. Omit to include every raw source.')
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=150)
    parser.add_argument('--seeds',type=int,nargs='+',default=[42,43,44])
    parser.add_argument('--models',nargs='+',choices=TAGS,default=list(TAGS))
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--audit-only',action='store_true')
    args=parser.parse_args()
    if args.epochs<1 or len(set(args.seeds))!=len(args.seeds) or len(set(args.models))!=len(args.models): parser.error('Epochs must be positive; models and seeds must be unique.')
    all_sources=[f'{"Inverter" if args.domain=="pv" else "Turbine"}_{i}' for i in range(1,4 if args.domain=='pv' else 7)]
    if args.station and args.station not in all_sources: parser.error('Unknown raw source ID')
    config=Protocol(epochs=args.epochs,seeds=tuple(args.seeds),models=tuple(args.models),device=args.device,
                    development=args.epochs!=150 or tuple(args.seeds)!=(42,43,44) or tuple(args.models)!=TAGS)
    torch.set_num_threads(4)
    run(args.domain,[args.station] if args.station else all_sources,config,args.output_dir,args.audit_only)

if __name__=='__main__': main()

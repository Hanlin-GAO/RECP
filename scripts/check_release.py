"""Check input schemas, every archived model, and a short real-scene simulation."""
import ast
import csv
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'runs/.matplotlib'))
sys.dont_write_bytecode = True
sys.path[:0] = [str(ROOT/'code/scheduling'), str(ROOT/'code/forecasting')]

def main():
    import numpy as np
    import pandas as pd
    import torch
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    import io_utils
    import run_manual_validation_matrix as schedule
    from scenarios import build_manual_mixed_scenario
    report = {'environment': {n: importlib.metadata.version(n) for n in
              ['numpy','pandas','torch','matplotlib','scipy','openpyxl']}, 'checks': {}, 'models': []}
    report['environment']['python'] = sys.version.split()[0]
    paths = list((ROOT/'code').rglob('*.py')) + list((ROOT/'scripts').glob('*.py'))
    for path in paths: ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
    report['checks']['python_syntax_files'] = len(paths)
    for path in (ROOT/'data').rglob('*.json'):
        json.loads(path.read_text(encoding='utf-8-sig'))
    for path in (ROOT/'data/experiments/vehicle_power').glob('*.csv'):
        frame = pd.read_csv(path)
        needed = ['time_s','normal'] + [f'{p}{i}' for p in ['mor','noon','eve'] for i in range(1,13)]
        assert set(needed).issubset(frame), (path, needed)
        assert np.isfinite(frame[needed].to_numpy()).all(), path
        assert np.allclose(np.diff(frame.time_s), 5), path
    report['checks']['vehicle_power_schemas'] = 'passed'
    for domain,prefix,merged in [('pv','Inverter','merged_2023_data.csv'),('wind','Turbine','merged_wind_data.csv')]:
        module = importlib.import_module('main' if domain == 'pv' else 'wind_main')
        features,window,build,postprocess,classes = io_utils._load_model_runtime(domain)
        for station_no in range(1,4):
            station = f'{prefix}_{station_no}'
            frame = pd.read_csv(ROOT/'data/forecasting/processed'/domain/station/merged)
            frame['Time'] = pd.to_datetime(frame['Time'])
            metrics = json.loads((ROOT/'results_reference/forecasting'/domain/station/'test_metrics.json').read_text())
            power_col = 'totalActivePower' if domain == 'pv' else 'Power'
            rated = float(metrics.get('config',{}).get('rated_power',frame[power_col].max()*1.05))
            cached_sequences = {}
            for tag in ['dnn','lstm','gru','tcn','trans','pinn']:
                ckpt = torch.load(ROOT/'models'/domain/station/f'best_model_{tag}.pth', map_location='cpu', weights_only=False)
                state = ckpt['model_state_dict']
                w = io_utils._infer_window_size(tag,state,len(features),window)
                if w not in cached_sequences: cached_sequences[w] = build(frame,features,window_size=w)
                X,y,driver,temp,times = cached_sequences[w]
                saved = pd.read_csv(ROOT/'results_reference/forecasting'/domain/station/f'test_predictions_{tag}.csv').head(32)
                indices = pd.Index(pd.to_datetime(times)).get_indexer(pd.to_datetime(saved['Time']))
                assert (indices >= 0).all(), (domain,station,tag,'missing saved timestamps')
                x_std = np.asarray(ckpt['x_std'],dtype=np.float32)
                x_std = np.where(abs(x_std)<1e-8,1.0,x_std)
                x = torch.tensor((X[indices]-np.asarray(ckpt['x_mean'],dtype=np.float32))/x_std)
                d,t = torch.tensor(driver[indices]),torch.tensor(temp[indices])
                model,is_pinn = io_utils._build_model_from_state(domain,tag,state,len(features),w,rated,classes)
                if is_pinn: model.set_output_scale(float(ckpt['y_mean']),float(ckpt['y_std']))
                model.load_state_dict(state)
                model.eval()
                with torch.no_grad():
                    output = model(x,**({'G':d,'T_amb':t} if domain=='pv' else {'v':d,'T_amb':t})) if is_pinn else model(x)
                pred = postprocess(output.numpy()*float(ckpt['y_std'])+float(ckpt['y_mean']),driver[indices],rated)
                assert len(pred)==len(saved) and np.isfinite(pred).all(), (domain,station,tag)
                delta = float(np.max(np.abs(pred-saved.Pred_Power.to_numpy())))
                # Also exercise a genuine backward pass without changing the supplied checkpoint.
                model.train()
                value = model(x[:2],**({'G':d[:2],'T_amb':t[:2]} if domain=='pv' else {'v':d[:2],'T_amb':t[:2]})) if is_pinn else model(x[:2])
                loss = value.square().mean()
                loss.backward()
                assert torch.isfinite(loss), (domain,station,tag,'nonfinite loss')
                report['models'].append({'domain':domain,'station':station,'model':tag,'samples':len(saved),
                                         'max_abs_difference_from_saved_predictions':delta,'forward_backward':'passed'})
                print(f'Checked {domain}/{station}/{tag}; saved prediction max difference = {delta:.8g}',flush=True)
    stations,rovers,drones,scene = build_manual_mixed_scenario()
    dt,steps,tail = schedule._runtime_spec(drones)
    stations,rovers,drones = schedule._prepare_soc_case(stations,rovers,drones,'H')
    rovers,drones,meta = schedule._apply_vehicle_output_mode(rovers,drones,'AM','H','actual')
    for station in stations: station['renewable_profile_kW'] = np.full(300,0.1)
    result = schedule._simulate_once(stations,rovers,drones,scene,dt,300)
    assert result['soc'].shape==(300,6)
    for key in ['soc','station_soc','x','y','P_load']: assert np.isfinite(result[key]).all(), key
    assert (result['soc']>=0).all() and (result['soc']<=1).all()
    report['checks']['short_schedule'] = {'steps':300,'vehicles':6,'finite_states':True,'supply':'constant 0.1 kW test input; not a paper result'}
    out = ROOT/'runs/release_check.json'
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(f'Passed checks. Report: {out}')

if __name__ == '__main__': main()

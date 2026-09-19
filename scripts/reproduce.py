"""Command-line entry points for the separated RECP research release."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'runs/.matplotlib'))
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    forecast = commands.add_parser('replay-forecast', help='Historical full-record inference; not a held-out forecast benchmark.')
    forecast.add_argument('--domain', choices=['pv','wind'], required=True)
    forecast.add_argument('--station', required=True, help='Archived display ID, e.g. Inverter_1 or Turbine_2.')
    forecast.add_argument('--model', choices=['dnn','lstm','gru','tcn','trans','pinn'], default='pinn')
    forecast.add_argument('--output', type=Path)
    train = commands.add_parser('train', help='Train with the reviewed chronological evaluation protocol.')
    train.add_argument('--domain', choices=['pv','wind'], required=True)
    train.add_argument('--station', help='Original raw source ID; wind Turbine_5 means Wind 5.xlsx.')
    train.add_argument('--output-dir', type=Path, required=True)
    train.add_argument('--epochs', type=int, help='Override both baseline and PINN epochs for a development run.')
    train.add_argument('--seeds',type=int,nargs='+',default=[42,43,44])
    train.add_argument('--models',nargs='+',choices=['dnn','lstm','gru','tcn','trans','pinn'],default=['dnn','lstm','gru','tcn','trans','pinn'])
    train.add_argument('--audit-only',action='store_true')
    train.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    plot = commands.add_parser('plot', help='Plot archived forecast tables in a fresh working directory.')
    plot.add_argument('--domain', choices=['pv','wind'], required=True)
    plot.add_argument('--output-dir', type=Path, required=True)
    schedule = commands.add_parser('schedule', help='Run fixed-load scheduling replay with archived predictors.')
    schedule.add_argument('--groups', nargs='+', default=['G1'])
    schedule.add_argument('--output-dir', type=Path, default=ROOT / 'runs/scheduling/manual_first_frame')
    schedule.add_argument('--comparison-protocol',choices=['fixed_load','historical_paired'],default='fixed_load')
    args = parser.parse_args()
    if args.command in {'train','plot','schedule'}:
        args.output_dir = args.output_dir.resolve()
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            parser.error('Choose an empty or new output directory to preserve earlier results.')
        if args.output_dir == ROOT or ROOT.is_relative_to(args.output_dir):
            parser.error('Output cannot be the package root or its ancestor.')
        for protected in ['data','models','results_reference','code','scripts','provenance']:
            if args.output_dir.is_relative_to(ROOT / protected):
                parser.error('Output must not be inside the distributed source, data, model, or reference folders.')
    if args.command == 'schedule':
        cmd = [sys.executable, str(ROOT/'code/scheduling/run_manual_validation_matrix.py'),
               '--groups', *args.groups, '--output-dir', str(args.output_dir),'--comparison-protocol',args.comparison_protocol]
        return subprocess.call(cmd, env=os.environ.copy())
    if args.command == 'train':
        cmd = [sys.executable,str(ROOT/'code/forecasting/reviewed_protocol.py'),'--domain',args.domain,
               '--output-dir',str(args.output_dir),'--device',args.device,'--seeds',*[str(seed) for seed in args.seeds],'--models',*args.models]
        if args.station: cmd += ['--station',args.station]
        if args.epochs is not None: cmd += ['--epochs',str(args.epochs)]
        if args.audit_only: cmd += ['--audit-only']
        return subprocess.call(cmd,env=os.environ.copy())
    sys.path.insert(0, str(ROOT/'code/forecasting'))
    if args.command == 'replay-forecast':
        sys.path.insert(0, str(ROOT/'code/scheduling'))
        from io_utils import load_vstry_power_frame
        import torch
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        output = (args.output or ROOT/f'runs/predictions/{args.domain}_{args.station}_{args.model}.csv').resolve()
        if output.exists() or output.with_suffix('.metadata.json').exists(): parser.error(f'Output already exists: {output}')
        if any(output.is_relative_to(ROOT/p) for p in ['data','models','results_reference','code','scripts','provenance']):
            parser.error('Choose an output under runs/ or outside this release.')
        frame = load_vstry_power_frame(str(ROOT), args.domain, args.station, 'predicted', args.model)
        output.parent.mkdir(parents=True, exist_ok=True)
        frame.rename(columns={'Power':'Pred_Power'}).to_csv(output, index=False)
        import json
        output.with_suffix('.metadata.json').write_text(json.dumps({'evaluation':'historical_full_record_replay',
            'held_out':False,'weights':'archived','weather':'target-time weather from archived feature table',
            'source':args.station,'domain':args.domain},indent=2),encoding='utf-8')
        print(f'Saved {len(frame)} predictions to {output}')
    elif args.command == 'plot':
        shutil.copytree(ROOT/'results_reference/forecasting'/args.domain, args.output_dir, dirs_exist_ok=True)
        os.environ[f'RECP_{args.domain.upper()}_OUTPUT_DIR'] = str(args.output_dir)
        name = 'plot.py' if args.domain == 'pv' else 'wind_plot.py'
        return subprocess.call([sys.executable, str(ROOT/'code/forecasting'/name)], env=os.environ.copy())
    return 0

if __name__ == '__main__':
    raise SystemExit(main())

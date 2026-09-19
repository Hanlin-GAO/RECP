"""Regression tests for split isolation, information parity, and release documentation."""
import ast
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import tokenize
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True
os.environ.setdefault('MPLBACKEND','Agg')
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'runs/.matplotlib'))
sys.path[:0]=[str(ROOT/'code/forecasting'),str(ROOT/'code/scheduling')]
import numpy as np
import pandas as pd
import torch
import reviewed_protocol as p
import run_manual_validation_matrix as schedule

class ProtocolChecks(unittest.TestCase):
    def setUp(self):
        t=pd.date_range('2023-01-01','2023-12-31',freq='D')
        x=np.arange(len(t),dtype=float)
        self.frame=pd.DataFrame({'Time':t,'Power':100+x,'Irradiance':100+x,'Temperature':20+np.sin(x),
                                 'WindSpeed':3+x/100,'hour_sin':np.zeros(len(t))})
        self.features=['Irradiance','Temperature','WindSpeed','hour_sin']
        self.sensors=self.features[:3]
        self.config=p.Protocol(window=4,epochs=1,hidden=8,layers=1,batch_size=64,seeds=(42,),models=('dnn','pinn'),development=True)
        self.samples,self.stats=p.prepare(self.frame,self.features,self.sensors,self.config)

    def test_future_data_cannot_change_training_statistics(self):
        altered=self.frame.copy()
        altered.loc[altered.Time>='2023-09-01',self.features+['Power']]=999999
        _,stats=p.prepare(altered,self.features,self.sensors,self.config)
        for field in ['median','mean','std','y_mean','y_std','training_power_bound','training_weather_quantiles']:
            self.assertEqual(self.stats[field],stats[field],field)

    def test_splits_share_no_raw_observation_interval(self):
        train,val,test=[self.samples[k] for k in ['train','val','test']]
        self.assertLess(train['times'].max(),val['input_start'].min())
        self.assertLess(val['times'].max(),test['input_start'].min())
        for sample in [train,val,test]: self.assertTrue((sample['input_end']<sample['times']).all())

    def test_masking_reaches_physics_and_temporal_inputs(self):
        x,weather=p.encode(self.samples['test']['X'],self.stats,1.0)
        means,std=np.array(self.stats['mean']),np.array(self.stats['std'])
        reconstructed=x[:,-1,:len(self.features)]*std+means
        np.testing.assert_allclose(weather,reconstructed[:,:2],rtol=1e-6)
        np.testing.assert_allclose(weather,np.broadcast_to(self.stats['median'][:2],weather.shape),rtol=1e-6)
        self.assertTrue((x[:,:,len(self.features):len(self.features)+3]==1).all())

    def test_masks_are_repeatable(self):
        first=p.encode(self.samples['test']['X'],self.stats,0.3,17)
        second=p.encode(self.samples['test']['X'],self.stats,0.3,17)
        for a,b in zip(first,second): np.testing.assert_array_equal(a,b)

    def test_weather_bins_do_not_use_future_observations(self):
        weather=pd.DataFrame({'Time':pd.date_range('2023-01-01',periods=15,freq='min'),'value':np.arange(15,dtype=float)})
        before=p.causal_weather_bins(weather,'value','5min')
        weather.loc[weather.Time>'2023-01-01 00:05:00','value']=99999
        after=p.causal_weather_bins(weather,'value','5min')
        pd.testing.assert_frame_equal(before.loc[before.Time<='2023-01-01 00:05:00'],after.loc[after.Time<='2023-01-01 00:05:00'])

    def test_all_architectures_accept_reviewed_features(self):
        torch.set_num_threads(2)
        x,w=p.encode(self.samples['train']['X'][:8],self.stats)
        for domain in ('pv','wind'):
            for tag in p.TAGS:
                with self.subTest(domain=domain,model=tag):
                    model=p.build_model(tag,domain,self.config,self.stats)
                    pred=p.forward(model,tag,domain,torch.from_numpy(x),torch.from_numpy(w))
                    self.assertEqual(tuple(pred.shape),(8,))
                    self.assertTrue(torch.isfinite(pred).all())
                    pred.square().mean().backward()

    def test_empty_subset_metrics_remain_empty(self):
        self.assertEqual(p.metrics(np.array([]),np.array([])),{'n':0,'mae':None,'rmse':None,'r2':None})

    def test_training_selects_without_test_argument(self):
        import inspect
        self.assertNotIn('test',inspect.signature(p.fit_model).parameters)
        torch.set_num_threads(2)
        model=p.build_model('pinn','pv',self.config,self.stats)
        history=p.fit_model(model,'pinn','pv',self.samples['train'],self.samples['val'],self.stats,self.config)
        self.assertEqual(len(history),1)
        with tempfile.TemporaryDirectory(dir=ROOT/'runs') as directory:
            results=p.evaluate(model,'pinn','pv',self.samples['test'],self.stats,self.config,Path(directory))
        self.assertEqual(len(results),len(self.config.missing_fractions)*4)

    def test_equal_load_comparison_uses_identical_traces(self):
        a,b,meta_a=schedule._apply_vehicle_output_mode([{}, {}, {}],[{}, {}, {}],'AM','L','actual','fixed_load')
        c,d,meta_b=schedule._apply_vehicle_output_mode([{}, {}, {}],[{}, {}, {}],'AM','L','predicted','fixed_load')
        for left,right in zip(a+b,c+d): np.testing.assert_array_equal(left['work_power_profile_kW'],right['work_power_profile_kW'])
        self.assertEqual(meta_a['explore_vehicle_columns'],meta_b['explore_vehicle_columns'])

    def test_all_comments_and_docstrings_are_english(self):
        for path in ROOT.rglob('*.py'):
            if 'runs' in path.relative_to(ROOT).parts: continue
            source=path.read_text(encoding='utf-8-sig')
            for token in tokenize.generate_tokens(io.StringIO(source).readline):
                if token.type==tokenize.COMMENT: self.assertIsNone(re.search(r'[\u3400-\u9fff]',token.string),f'{path}:{token.start[0]}')
            for node in ast.walk(ast.parse(source)):
                if isinstance(node,(ast.Module,ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)):
                    self.assertIsNone(re.search(r'[\u3400-\u9fff]',ast.get_docstring(node) or ''),str(path))

    def test_legacy_scripts_fail_before_running_experiments(self):
        for path in (ROOT/'code/legacy').rglob('*.py'):
            result=subprocess.run([sys.executable,str(path)],capture_output=True,text=True,encoding='utf-8')
            self.assertNotEqual(result.returncode,0)
            self.assertIn('Inspection-only historical code',result.stderr)

if __name__=='__main__':
    (ROOT/'runs').mkdir(exist_ok=True)
    unittest.main(verbosity=2)

"""Run portable CPU examples; every generated file stays in a chosen run folder.

The demo creates analytic synthetic coefficients from the included model. It
does not represent independently fitted data or a GPU validation campaign.
"""
from pathlib import Path
import argparse
import importlib.util
import json
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
SOURCE_FILES = [
    'single_message_ep/primary/zero_field_ep.py',
    'single_message_ep/primary/inputs/energy_model.py',
    'single_message_ep/independent/audit_zero_field_ep.py',
    'response_shape_validation/c1/run_c1.py',
    'response_shape_validation/c3/prepare_public_settings.py',
    'response_shape_validation/c3/predict.py',
    'response_shape_validation/c3/validate.py',
    'response_shape_validation/c4/run_c4.py',
]


def stage_sources(output):
    for rel in SOURCE_FILES:
        src, dest = ROOT / rel, output / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and dest.read_bytes() != src.read_bytes():
            raise RuntimeError(f'Source changed since this run: {rel}; use a new run folder.')
        if not dest.exists():
            shutil.copy2(src, dest)


def run(output, relative, *args):
    subprocess.run([sys.executable, '-B', str(output / relative), *args],
                   cwd=output, check=True)


def analytic_demo_inputs(output):
    """Reuse the existing fixed-pole model to generate an explicit identity test."""
    location = output / 'single_message_ep/primary/zero_field_ep.py'
    spec = importlib.util.spec_from_file_location('demo_zero_field', location)
    model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)
    unit = 2 * model.mp.pi * model.M('1e9')
    gyro = model.M('1.7595e11')
    cases, truth = [], {}
    for cid, value in [('A', '.084287616'), ('B', '.126'), ('C', '.99')]:
        r = model.M(value)
        mat = model.magnetic(r, model.GAM)
        aa = 2 * gyro * r * mat['a'] / unit
        bb = 2 * gyro * r * mat['b'] / unit
        cases.append({'case_id': cid, 'Gamma_bar': float(model.GAM),
                      'Omega_bar': float(model.OMG),
                      'A_tilde_per_T': float(aa), 'B_tilde_per_T': float(bb),
                      'rho_s': float(aa / bb / unit)})
        truth[cid] = {'r': float(r), 'alpha': float(mat['alpha']),
                      'BE_T': float(mat['BE']), 'BA_T': float(mat['BA']),
                      'gamma_rad_s_T': float(gyro), 'Gamma_bar': float(model.GAM),
                      'Omega_bar': float(model.OMG)}
    folder = output / 'response_shape_validation/c2'
    (folder / 'private').mkdir(parents=True, exist_ok=True)
    scope = ('Synthetic analytic coefficients from the declared fixed-pole family. '
             'Identity demonstration only: no measured data, time-domain fitting '
             'or independent GPU validation.')
    (folder / 'calibration_frozen.json').write_text(json.dumps(
        {'scope': scope, 'unit_rate_rad_s': float(unit), 'cases': cases}, indent=2)+'\n', encoding='utf-8')
    (folder / 'private/truth_by_case.json').write_text(json.dumps(
        {'scope': 'Post-prediction validation inputs for the synthetic identity demo.',
         'cases': truth}, indent=2)+'\n', encoding='utf-8')
    (output / 'demo_scope.json').write_text(json.dumps({'scope': scope}, indent=2)+'\n', encoding='utf-8')


def predict(output):
    c2 = output / 'response_shape_validation/c2'
    c3 = output / 'response_shape_validation/c3'
    if not (c2 / 'calibration_frozen.json').is_file():
        raise FileNotFoundError('First generate and freeze C2 data, or run the explicit synthetic demo.')
    if (c3 / 'predictions_frozen.json').exists():
        raise FileExistsError('Frozen predictions already exist; use a new run folder to change calibration.')
    shutil.copy2(c2 / 'calibration_frozen.json', c3 / 'calibration_public_frozen.json')
    run(output, 'response_shape_validation/c3/prepare_public_settings.py')
    run(output, 'response_shape_validation/c3/predict.py')
    run(output, 'response_shape_validation/c3/validate.py')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['demo', 'zero-field', 'predict', 'bounds'])
    parser.add_argument('--output', type=Path, required=True,
                        help='Run root; C2 data belong in response_shape_validation/c2 below this directory.')
    args = parser.parse_args()
    output = args.output.resolve()
    if args.stage in ['demo', 'zero-field'] and output.exists():
        raise FileExistsError('Use a new output folder; generated observations and frozen outputs are preserved.')
    output.mkdir(parents=True, exist_ok=True)
    stage_sources(output)
    if args.stage == 'demo':
        analytic_demo_inputs(output)
        predict(output)
        print('Completed synthetic analytic identity demo; this is not independent GPU evidence.')
    elif args.stage == 'zero-field':
        run(output, 'single_message_ep/primary/zero_field_ep.py')
        run(output, 'response_shape_validation/c1/run_c1.py')
        run(output, 'single_message_ep/independent/audit_zero_field_ep.py')
    elif args.stage == 'predict':
        predict(output)
    else:
        cal = json.loads((output / 'response_shape_validation/c3/calibration_public_frozen.json').read_text(encoding='utf-8'))
        if any('numerical_variation_envelope' not in c for c in cal['cases']):
            raise ValueError('C4 requires actual C2 refinement envelopes; the synthetic identity demo supplies none.')
        run(output, 'response_shape_validation/c4/run_c4.py', '--final-inputs-confirmed')


if __name__ == '__main__':
    main()

"""Portable checkpointed numerical tests. Run: python tests/verify.py --out OUT.

OUT is an external result folder, never a source directory. Checkpoints include
configuration, source/runtime identities and NumPy generator state. Re-run to
resume; --stop-after permits a fresh-process interruption/restart comparison.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from afm_observables import cluster_octupole,gamma_pair,registered_proxy
from afm_observables.multipole import solid_gradients
from afm_observables.transport import conductivity,resistivity,hall_resistivity_isotropic,assigned_vector_response
from afm_observables.gradient import require_coercive
from afm_observables.field_adapter import site_coefficients,evaluate_fields,cuda_real_coefficients
from afm_observables.variational import energy_field

def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def save(p,value):
    tmp=p.with_suffix('.tmp')
    with tmp.open('w',encoding='utf-8') as f:
        json.dump(value,f,sort_keys=True,indent=2,allow_nan=False)
        f.flush(); os.fsync(f.fileno())
    tmp.replace(p)

def test_rank3(rng):
    r=rng.normal(size=(2,6,3)); m=rng.normal(size=r.shape)
    value=cluster_octupole(r,m,[.5,.5])
    # Independent Cartesian Tgamma gradients, including the mz terms.
    x,y,z=np.moveaxis(r,-1,0)
    gx=np.stack([4*z*z-3*x*x-y*y,-2*x*y,8*x*z],axis=-1)
    gy=np.stack([-2*x*y,4*z*z-x*x-3*y*y,8*y*z],axis=-1)
    expected=np.sqrt(3/8)*np.array([np.sum(m*gx),np.sum(m*gy)])/2
    np.testing.assert_allclose(gamma_pair(value),expected,rtol=2e-14,atol=2e-14)
    np.testing.assert_allclose(cluster_octupole(r*1e-10,m*1e-23,[.5,.5]),value*1e-43,rtol=2e-14,atol=1e-56)
    np.testing.assert_array_equal(cluster_octupole(r,-m,[.5,.5]),-value)
    np.testing.assert_array_equal(cluster_octupole(-r,m,[.5,.5]),value)
    # Reordering sites and their moments together cannot change the multipole.
    order=rng.permutation(6)
    np.testing.assert_allclose(cluster_octupole(r[:,order],m[:,order],[.5,.5]),value,rtol=2e-14,atol=2e-14)

def test_transport(rng):
    for _ in range(32):
        s=10**rng.uniform(-2,7); h=s*rng.normal(size=3)
        rho=resistivity(s,h); anti=(rho-rho.T)/2
        np.testing.assert_allclose(hall_resistivity_isotropic(s,h),anti[[1,2,0],[2,0,1]],rtol=2e-13,atol=1e-15/s)
        np.testing.assert_allclose(resistivity(s,-h),rho.T,rtol=2e-13,atol=1e-15/s)
        a=rng.normal(size=(3,3)); spd=a@a.T+np.eye(3)
        np.testing.assert_allclose(conductivity(spd,h)@resistivity(spd,h),np.eye(3),rtol=0,atol=1e-7)
    for s in (0,-1,np.nan,np.inf):
        try: hall_resistivity_isotropic(s,[1,2,3])
        except ValueError: pass
        else: raise AssertionError('Invalid longitudinal input was accepted')
    np.testing.assert_array_equal(assigned_vector_response([1,0,0],2),[2,0,0])

def test_proxy_gradient(rng):
    m=rng.normal(size=(3,3))
    np.testing.assert_array_equal(registered_proxy(m,[0,0,0]),[0,0,0])
    shifted=m.copy(); shifted[1,0]+=1e-9
    assert np.linalg.norm(registered_proxy(m,[1,1,1])-registered_proxy(shifted,[1,1,1]))<1e-9
    good=np.tile(np.eye(3)*2e-11,(3,1,1)); require_coercive(good)
    bad=np.tile(np.full((3,3),-1.5e-11),(3,1,1))
    for i in range(3): bad[:,i,i]=2e-11
    try: require_coercive(bad)
    except ValueError: pass
    else: raise AssertionError('Indefinite continuum gradient accepted')

def test_field_adapter(rng):
    r=rng.normal(size=(2,6,3))*1e-10
    ids=np.tile(np.arange(6),(2,1))
    sizes=np.ones((2,6))*3e-23
    c=site_coefficients(r,ids,sizes,[.5,.5],6)
    fields=rng.normal(size=(6,3,2,3))
    output=evaluate_fields(c,fields)
    for i in range(2):
        for j in range(3):
            expected=cluster_octupole(r,np.tile(fields[:,:,i,j],(2,1,1))*sizes[:,:,None],[.5,.5])
            np.testing.assert_allclose(output[:,i,j],expected,rtol=2e-14,atol=1e-56)
    real=cuda_real_coefficients(c)
    np.testing.assert_array_equal(real[:7]+1j*real[7:],c)

def test_variational_faces(rng):
    # Mathematical derivative controls, not material parameters or predictions.
    coordinates=np.array(list(__import__('itertools').product(range(2),range(3),range(2))))
    edges=np.array([(i,j) for i in range(len(coordinates)) for j in range(i+1,len(coordinates))
                    if np.abs(coordinates[i]-coordinates[j]).sum()==1])
    m=rng.normal(size=(len(coordinates),3,3)); m/=np.linalg.norm(m,axis=2)[:,:,None]
    mu=rng.uniform(.5,1.5,size=m.shape[:2]); mu[0,1]=0
    a=rng.normal(size=(len(edges),3,3)); g=a@a.transpose(0,2,1)
    d=rng.normal(size=g.shape+(3,))*.1
    energy,gradient,field=energy_field(m,mu,edges,g,d)
    np.testing.assert_allclose(-field*mu[:,:,None],gradient,rtol=1e-14,atol=1e-14)
    for _ in range(12):
        v=rng.normal(size=m.shape); v/=np.linalg.norm(v)
        h=1e-5
        fd=(energy_field(m+h*v,mu,edges,g,d)[0]-energy_field(m-h*v,mu,edges,g,d)[0])/(2*h)
        np.testing.assert_allclose(fd,np.sum(gradient*v),rtol=1e-8,atol=2e-8)
    # Reversing a face requires D'=-D^T. G is reciprocal.
    reverse=energy_field(m,mu,edges[:,::-1],g,-d.transpose(0,2,1,3))
    np.testing.assert_allclose(reverse[0],energy,rtol=2e-14)
    np.testing.assert_allclose(reverse[1],gradient,rtol=2e-14,atol=2e-14)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--out',required=True)
    parser.add_argument('--stop-after',type=int)
    args=parser.parse_args(); out=Path(args.out).resolve()
    if out==ROOT or ROOT in out.parents:
        raise ValueError('Choose an output folder outside the source tree')
    out.mkdir(parents=True,exist_ok=True)
    identity=dict(code={p.relative_to(ROOT).as_posix():digest(p) for p in sorted(ROOT.rglob('*.py'))},
        python=digest(sys.executable),numpy=np.__version__,
        numerical_binary=digest(np._core._multiarray_umath.__file__),seed=20261109)
    ip=out/'identity.json'
    if ip.exists(): assert json.loads(ip.read_text())==identity,'Source/runtime identity changed'
    else: save(ip,identity)
    p=out/'checkpoint.json'; rng=np.random.default_rng(identity['seed'])
    state=json.loads(p.read_text()) if p.exists() else dict(next_index=0,passed=[],rng_state=rng.bit_generator.state)
    rng.bit_generator.state=state['rng_state']
    tasks=[test_rank3,test_transport,test_proxy_gradient,test_field_adapter,test_variational_faces]
    for i in range(state['next_index'],len(tasks)):
        tasks[i](rng)
        state.update(next_index=i+1,rng_state=rng.bit_generator.state)
        state['passed'].append(tasks[i].__name__)
        save(out/f'checkpoint_{i+1}.json',state); save(p,state)
        print('PASS',tasks[i].__name__)
        if args.stop_after and i+1>=args.stop_after: return
    save(out/'summary.json',dict(complete=True,**state))

if __name__=='__main__': main()

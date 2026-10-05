#!/usr/bin/env python3
"""AXON-R viewer: current multi-via-point protected fall vs unprotected.
HOW TO RUN THIS?
 python scripts/view_pose_library_via_points.py --model unitree_g1/g1_pendulum.xml --library build_pose/pose_library_via_points.json --p1-module generate_fall_dataset_final --p3-module phase3_pose_jerk_v7_real --scenario 3 --bin mid 
"""
import argparse, importlib, importlib.util, json, sys, time
from pathlib import Path
import mujoco
import mujoco.viewer
import numpy as np

LEAD_TIME_S=0.30
POST_IMPACT_HOLD_S=1.5
MAX_SIM_TIME_S=4.0
# Exact match to build_pose_library_via_points_real.py's BIN_MAGNITUDE_FRACTIONS
# -- the fraction of p1.MAGNITUDE_RANGES[fn_name]'s span each bin was actually
# tuned and gated against. A magnitude outside this range was NEVER seen
# during optimization or holdout gating, so testing against it (as this
# viewer's old fixed --magnitude 100.0 default always did) tells you nothing
# about whether the library's reported reduction is right -- it's a different,
# untested condition, not a check on the library.
BIN_MAGNITUDE_FRACTIONS={'low':(0.0,0.33),'mid':(0.33,0.66),'high':(0.66,1.0)}

def load_module(name_or_path):
    s=str(name_or_path)
    if s.endswith('.py'): s=s[:-3]
    p=Path(s)
    if p.exists():
        spec=importlib.util.spec_from_file_location(p.stem,str(p.resolve()))
        if spec is None or spec.loader is None: raise ImportError(f'Cannot load module: {p}')
        mod=importlib.util.module_from_spec(spec); sys.modules[p.stem]=mod
        spec.loader.exec_module(mod); return mod
    return importlib.import_module(s)

def resolve_path(root,v):
    p=Path(v); return p if p.is_absolute() else (root/p).resolve()

def find_scenario(p1,sid):
    for s in p1.SCENARIOS:
        if int(s[0])==sid: return s
    raise ValueError(f'Scenario {sid} not found in Phase-1 SCENARIOS.')

def parts(s):
    sid,cat,fn,d=s; return sid,cat,fn,0 if d is None else d

def make_disturbance(p1,model,fn,magnitude,direction):
    builder=p1.SCENARIO_BUILDERS.get(fn)
    if builder is None: raise KeyError(f"SCENARIO_BUILDERS has no entry for '{fn}'.")
    f=builder(model,magnitude,direction=direction)
    if not callable(f): raise TypeError(f"Scenario builder '{fn}' did not return a callable disturbance.")
    return f

def reset_data(model):
    d=mujoco.MjData(model); mujoco.mj_resetDataKeyframe(model,d,0); mujoco.mj_forward(model,d); return d

def geom_name(model,gid):
    if gid<0: return ''
    return mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,int(gid)) or ''

def ground_and_feet(model,p1):
    # Exact-ID resolution, matching phase3_pose_jerk_v7's own
    # ground_id = mj_name2id(..., "ground") / foot_ids = p1.get_foot_geom_ids(model)
    # exactly -- NOT name-substring heuristics. The viewer's job is to show
    # what the scoring pipeline actually measured, so it must use the same
    # geom/body resolution the pipeline uses, or its numbers aren't
    # comparable to the library's reported reductions even for a matching
    # condition. Fails loudly (like the pipeline does) rather than silently
    # picking a wrong geom by first-substring-match.
    ground=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'ground')
    if ground<0:
        raise RuntimeError("ground_and_feet: no geom named 'ground' -- this viewer no longer guesses by "
                            "substring match, since that can silently pick the wrong geom. Check the model.")
    fn=getattr(p1,'get_foot_geom_ids',None)
    if not callable(fn):
        raise RuntimeError("ground_and_feet: p1.get_foot_geom_ids is required (same source the pipeline uses) "
                            "-- the viewer no longer falls back to a name-substring guess for feet.")
    feet=set(int(x) for x in fn(model))
    return ground,feet

def contact_force(model,data,i):
    w=np.zeros(6); mujoco.mj_contactForce(model,data,i,w); return float(np.linalg.norm(w[:3]))

def impact_body_ids(model):
    # Exact match to phase3_pose_jerk_v7's impact_body_ids: body NAME ids,
    # not geom-name substring matching. A contact is classified by which
    # BODY the far geom belongs to, same as contact_peak_forces.
    pelvis=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'pelvis')
    bob=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'pendulum_bob')
    if pelvis<0 or bob<0:
        raise RuntimeError(f"impact_body_ids: unresolved body id(s) pelvis={pelvis} head={bob}")
    return {'pelvis':pelvis,'head':bob}

def contacts(model,data,ground,feet,body_ids):
    # Exact reimplementation of phase3_pose_jerk_v7's contact_peak_forces:
    # a contact counts only against `ground`, is classified by the BODY id
    # of the non-ground geom (not a name-substring guess), and anything
    # that isn't pelvis/head and isn't a foot lands in 'other' -- identical
    # bucketing rule to what the library's reported reductions were scored
    # with, so a matching condition's numbers are directly comparable.
    f={'head':0.0,'pelvis':0.0,'other':0.0}; active={k:False for k in f}
    for i in range(data.ncon):
        c=data.contact[i]
        if ground not in (c.geom1,c.geom2): continue
        other_g=c.geom2 if c.geom1==ground else c.geom1
        if other_g in feet: continue
        other_body=model.geom_bodyid[other_g]
        mag=contact_force(model,data,i)
        matched=False
        for name,bid in body_ids.items():
            if other_body==bid:
                if mag>f[name]: f[name]=mag
                active[name]=True; matched=True
        if not matched:
            if mag>f['other']: f['other']=mag
            active['other']=True
    return f,active

def natural_impact(p1,model,scenario,magnitude,direction,timing):
    _,_,fn,_=parts(scenario); ground,feet=ground_and_feet(model,p1); body_ids=impact_body_ids(model); dt=float(model.opt.timestep)
    stand=model.key_ctrl[0].copy(); d=reset_data(model); disturb=make_disturbance(p1,model,fn,magnitude,direction)
    impact=None
    for _ in range(int(MAX_SIM_TIME_S/dt)):
        t=float(d.time); d.ctrl[:]=stand; disturb(model,d,t); mujoco.mj_step(model,d)
        _,a=contacts(model,d,ground,feet,body_ids)
        if any(a.values()): impact=float(d.time); break
    if impact is None: raise RuntimeError('Natural fall did not produce non-foot ground contact.')
    trigger=max(0.0,impact-LEAD_TIME_S)
    d=reset_data(model); disturb=make_disturbance(p1,model,fn,magnitude,direction)
    while d.time<trigger:
        d.ctrl[:]=stand; disturb(model,d,float(d.time)); mujoco.mj_step(model,d)
    v=float(np.linalg.norm(d.qvel[:3]))
    return impact,trigger,v

def minimum_jerk(t,T,q0,q1):
    if T<=0: return q1.copy()
    tau=min(max(t/T,0.0),1.0); s=10*tau**3-15*tau**4+6*tau**5
    return q0+(q1-q0)*s

def multi_point_trajectory(t_since_trigger,q_start,traj):
    # Exact chained minimum-jerk rule used by phase3_pose_jerk_v7.
    w=[(0.0,np.asarray(q_start,dtype=float))]+[(float(t),np.asarray(q,dtype=float)) for t,q in traj]
    for i in range(1,len(w)):
        t0,q0=w[i-1]; t1,q1=w[i]
        if t_since_trigger<=t1 or i==len(w)-1:
            return minimum_jerk(max(t_since_trigger-t0,0.0),max(t1-t0,1e-9),q0,q1)
    return w[-1][1]

def select_entry(lib,sid,vbin):
    m=[]
    for key,e in lib.items():
        if key.startswith('_') or not isinstance(e,dict): continue
        if sid in e.get('scenario_ids',[]) and e.get('velocity_bin')==vbin: m.append((key,e))
    if not m: raise KeyError(f'No library entry for scenario {sid}, bin {vbin}.')
    if len(m)>1: raise ValueError(f'Ambiguous library matches: {[x[0] for x in m]}')
    return m[0]

def load_traj(entry):
    raw=entry.get('pose_traj')
    if not raw: raise ValueError('Selected entry has no pose_traj.')
    traj=[(float(x[0]),np.asarray(x[1],dtype=float)) for x in raw]
    ts=[x[0] for x in traj]
    if any(t<=0 for t in ts) or any(b<=a for a,b in zip(ts,ts[1:])):
        raise ValueError(f'pose_traj times must be strictly increasing and >0: {ts}')
    return traj

def run_viewer(p1,model,scenario,magnitude,direction,timing,protected,traj,pose_id,trigger):
    sid,cat,fn,_=parts(scenario); d=reset_data(model); stand=model.key_ctrl[0].copy()
    disturb=make_disturbance(p1,model,fn,magnitude,direction); ground,feet=ground_and_feet(model,p1); body_ids=impact_body_ids(model)
    q_start=None; triggered=False; first=False; impact=None; peak={'head':0.0,'pelvis':0.0,'other':0.0}
    with mujoco.viewer.launch_passive(model,d) as viewer:
        viewer.cam.distance=2.7; viewer.cam.azimuth=90; viewer.cam.elevation=-15
        wall=time.perf_counter(); print('\n'+'='*70); print('PROTECTED TRAJECTORY' if protected else 'UNPROTECTED'); print('='*70)
        print(f'Scenario : {sid} / {cat}'); print(f'Magnitude: {magnitude:.3f}'); print(f'Direction: {direction}')
        if protected:
            print(f'Pose ID  : {pose_id}'); print(f'Trigger  : {trigger:.3f}s')
            for i,(t,q) in enumerate(traj,1): print(f'  P{i}: t={t:.4f}s, ctrl_dim={q.size}')
        while viewer.is_running() and d.time<MAX_SIM_TIME_S:
            sim_t=float(d.time); rem=wall+sim_t-time.perf_counter()
            if rem>0: time.sleep(min(rem,0.004))
            d.ctrl[:]=stand; disturb(model,d,sim_t)
            if protected and not triggered and sim_t>=trigger:
                q_start=d.qpos[7:7+model.nu].copy(); triggered=True; print(f'[TRIGGER] {pose_id} at {sim_t:.3f}s')
            if protected and triggered:
                qcmd=multi_point_trajectory(sim_t-trigger,q_start,traj)
                if qcmd.size!=model.nu: raise ValueError(f'Trajectory control dimension {qcmd.size} != model.nu={model.nu}')
                d.ctrl[:model.nu]=qcmd
            mujoco.mj_step(model,d); forces,active=contacts(model,d,ground,feet,body_ids)
            for k in peak: peak[k]=max(peak[k],forces[k])
            if any(active.values()) and not first:
                first=True; impact=float(d.time)
                print(f'[CONTACT] non-foot ground contact at {impact:.3f}s')
                print(f"[CONTACT FORCE] head={forces['head']:.2f} N, pelvis={forces['pelvis']:.2f} N, other={forces['other']:.2f} N")
            viewer.sync()
            if first and d.time>=impact+POST_IMPACT_HOLD_S: break
        viewer.sync()
    print(f"\n[{('PROTECTED TRAJECTORY' if protected else 'UNPROTECTED')} PEAK FORCES]")
    for k in peak: print(f'  {k.title():7s}: {peak[k]:.2f} N')
    return peak

def main():
    ap=argparse.ArgumentParser(description='Current AXON-R multi-via-point protected vs unprotected MuJoCo viewer.')
    ap.add_argument('--model',required=True); ap.add_argument('--library',required=True)
    ap.add_argument('--p1-module',default='generate_fall_dataset_final'); ap.add_argument('--p3-module',default=None)
    ap.add_argument('--scenario',type=int,default=6); ap.add_argument('--bin',dest='velocity_bin',choices=('low','mid','high'))
    ap.add_argument('--magnitude',type=float,default=None,
                     help='Disturbance magnitude. If omitted, derived from the SAME '
                          'p1.MAGNITUDE_RANGES[fn_name] x BIN_MAGNITUDE_FRACTIONS[bin] logic the '
                          'optimizer and gate actually used -- the bin midpoint. Pass a value explicitly '
                          'ONLY to deliberately stress-test outside the calibrated envelope; the printed '
                          'range tells you whether you are doing that.')
    ap.add_argument('--magnitude-frac',type=float,default=0.5,
                     help='Where in the bin fraction range (0=bin low edge, 1=bin high edge) to place the '
                          'derived default magnitude. Ignored if --magnitude is given explicitly.')
    ap.add_argument('--timing',type=float,default=0.0)
    args=ap.parse_args(); root=Path.cwd(); model_path=resolve_path(root,args.model); library_path=resolve_path(root,args.library)
    if not model_path.exists(): raise FileNotFoundError(f'Model not found: {model_path}')
    if not library_path.exists(): raise FileNotFoundError(f'Library not found: {library_path}')
    p1=load_module(args.p1_module)
    if args.p3_module:
        p3=load_module(args.p3_module)
        if not hasattr(p3,'multi_point_trajectory'): print('[INFO] Phase-3 module has no multi_point_trajectory; local equivalent used.')
        else: print('[INFO] Phase-3 multi_point_trajectory found; viewer follows the same definition.')
    scenario=find_scenario(p1,args.scenario); sid,cat,fn,direction=parts(scenario); model=mujoco.MjModel.from_xml_path(str(model_path))
    with open(library_path,'r',encoding='utf-8') as f: lib=json.load(f)
    vbin=args.velocity_bin or input('Select velocity bin [low/mid/high]: ').strip().lower()
    if vbin not in ('low','mid','high'): raise ValueError('Velocity bin must be low, mid, or high.')
    key,entry=select_entry(lib,sid,vbin); traj=load_traj(entry)
    mrange=getattr(p1,'MAGNITUDE_RANGES',{}).get(fn)
    if mrange is None:
        print(f"[WARNING] p1.MAGNITUDE_RANGES has no entry for '{fn}' -- cannot derive or validate a "
              "bin-appropriate magnitude. --magnitude is required in this case.")
        if args.magnitude is None: raise ValueError(f"--magnitude is required: no MAGNITUDE_RANGES['{fn}']")
        magnitude=args.magnitude
    else:
        lo,hi=mrange; f0,f1=BIN_MAGNITUDE_FRACTIONS[vbin]
        bin_lo,bin_hi=lo+f0*(hi-lo),lo+f1*(hi-lo)
        if args.magnitude is None:
            magnitude=bin_lo+args.magnitude_frac*(bin_hi-bin_lo)
            print(f"[MAGNITUDE] derived {magnitude:.3f} from bin '{vbin}' of '{fn}' "
                  f"(calibrated range [{lo:.3f},{hi:.3f}], bin range [{bin_lo:.3f},{bin_hi:.3f}])")
        else:
            magnitude=args.magnitude
            in_range='inside' if bin_lo<=magnitude<=bin_hi else 'OUTSIDE'
            print(f"[MAGNITUDE] using explicit {magnitude:.3f} -- {in_range} bin '{vbin}''s calibrated "
                  f"range [{bin_lo:.3f},{bin_hi:.3f}] (full scenario range [{lo:.3f},{hi:.3f}])")
            if in_range=='OUTSIDE':
                print("[WARNING] this magnitude was never seen during this pose's optimization or holdout "
                      "gating -- a bad result here says nothing about whether the library's reported "
                      "reduction is correct, only about behavior outside its designed envelope.")
    for i,(_,q) in enumerate(traj,1):
        if q.size!=model.nu: raise ValueError(f'P{i} has {q.size} controls, model.nu={model.nu}.')
    print('\n'+'='*70); print('AXON-R CURRENT POSE LIBRARY: PROTECTED vs UNPROTECTED'); print('='*70)
    print(f'Scenario : {sid} / {cat} ({fn})'); print(f'Bin      : {vbin}'); print(f'Entry    : {key}'); print(f'Mode     : {entry.get("mode","unknown")}'); print(f'Kind     : {entry.get("kind","unknown")}'); print(f'Gate     : {"PASSED" if entry.get("gate_passed") else "FAILED"}'); print(f'Points   : {len(traj)}')
    if entry.get('kind')!='pose_traj': print('[WARNING] This entry is not a validated optimized trajectory; choose a gate-passed pose_traj entry for demonstration.')
    natural,trigger,v=natural_impact(p1,model,scenario,magnitude,direction,args.timing)
    print(f'\nNatural impact time : {natural:.3f}s'); print(f'Trigger time        : {trigger:.3f}s'); print(f'Velocity at trigger : {v:.3f} m/s')
    print('\nControls: u=unprotected, p=protected trajectory, r=protected again, q=quit')
    u=p=None
    while True:
        try: choice=input('Choice [u/p/r/q]: ').strip().lower()
        except (EOFError,KeyboardInterrupt): choice='q'
        if choice=='q': print('Viewer closed.'); break
        if choice=='u': u=run_viewer(p1,model,scenario,magnitude,direction,args.timing,False,None,None,trigger)
        elif choice in ('p','r'): p=run_viewer(p1,model,scenario,magnitude,direction,args.timing,True,traj,key,trigger)
        else: print('Use u, p, r, or q.'); continue
        if u is not None and p is not None:
            print('\n'+'='*70); print('PEAK FORCE COMPARISON (single visualized condition)'); print('='*70); print(f"{'Body':10s}{'Unprotected':>16s}{'Protected':>16s}{'Reduction':>14s}")
            for k in ('head','pelvis','other'):
                red=100*(u[k]-p[k])/u[k] if u[k]>1e-9 else None
                print(f'{k:10s}{u[k]:16.2f}{p[k]:16.2f}{(f"{red:.2f}%" if red is not None else "N/A"):>14s}')
            print('\nNote: this is one visualized condition; it is not the library holdout median.')

if __name__=='__main__': main()

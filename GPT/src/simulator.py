"""Headless TraCI wrapper for deterministic 4x4 signal-control experiments."""
from __future__ import annotations
import json, os, time
from pathlib import Path
import numpy as np
import sumo
os.environ.setdefault("SUMO_HOME",sumo.SUMO_HOME)
import traci
from .network import TLS_IDS, SCENARIOS, geometry_hash, make_routes

class TrafficEnv:
 def __init__(self,net_file:Path,work_dir:Path,scenario:str,seed:int,duration:int=600):
  if scenario not in SCENARIOS and scenario not in ("moderate_surge","variable_arrival","randomized_route"): raise ValueError(f"invalid scenario: {scenario}")
  self.net_file=Path(net_file); self.work=Path(work_dir); self.work.mkdir(parents=True,exist_ok=True)
  if not self.net_file.exists(): raise FileNotFoundError(self.net_file)
  self.scenario=scenario; self.seed=seed; self.duration=duration; self.route=self.work/f"{scenario}_{seed}.rou.xml"
  self.demand=make_routes(self.route,scenario,seed,duration); self.rng=np.random.default_rng(seed); self.step_count=0; self.arrived=0; self.departed=0
  self.wait_sum=0.; self.speed_sum=0.; self.vehicle_samples=0; self.queue_sum=0.; self.stops=0; self.reward_sum=0.; self.action_lat=[]; self.failed_actions=0
 def start(self):
  binary=str(Path(sumo.SUMO_HOME)/"bin/sumo")
  traci.start([binary,"-n",str(self.net_file),"-r",str(self.route),"--seed",str(self.seed),"--no-step-log","true","--no-warnings","true","--time-to-teleport","-1","--duration-log.disable","true"])
  ids=tuple(sorted(traci.trafficlight.getIDList()))
  if ids!=tuple(sorted(TLS_IDS)): traci.close(); raise RuntimeError(f"expected 16 signal IDs, got {len(ids)}: {ids}")
  self.lanes={t:tuple(dict.fromkeys(traci.trafficlight.getControlledLanes(t))) for t in TLS_IDS}
  self.green_phases={}
  for t in TLS_IDS:
   phases=traci.trafficlight.getAllProgramLogics(t)[0].phases
   greens=[i for i,p in enumerate(phases) if "G" in p.state and "y" not in p.state]
   if len(greens)<2: raise RuntimeError(f"{t} lacks two legal green phases")
   self.green_phases[t]=(greens[0],greens[1])
  if self.scenario=="road_closure":
   lane="E_J1_1__J2_1_0"
   if lane not in traci.lane.getIDList(): raise RuntimeError(f"closure lane missing: {lane}")
   traci.lane.setMaxSpeed(lane,.1)
  return self.observe()
 def observe(self):
  obs=[]; masks=[]
  for ti,t in enumerate(TLS_IDS):
   vals=[]
   for lane in self.lanes[t]: vals.extend([traci.lane.getLastStepHaltingNumber(lane)/20.,traci.lane.getLastStepMeanSpeed(lane)/13.89])
   # Fixed compact representation: mean/max queue, mean speed, occupancy, phase, availability mask.
   q=np.array(vals[0::2] or [0.]); s=np.array(vals[1::2] or [0.]); base=np.array([q.mean(),q.max(),s.mean(),len(traci.vehicle.getIDList())/200.,traci.trafficlight.getPhase(t)%2,1.],dtype=np.float32)
   mask=np.ones(6,dtype=np.float32)
   if self.scenario=="noisy_sensors": base[:4]+=self.rng.normal(0,.04,4).astype(np.float32)
   if self.scenario=="missing_sensors" and ti%4==0: base[:4]=0; mask[:4]=0
   obs.append(np.concatenate([base,mask])); masks.append(np.array([1,0] if self.scenario=="partial_signal_failure" and ti%5==0 else [1,1],dtype=bool))
  return np.stack(obs),np.stack(masks)
 def step(self,actions,hold=5):
  if len(actions)!=16: raise ValueError("actions must contain exactly 16 values")
  t0=time.perf_counter_ns()
  for i,(tls,a) in enumerate(zip(TLS_IDS,actions)):
   legal=[0] if self.scenario=="partial_signal_failure" and i%5==0 else [0,1]
   if int(a) not in legal: self.failed_actions+=1; a=legal[0]
   traci.trafficlight.setPhase(tls,self.green_phases[tls][int(a)])
  self.action_lat.append((time.perf_counter_ns()-t0)/1e6/16)
  for _ in range(hold):
   if self.step_count>=self.duration: break
   traci.simulationStep(); self.step_count+=1; self.arrived+=traci.simulation.getArrivedNumber(); self.departed+=traci.simulation.getDepartedNumber()
   ids=traci.vehicle.getIDList(); queues=sum(traci.lane.getLastStepHaltingNumber(x) for x in traci.lane.getIDList() if not x.startswith(":")); self.queue_sum+=queues
   for v in ids:
    speed=traci.vehicle.getSpeed(v); self.speed_sum+=speed; self.wait_sum+=traci.vehicle.getWaitingTime(v); self.stops+=int(speed<.1); self.vehicle_samples+=1
  reward=-(self.queue_sum/max(1,self.step_count))/50.; self.reward_sum+=reward
  return self.observe(),reward,self.step_count>=self.duration
 def close(self):
  try: traci.close()
  except Exception: pass
 def metrics(self):
  wall_steps=max(1,self.step_count); vs=max(1,self.vehicle_samples)
  return {"geometry_hash":geometry_hash(),"planned_vehicles":self.demand["vehicles"],"departed_vehicles":self.departed,"arrived_vehicles":self.arrived,"unfinished_vehicles":max(0,self.departed-self.arrived),"throughput_vph":self.arrived*3600/wall_steps,"mean_waiting_time_s":self.wait_sum/vs,"mean_speed_mps":self.speed_sum/vs,"mean_queue_vehicles":self.queue_sum/wall_steps,"stops":self.stops,"cumulative_reward":self.reward_sum,"failed_actions":self.failed_actions,"mean_action_latency_ms":float(np.mean(self.action_lat)) if self.action_lat else None,"action_latency_samples":len(self.action_lat),"simulator_steps":self.step_count}

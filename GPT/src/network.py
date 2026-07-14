"""Deterministic SUMO network and demand generation for a 4x4 signal grid."""
from __future__ import annotations
import hashlib, json, math, random, subprocess
from pathlib import Path
from xml.etree import ElementTree as ET

GRID=4; SPACING=150.0
TLS_IDS=tuple(f"J{x}_{y}" for y in range(GRID) for x in range(GRID))
SCENARIOS=("normal","high_demand","sudden_surge","uneven_directional","road_closure","noisy_sensors","missing_sensors","partial_signal_failure")
TRAIN_PATTERNS=("normal","uneven_directional","moderate_surge","variable_arrival","randomized_route")

def geometry_hash():
    return hashlib.sha256(json.dumps({"grid":GRID,"spacing":SPACING,"tls":TLS_IDS,"bidirectional":True},sort_keys=True).encode()).hexdigest()
def edge(a,b): return f"E_{a}__{b}"
def boundaries(): return tuple(f"{s}_{i}" for i in range(GRID) for s in "WESN")

def build_network(out_dir:Path,netconvert:str)->Path:
    out_dir.mkdir(parents=True,exist_ok=True); nodes=ET.Element("nodes"); edges=ET.Element("edges")
    for y in range(GRID):
      for x in range(GRID): ET.SubElement(nodes,"node",id=f"J{x}_{y}",x=str(x*SPACING),y=str(y*SPACING),type="traffic_light")
    for i in range(GRID):
      for name,x,y in [(f"W_{i}",-SPACING,i*SPACING),(f"E_{i}",GRID*SPACING,i*SPACING),(f"S_{i}",i*SPACING,-SPACING),(f"N_{i}",i*SPACING,GRID*SPACING)]:
        ET.SubElement(nodes,"node",id=name,x=str(x),y=str(y),type="priority")
    def add(a,b): ET.SubElement(edges,"edge",id=edge(a,b),attrib={"from":a,"to":b,"numLanes":"1","speed":"13.89"})
    for y in range(GRID):
      for x in range(GRID-1): add(f"J{x}_{y}",f"J{x+1}_{y}"); add(f"J{x+1}_{y}",f"J{x}_{y}")
    for x in range(GRID):
      for y in range(GRID-1): add(f"J{x}_{y}",f"J{x}_{y+1}"); add(f"J{x}_{y+1}",f"J{x}_{y}")
    for i in range(GRID):
      for b,j in [(f"W_{i}",f"J0_{i}"),(f"E_{i}",f"J3_{i}"),(f"S_{i}",f"J{i}_0"),(f"N_{i}",f"J{i}_3")]: add(b,j); add(j,b)
    nod=out_dir/"grid.nod.xml"; edg=out_dir/"grid.edg.xml"; net=out_dir/"grid.net.xml"
    ET.ElementTree(nodes).write(nod,encoding="utf-8",xml_declaration=True); ET.ElementTree(edges).write(edg,encoding="utf-8",xml_declaration=True)
    p=subprocess.run([netconvert,"--node-files",str(nod),"--edge-files",str(edg),"--no-turnarounds","true","--tls.default-type","static","-o",str(net)],capture_output=True,text=True)
    if p.returncode: raise RuntimeError(f"netconvert failed: {p.stderr.strip()}")
    (out_dir/"geometry.json").write_text(json.dumps({"geometry_hash":geometry_hash(),"tls_ids":TLS_IDS,"controlled_intersections":16,"spacing_m":SPACING},indent=2)+"\n")
    return net

def path_nodes(start,end,rng,randomized=False):
    def adjacent(b):
      i=int(b.split("_")[1]); return {"W":(0,i),"E":(3,i),"S":(i,0),"N":(i,3)}[b[0]]
    cur=list(adjacent(start)); target=adjacent(end); nodes=[start,f"J{cur[0]}_{cur[1]}"]
    moves=[(1 if target[0]>cur[0] else -1,0)]*abs(target[0]-cur[0])+[(0,1 if target[1]>cur[1] else -1)]*abs(target[1]-cur[1])
    if randomized: rng.shuffle(moves)
    for dx,dy in moves: cur[0]+=dx; cur[1]+=dy; nodes.append(f"J{cur[0]}_{cur[1]}")
    nodes.append(end); return nodes

def make_routes(out:Path,scenario:str,seed:int,duration:int=600):
    valid=set(SCENARIOS)|set(TRAIN_PATTERNS)
    if scenario not in valid: raise ValueError(f"unknown scenario {scenario!r}; expected one of {sorted(valid)}")
    if not isinstance(seed,int): raise TypeError("seed must be an integer")
    if duration<=0: raise ValueError("duration must be positive")
    rng=random.Random(seed); root=ET.Element("routes"); ET.SubElement(root,"vType",id="car",accel="2.6",decel="4.5",sigma="0.5",length="5",maxSpeed="13.89")
    rates={"normal":.16,"high_demand":.34,"sudden_surge":.15,"uneven_directional":.23,"road_closure":.18,"noisy_sensors":.18,"missing_sensors":.18,"partial_signal_failure":.18,"moderate_surge":.20,"variable_arrival":.18,"randomized_route":.18}
    opposite={"W":"E","E":"W","S":"N","N":"S"}; routes=[]; vid=0
    for t in range(duration):
      rate=rates[scenario]
      if scenario in ("sudden_surge","moderate_surge") and duration*.35<=t<duration*.65: rate*=2.3
      if scenario=="variable_arrival": rate*=.45+1.1*(.5+.5*math.sin(t/37))
      for _ in range(int(rate)+(rng.random()<rate%1)):
        side=rng.choice("WESN") if scenario!="uneven_directional" or rng.random()>.75 else rng.choice("WE")
        start=f"{side}_{rng.randrange(GRID)}"; end=f"{opposite[side]}_{rng.randrange(GRID)}"
        nodes=path_nodes(start,end,rng,scenario=="randomized_route"); route=[edge(a,b) for a,b in zip(nodes,nodes[1:])]
        ET.SubElement(root,"route",id=f"r{vid}",edges=" ".join(route)); ET.SubElement(root,"vehicle",id=f"v{vid}",type="car",route=f"r{vid}",depart=str(t),departLane="best")
        routes.append(route); vid+=1
    if not routes: raise RuntimeError("demand generation produced no routes")
    ET.ElementTree(root).write(out,encoding="utf-8",xml_declaration=True)
    return {"vehicles":vid,"min_route_edges":min(map(len,routes)),"max_route_edges":max(map(len,routes))}

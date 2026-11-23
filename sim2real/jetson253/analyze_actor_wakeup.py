"""Join scheduler tracepoints to the slowest native state-receipt intervals."""
import argparse
from bisect import bisect_left, bisect_right
from collections import Counter
import csv
import json
from pathlib import Path
import re

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("run", type=Path)
parser.add_argument("tracedir", type=Path)
args=parser.parse_args()
pid=int((args.tracedir/"pid").read_text())
pattern=re.compile(r"\[(\d+)\].*?\s(\d+)\.(\d+):\s+(\S+?):\s*(.*)")
def read_events(name):
    events=[]
    for line in (args.tracedir/f"{name}.trace").open():
        m=pattern.search(line)
        if not m: continue
        cpu,sec,frac,event,detail=m.groups()
        ts=int(sec)*10**9+int(frac.ljust(9,"0")[:9])
        events.append((ts,int(cpu),event,detail))
    events.sort(key=lambda x:x[0])
    return events
def rows(name):
    with (args.run/name).open() as f: return list(csv.DictReader(f))
can={int(r['sequence']):r for r in rows("can.json.timing.csv")}
actor=rows("native.bin.ipc.csv")
main=read_events("main")
detail=read_events("detail")
mt=[ev[0] for ev in main]
dt=[ev[0] for ev in detail]
assert main and detail, "Missing trace events"
def fmt(e,ref):
    t,c,k,v=e
    return [round((t-ref)/1e6,6),c,k,v]
def task_label(d,prefix):
    name=re.search(prefix+r"_comm=(.*?) "+prefix+r"_pid=(\d+)",d)
    return (name[1]+"-"+name[2]) if name else "unknown"
def occupancy(cpu, start, end):
    left=bisect_left(dt,start)
    right=bisect_right(dt,end)
    switches=[e for e in detail[left:right] if e[1]==cpu and e[2]=="sched_switch"]
    if not (dt[0]<=start<end<=dt[-1]):return None
    last=start; duration=Counter()
    if switches: task=task_label(switches[0][3],"prev")
    else:
        prior=next((detail[i] for i in range(left-1,-1,-1) if detail[i][1]==cpu and detail[i][2]=='sched_switch'),None)
        if prior is None:return None
        task=task_label(prior[3],"next")
    for t,_,_,d in switches:
        duration[task]+=t-last; task=task_label(d,"next"); last=t
    duration[task]+=end-last
    return {k:round(v/1e6,6) for k,v in duration.most_common()}
rank=[]
for r in actor:
    s=int(r['sequence'])
    if s<=1 or s not in can:continue
    p=int(can[s]['publish_ns']); recv=int(r['receive_ns'])
    if p>0 and recv>p:
        rank.append((recv-p,s,p,recv,int(r['infer_start_ns']),int(r['infer_end_ns'])))
def analyze(x):
    age,s,p,recv,i0,i1=x
    lo=bisect_left(mt,p-100000);hi=bisect_right(mt,recv+100000)
    ev=main[lo:hi]
    wake=next((t for t,c,k,d in ev if k=='sched_wakeup' and t>=p-1000),None)
    waking=next((t for t,c,k,d in ev if k=='sched_waking' and t>=p-1000),None)
    ins=next((t for t,c,k,d in ev if k=='sched_switch' and re.search(rf'next_pid={pid}\b',d) and t>=p-1000),None)
    return dict(seq=s,publish_ns=p,receipt_ms=age/1e6,infer_ms=(i1-i0)/1e6,
        publish_to_waking_ms=(waking-p)/1e6 if waking else None,
        waking_to_runnable_ms=(wake-waking)/1e6 if wake and waking else None,
        runnable_to_first_switch_ms=(ins-wake)/1e6 if ins and wake else None,
        switch_to_receipt_ms=(recv-ins)/1e6 if ins else None,
        cpu4_scheduled_ms=occupancy(4,p,recv),events=[fmt(e,p) for e in ev],
        cpu4_detail=[fmt(e,p) for e in detail[bisect_left(dt,p-50000):bisect_right(dt,recv+50000)] if e[1]==4][:250])
out=dict(run=str(args.run),trace=str(args.tracedir),pid=pid,main_trace_range_ns=[mt[0],mt[-1]],
    detail_trace_range_ns=[dt[0],dt[-1]],main_trace_stats=(args.tracedir/'main.stats').read_text(),
    detail_trace_stats=(args.tracedir/'detail.stats').read_text(),frames=len(rank),
    worst_receipts=[analyze(x) for x in sorted(rank,reverse=True)[:10]])
i=max(rank,key=lambda x:x[5]-x[4])
out['slowest_inference']=dict(seq=i[1],ms=(i[5]-i[4])/1e6,cpu4=occupancy(4,i[4],i[5]),cpu5=occupancy(5,i[4],i[5]))
path=args.run/'scheduler-diagnosis.json'
path.write_text(json.dumps(out,indent=2)+'\n')
print('range',out['detail_trace_range_ns'],'frames',len(rank),'inference',out['slowest_inference'])
for r in out['worst_receipts'][:5]:
    print({k:v for k,v in r.items() if k not in ('events','cpu4_detail')})
print('worst frame',out['worst_receipts'][0]['events'])

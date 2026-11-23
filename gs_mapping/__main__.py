"""python3 -m gs_mapping JOURNAL_DIR --out NEW_DIR [--backend cuda] [--follow]"""
import argparse
from dataclasses import asdict
from pathlib import Path
import json
from .policy import Config
from .replay import Session, records


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("journal",type=Path)
    p.add_argument("--out",required=True,type=Path)
    p.add_argument("--config",type=Path)
    p.add_argument("--backend",choices=("cpu","cuda"),default="cpu")
    p.add_argument("--follow",action="store_true")
    p.add_argument("--fixed",action="store_true")
    p.add_argument("--holdout-stride",type=int,default=0)
    args=p.parse_args()
    if args.holdout_stride!=0 and args.holdout_stride<2:
        p.error("holdout stride must be zero or at least two")
    cfg=Config(**(json.loads(args.config.read_text()) if args.config else {}))
    if args.fixed:
        cfg.fixed=True
    cfg.check()
    if args.backend=="cuda":
        from .gs_livo_cuda import extension
        extension()  # build/device errors must not result in a silent CPU run
    events=args.journal/"events.jsonl"
    if not events.is_file():
        p.error("missing "+str(events))
    args.out.mkdir(parents=True,exist_ok=False)
    s=Session(args.journal,args.out,cfg,args.backend,args.holdout_stride)
    try:
        for event in records(events,args.follow):
            s.event(event)
    except KeyboardInterrupt:
        pass
    if not s.header:
        p.error("empty journal")
    s.flush()
    summary=dict(backend=args.backend,config=asdict(cfg),epochs=s.reports)
    (args.out/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    print(json.dumps(s.reports,ensure_ascii=False))


if __name__=="__main__":
    main()

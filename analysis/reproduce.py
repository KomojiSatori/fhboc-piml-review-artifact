"""Reproduce manuscript statistics from the accompanying reviewer archive."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.special import expit
from scipy.stats import beta


ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def truth(value):
    return value is True or str(value).lower() in ("true", "1")


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def csvout(path, records):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)


def verify_files():
    manifest = read(ROOT / "MANIFEST.json")
    for name, info in manifest["files"].items():
        p = ROOT / name
        assert p.is_file(), f"Missing archive file: {name}"
        assert p.stat().st_size == info["bytes"], f"Size mismatch: {name}"
        assert hashlib.sha256(p.read_bytes()).hexdigest() == info["sha256"], name
    return len(manifest["files"])


def main_experiment(out):
    events = []
    features, references = [], []
    calibration_checks = 0
    for folder in sorted((ROOT / "data/main").glob("c*_g*")):
        current = rows(folder / "events.jsonl")
        assert len({(r["observation"], r["method"]) for r in current}) == len(current)
        events.extend(current)
        cal = read(folder / "calibration.json")
        groups = defaultdict(list)
        for r in cal["records"]:
            groups[r["method"]].append(r["answer"]["score"])
        for method, scores in groups.items():
            assert len(scores) == 400
            assert sorted(scores)[392] == cal["gates"][method]
            calibration_checks += 1
        by_key = {(r["observation"], r["method"]): r for r in current}
        for r in current:
            assert r["threshold"] == cal["gates"][r["method"]]
            expected = not r["stopped"] and r["score"] > r["threshold"]
            assert r["detected"] == expected, (folder.name, r["observation"], r["method"])
            if r["signal"] and r["detected"]:
                hit = abs(r["fd_error_hz"]) <= 250 and abs(r["tau_error_samples"]) <= 1
                assert r["handoff"] == hit
            else:
                assert not r["handoff"]
            if r["method"] in ("manual-stop", "learned-stop"):
                ref = by_key[r["observation"], "PIML-64"]
                assert r["detected"] == (ref["detected"] and not r["stopped"])
                assert r["handoff"] == (ref["handoff"] and not r["stopped"])
        for r in rows(folder / "controller_inputs.jsonl"):
            features.append(r["features"])
            references.append(by_key[r["observation"], "PIML-64"])
    assert len(events) == 191520 and calibration_checks == 90
    groups = defaultdict(list)
    for r in events:
        groups[r["condition"], r["stratum"], r["cn0"], r["method"]].append(r)
    summary = []
    for (c, s, cn, m), rr in groups.items():
        n = len(rr)
        k, h = sum(r["detected"] for r in rr), sum(r["handoff"] for r in rr)
        summary.append(dict(experiment="MAIN_GAUSSIAN", condition=c, stratum=s,
                            cn0=cn, method=m, n=n, detected=k, handoff=h,
                            Pd=k/n if cn is not None else None,
                            Ph=h/n if cn is not None else None,
                            Pfa=k/n if cn is None else None,
                            mean_seconds=float(np.mean([r["seconds"] for r in rr]))))
    expected = rows(ROOT / "reference/main_performance.csv")
    lookup = {(r["condition"], r["stratum"], r["cn0"], r["method"]): r for r in summary}
    for r in expected:
        key = (int(r["condition"]), r["stratum"], float(r["cn0"]) if r["cn0"] else None, r["method"])
        actual = lookup[key]
        for field in ("n", "detected", "handoff"):
            assert actual[field] == int(r[field]), (key, field)
        assert abs(actual["mean_seconds"]-float(r["mean_seconds"])) < 1e-10
    csvout(out / "main_performance.csv", summary)
    method_totals = {}
    for m in sorted({r["method"] for r in events}):
        rr = [r for r in events if r["method"] == m]
        h1, h0 = [r for r in rr if r["signal"]], [r for r in rr if not r["signal"]]
        method_totals[m] = dict(H1=len(h1), test_H0=len(h0), detected=sum(r["detected"] for r in h1),
                                handoff=sum(r["handoff"] for r in h1), false_alarms=sum(r["detected"] for r in h0),
                                mean_ms=1000*float(np.mean([r["seconds"] for r in rr])))
    expected_totals = read(ROOT / "reference/expected_claims.json")["methods"]
    for m, r in method_totals.items():
        for k in ("detected", "handoff", "false_alarms"):
            assert r[k] == expected_totals[m][k], (m, k)
        assert abs(r["mean_ms"]-expected_totals[m]["mean_ms"]) < 1e-8
    # Same geometry/noise hierarchy and fixed RNG as the manuscript interval.
    rr = [r for r in events if r["condition"] == 2 and r["stratum"] == "nominal"
          and r["cn0"] == 34 and r["method"] == "BASIC-quarter-ms"]
    full = {(r["geometry"],r["observation"]): r for r in events if r["condition"] == 2 and r["method"] == "PIML-full"}
    by = [[float(r["handoff"])-float(full[r["geometry"],r["observation"]]["handoff"])
           for r in rr if r["geometry"] == g] for g in sorted({r["geometry"] for r in rr})]
    rng = np.random.default_rng(2026101702)
    boot = [np.mean(np.concatenate([rng.choice(by[j],len(by[j]),replace=True)
                                   for j in rng.integers(0,len(by),len(by))])) for _ in range(2000)]
    interval = (-100*np.percentile(boot,[97.5,2.5])).tolist()
    assert np.allclose(interval,[10.833333333333334,24.791666666666668])
    # Portable JSON inference: no executable model pickle or sklearn needed.
    x = np.asarray(features, dtype=float)
    signal = np.array([r["signal"] for r in references])
    detected = np.array([r["detected"] for r in references])
    handoff = np.array([r["handoff"] for r in references])
    policies = read(ROOT / "models/stop_variants.json")
    ablations = []
    for name,p in policies.items():
        probability = expit(x[:,p["columns"]] @ np.asarray(p["raw_weights"]) + p["raw_bias"])
        stop = probability <= .1
        ablations.append(dict(model=name,dimensions=len(p["columns"]),continued=int((~stop).sum()),
                              lost_detections=int((signal & detected & stop).sum()),lost_handoffs=int((handoff & stop).sum())))
    expected = read(ROOT / "reference/expected_claims.json")["ablations"]
    for r in ablations:
        if r["model"] not in expected:continue
        for k,src in (("continued","continued"),("lost_detections","detection_losses"),("lost_handoffs","handoff_losses")):
            assert r[k] == expected[r["model"]][src], (r["model"],k)
    csvout(out / "controller_input_ablation.csv",ablations)
    # Actual role seeds and cross-strength pairing in the released metadata.
    seed_roles = defaultdict(set)
    for folder in sorted((ROOT / "data/main").glob("c*_g*")):
        metadata = read(folder / "geometry.json")["metadata"]
        for m in metadata:seed_roles[m["role"]].add(int(m["noise_seed"]))
    assert {k:len(v) for k,v in seed_roles.items()} == {"H1":1440,"calibration_H0":7200,"holdout_H0":14400}
    assert all(not seed_roles[a]&seed_roles[b] for a in seed_roles for b in seed_roles if a != b)
    report = dict(methods=method_totals,bootstrap_high_nominal_34_difference_pp=interval,
                  calibration_thresholds_verified=calibration_checks,method_events=len(events),
                  seed_counts={k:len(v) for k,v in seed_roles.items()},controller_variants=len(ablations))
    write(out / "main_claims.json",report)
    return summary,report


def component_summary(path,out):
    data = rows(path)
    group = defaultdict(list)
    for r in data:
        if truth(r["signal"]):
            group[int(r["condition"]),float(r["cn0_db_hz"]),r["method"]].append(r)
    result=[]
    for (c,cn,m),rr in sorted(group.items()):
        result.append(dict(experiment="COMPONENT_GAUSSIAN_6F",condition=c,cn0=cn,method=m,
                           H1=len(rr),detected=sum(truth(r["detected"]) for r in rr),
                           handoff=sum(truth(r["handoff"]) for r in rr)))
    assert {r["cn0"] for r in result} == {32,36}
    csvout(out / "component_independent_calibration.csv",result)
    empirical=[]
    for (c,cn,m),rr in sorted(group.items()):
        bg=[r for r in data if int(r['condition'])==c and r['method']==m and r['role']=='holdout_H0']
        cut=sorted([float(r['score'])-float(r['threshold']) for r in bg],reverse=True)[4]
        chosen=[r for r in rr if float(r['score'])-float(r['threshold'])>cut]
        empirical.append(dict(experiment='COMPONENT_GAUSSIAN_6F',condition=c,cn0=cn,method=m,
                              H1=len(rr),test_H0=len(bg),Pfa=.02,detected=len(chosen),
                              handoff=sum(truth(r['localized']) for r in chosen)))
    for c,cn,ind,coh in ((1,32,17,19),(2,32,18,28),(1,36,88,97),(2,36,86,94)):
        for method,expected in (('G_ind',ind),('G_coh',coh)):
            assert next(r['handoff'] for r in empirical if r['condition']==c and r['cn0']==cn and r['method']==method)==expected
    csvout(out / 'component_empirical_Pfa_2pct.csv',empirical)
    return result


def recorded_summary(path,out,name):
    data=rows(path)
    result=[]
    for source in ("cttc","texbat"):
        for method in sorted({r["method"] for r in data if r["source"]==source}):
            group=[r for r in data if r["source"]==source and r["method"]==method]
            gates={g:max(float(r["score"]) for r in group if r["geometry"]==g and r["role"]=="calibration_H0")
                   for g in sorted({r["geometry"] for r in group})}
            bg=[r for r in group if r["role"]=="holdout_H0"]
            cut=sorted([float(r["score"])-gates[r["geometry"]] for r in bg],reverse=True)[4]
            assert len(bg)==200
            for level in (-34,-30):
                h=[r for r in group if truth(r["signal"]) and float(r["sbr_db"])==level]
                assert len(h)==100
                for point,offset in (("independent_calibration",0),("empirical_test_Pfa_2pct",cut)):
                    accepted=[r for r in h if float(r["score"])-gates[r["geometry"]]>offset]
                    fa=sum(float(r["score"])-gates[r["geometry"]]>offset for r in bg)
                    result.append(dict(experiment="RECORDED_BACKGROUND",source=source,sbr_db=level,method=method,
                                       operating_point=point,H1=len(h),test_H0=len(bg),detected=len(accepted),
                                       handoff=sum(truth(r["localized"]) for r in accepted),false_alarms=fa,
                                       Pd=len(accepted)/len(h),Ph=sum(truth(r["localized"]) for r in accepted)/len(h),Pfa=fa/len(bg)))
    csvout(out / (name+".csv"),result)
    return result


def figures(out,summary,recorded,components,report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    methods=["NCI","NCI-compensated","BASIC-quarter-ms","PIML-full","PIML-64","manual-stop","learned-stop"]
    for stratum in ("nominal","transfer"):
        fig,axes=plt.subplots(2,2,figsize=(10,7),sharex=True,sharey=True)
        for c in (1,2):
            for col,metric in enumerate(("Pd","Ph")):
                for method in methods:
                    rr=sorted([r for r in summary if r["condition"]==c and r["stratum"]==stratum
                               and r["method"]==method and r["cn0"] is not None],key=lambda r:r["cn0"])
                    axes[c-1,col].plot([r["cn0"] for r in rr],[r[metric] for r in rr],marker="o",ms=3,label=method)
                axes[c-1,col].set(title=f'{"Low" if c==1 else "High"} dynamics: {metric}',ylim=(0,1.02),xlabel="C/N0 (dB-Hz)")
                axes[c-1,col].grid(alpha=.2)
        handles,labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc="lower center",ncol=4,fontsize=8)
        fig.tight_layout(rect=[0,.09,1,1]);fig.savefig(out/f"main_{stratum}.png",dpi=180);plt.close(fig)
    for name,data,names in (("recorded_methods",recorded,["NCI","NCI-compensated","BASIC-short","Gcoh-WIN","PIML-frozen"]),
                            ("recorded_components",components,["G_ind","G_coh","P_WIN","A_WIN","S_WIN","F_WIN"])):
        fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
        for ax,source in zip(axes,("cttc","texbat")):
            available={r["method"] for r in data}
            for method in names:
                if method not in available:continue
                rr=sorted([r for r in data if r["source"]==source and r["method"]==method
                           and r["operating_point"]=="empirical_test_Pfa_2pct"],key=lambda r:r["sbr_db"])
                ax.plot([r["sbr_db"] for r in rr],[100*r["Ph"] for r in rr],marker="o",label=method)
            ax.set(title=source.upper(),xlabel="SBR (dB)",ylabel="Correct initialization (%)",ylim=(0,100));ax.grid(alpha=.2)
        axes[-1].legend(fontsize=8);fig.tight_layout();fig.savefig(out/(name+".png"),dpi=180);plt.close(fig)
    names=["PIML-full","PIML-64","manual-stop","learned-stop"]
    fig,ax=plt.subplots(figsize=(7,4));ax.bar(names,[report["methods"][m]["mean_ms"] for m in names])
    ax.set(ylabel="Recorded mean backend latency (ms)");fig.tight_layout();fig.savefig(out/"backend_latency.png",dpi=180);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"reproduced")
    parser.add_argument("--no-figures",action="store_true")
    args=parser.parse_args()
    verified=verify_files()
    args.output.mkdir(parents=True,exist_ok=True)
    summary,report=main_experiment(args.output)
    components=component_summary(ROOT/"data/component/method_events.csv",args.output)
    recorded=recorded_summary(ROOT/"data/recorded/method_events.csv",args.output,"recorded_methods")
    ablation=recorded_summary(ROOT/"data/recorded/ablation_events.csv",args.output,"recorded_ablation")
    for source,values in (("cttc",(80,78,6)),("texbat",(64,63,4))):
        r=next(r for r in recorded if r["source"]==source and r["sbr_db"]==-30
               and r["method"]=="PIML-frozen" and r["operating_point"]=="independent_calibration")
        assert (r["detected"],r["handoff"],r["false_alarms"])==values
    for source,number in (("cttc",76),("texbat",67)):
        r=next(r for r in recorded if r["source"]==source and r["sbr_db"]==-30
               and r["method"]=="PIML-frozen" and r["operating_point"]=="empirical_test_Pfa_2pct")
        assert r["handoff"]==number
    if not args.no_figures:figures(args.output,summary,recorded,ablation,report)
    write(args.output/"verification.json",dict(passed=True,archive_files_verified=verified,
          experiments=["MAIN_GAUSSIAN","COMPONENT_GAUSSIAN_6F","RECORDED_BACKGROUND"],
          main_events=report["method_events"],controller_variants=report["controller_variants"],
          figures_generated=not args.no_figures))
    print(json.dumps(dict(passed=True,files_verified=verified,main_events=report["method_events"],output=str(args.output))))


if __name__=="__main__":main()

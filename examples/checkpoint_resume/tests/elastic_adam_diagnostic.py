"""Explain failed elastic adapter coordinates using retained F32 checkpoint payloads."""
import argparse
import math
from pathlib import Path
import struct

from test_elastic_microbatch import read_json, tensor_payloads, write_json


def coordinate(tensors, name, index):
    return struct.unpack_from("<f", tensors[name][1], index * 4)[0]


def explain(root):
    result = read_json(root / "result.json")
    summary = {"status": result["status"], "failures": result["failures"], "updates": []}
    snapshots = root / "snapshots"
    for update in result["updates"]:
        t = update["step"]
        row = {"step": t, "groups": {}, "outliers": {}, "metrics_relative_error": {}}
        summary["updates"].append(row)
        base, elastic = (update["arms"][a]["metrics"] for a in ("baseline", "elastic"))
        for k in base:
            row["metrics_relative_error"][k] = abs(base[k] - elastic[k]) / abs(base[k])
        for group, tensors in update["tensors"].items():
            row["groups"][group] = {
                "max_absolute": max(v["max_abs_error"] for v in tensors.values()),
                "max_normalized": max((v["normalized_error"] or 0 for v in tensors.values())),
                "exceedances": sum(v["max_abs_error"] > v["limit"] for v in tensors.values()),
            }
        for arm in ("baseline", "elastic"):
            snap = snapshots / f"{arm}-{t}"
            state = read_json(snap / "state.json")
            cfg = state["optimizer"]["config"]
            if cfg["weight_decay"] or cfg["amsgrad"]:
                raise ValueError("Diagnostic requires zero weight decay and plain Adam moments")
            now = tensor_payloads(snap / "state.safetensors")
            prev = tensor_payloads(snap / "initial.safetensors" if t == 1 else
                                   snapshots / f"{arm}-{t - 1}" / "state.safetensors")
            for name, info in update["tensors"]["parameter"].items():
                if info["max_abs_error"] <= info["limit"]:
                    continue
                record = row["outliers"].setdefault(name, dict(info))
                idx = info["index"]
                suffix = name.split("/", 1)[1]
                m, v = (coordinate(now, f"optimizer/{moment}/{suffix}", idx) for moment in ("m", "v"))
                m_hat = m / (1 - cfg["beta1"] ** t)
                rms = math.sqrt(v / (1 - cfg["beta2"] ** t))
                predicted = -cfg["learning_rate"] * m_hat / (rms + cfg["epsilon"])
                old, new = coordinate(prev, name, idx), coordinate(now, name, idx)
                record[arm] = {"previous": old, "current": new, "observed_update": new - old,
                               "m": m, "v": v, "rms_over_epsilon": rms / cfg["epsilon"],
                               "predicted_update_from_m_v": predicted,
                               "reconstruction_abs_residual": abs(new - old - predicted)}
    write_json(root / "adam_diagnostic.json", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_directory", type=Path)
    explain(parser.parse_args().result_directory)

import json
import sys
import os
from .systems import TwoLinkArm, LotkaVolterra, LorenzSystem, VanDerPol, ZeroDynamics, LinearSystem
from pathlib import Path

def get_system(system_name):
    with open(os.path.join(str(Path(__file__).parent), "system_params.json"), 'r') as f:
        data = json.load(f)
    if system_name not in data:
        raise ValueError("System name not found in system_params.json")
    data = data[system_name]
    if data["type"] == "TwoLinkArm":
        return TwoLinkArm(data["properties"], data["params"])
    elif data["type"] == "LotkaVolterra":
        return LotkaVolterra(data["properties"], data["params"])
    elif data["type"] == "VanDerPol":
        return VanDerPol(data["properties"], data["params"])
    elif data["type"] == "LorenzSystem":
        return LorenzSystem(data["properties"], data["params"])
    elif data["type"] == "ZeroDynamics":
        return ZeroDynamics(data["properties"])
    elif data["type"] == "LinearSystem":
        return LinearSystem(data["properties"], data["params"])
    else:
        raise ValueError("System type not found in systems.json")



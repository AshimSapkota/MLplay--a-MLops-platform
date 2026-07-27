"""
promote.py

This is the ONLY sanctioned way to promote a model version in this workflow.
Nobody should call client.set_registered_model_alias(...) directly — that's
the equivalent of pushing straight to main, bypassing CI.
"""

import sys
from mlflow import MlflowClient
from registry_gate import gate_for_candidate, gate_for_champion


def promote(name: str, version: str, target_alias: str, tracking_uri="http://127.0.0.1:5000"):
    client = MlflowClient(tracking_uri=tracking_uri)

    if target_alias == "candidate":
        result = gate_for_candidate(client, name, version)
    elif target_alias == "champion":
        result = gate_for_champion(client, name, version)
    else:
        print(f"No gate defined for alias '{target_alias}' — refusing to promote blind.")
        sys.exit(1)

    print(f"\nGate check for {name} v{version} -> '{target_alias}': {result}\n")

    if not result.passed:
        print("PROMOTION BLOCKED. Fix the issues above and try again.")
        sys.exit(1)  # non-zero exit — this is what a real CI job checks

    client.set_registered_model_alias(name=name, alias=target_alias, version=version)
    print(f"Promoted {name} v{version} to '{target_alias}'.")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python promote.py <model_name> <version> <target_alias>")
        sys.exit(1)
    promote(sys.argv[1], sys.argv[2], sys.argv[3])

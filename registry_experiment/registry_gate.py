"""
registry_gate.py

Enforces registration/promotion policy for the model registry — the code form
of the "required metadata" table and naming convention from the governance doc.

Real-life framing: this is the thing that sits between "a data scientist thinks
their model is ready" and "the model actually becomes reachable by serving code."
Nobody should be able to skip it, the same way nobody should be able to skip
tests before merging code.
"""

import re
from dataclasses import dataclass, field
from mlflow import MlflowClient


# --- Policy definitions (this is the part a real team would review/version in Git) ---

NAME_PATTERN = re.compile(r"^(credit|fraud|kyc|aml|support|marketing|internal-tools)-[a-z0-9\-]+$")

REQUIRED_TAGS_FOR_CANDIDATE = [
    "intended_use",       # free text — prevents scope creep / model misuse
    "known_limitations",  # free text — regulatory documentation requirement
]

REQUIRED_TAGS_FOR_CHAMPION = REQUIRED_TAGS_FOR_CANDIDATE + [
    "approved_by",        # model risk officer sign-off identity
    "approval_date",
]

MINIMUM_METRICS_FOR_CANDIDATE = {
    "accuracy_score": 0.75,   # arbitrary example thresholds — a real team would
    "f1_score": 0.70,          # calibrate these per model type, not copy blindly
}


@dataclass
class GateResult:
    passed: bool
    errors: list = field(default_factory=list)

    def __str__(self):
        if self.passed:
            return "PASSED"
        return "FAILED:\n  - " + "\n  - ".join(self.errors)


def check_naming(model_name: str) -> GateResult:
    """Enforce <domain>-<use_case>-<algorithm_family> naming convention."""
    if NAME_PATTERN.match(model_name):
        return GateResult(passed=True)
    return GateResult(
        passed=False,
        errors=[
            f"Model name '{model_name}' doesn't match required convention "
            f"'<domain>-<use_case>-<algo>' where domain is one of: "
            f"credit, fraud, kyc, aml, support, marketing, internal-tools"
        ],
    )


def check_required_tags(client: MlflowClient, name: str, version: str, required_tags: list) -> GateResult:
    """Confirm required metadata tags are present on this model version."""
    mv = client.get_model_version(name=name, version=version)
    tags = mv.tags or {}
    missing = [t for t in required_tags if t not in tags or not tags[t].strip()]
    if not missing:
        return GateResult(passed=True)
    return GateResult(
        passed=False,
        errors=[f"Missing required tag: '{t}'" for t in missing],
    )


def check_metric_thresholds(client: MlflowClient, name: str, version: str, thresholds: dict) -> GateResult:
    """Pull the metrics from the run that produced this version and check thresholds."""
    mv = client.get_model_version(name=name, version=version)
    run = client.get_run(mv.run_id)
    metrics = run.data.metrics

    errors = []
    for metric_name, min_value in thresholds.items():
        actual = metrics.get(metric_name)
        if actual is None:
            errors.append(f"Metric '{metric_name}' not found on run {mv.run_id} — was it logged?")
        elif actual < min_value:
            errors.append(f"Metric '{metric_name}'={actual:.3f} is below required minimum {min_value}")

    return GateResult(passed=not errors, errors=errors)


def gate_for_candidate(client: MlflowClient, name: str, version: str) -> GateResult:
    """All checks required before a version may become 'candidate'."""
    results = [
        check_naming(name),
        check_required_tags(client, name, version, REQUIRED_TAGS_FOR_CANDIDATE),
        check_metric_thresholds(client, name, version, MINIMUM_METRICS_FOR_CANDIDATE),
    ]
    all_errors = [e for r in results for e in r.errors]
    return GateResult(passed=not all_errors, errors=all_errors)


def gate_for_champion(client: MlflowClient, name: str, version: str) -> GateResult:
    """Stricter checks required before a version may become 'champion'."""
    results = [
        check_naming(name),
        check_required_tags(client, name, version, REQUIRED_TAGS_FOR_CHAMPION),
        check_metric_thresholds(client, name, version, MINIMUM_METRICS_FOR_CANDIDATE),
    ]
    all_errors = [e for r in results for e in r.errors]
    return GateResult(passed=not all_errors, errors=all_errors)

"""Load runtime profiles from YAML."""

from __future__ import annotations

import yaml

from bgrag.config import Settings
from bgrag.profiles.models import RuntimeProfile


def load_profile(profile_name: str, settings: Settings) -> RuntimeProfile:
    profile_path = settings.resolved_profiles_dir / f"{profile_name}.yaml"
    if not profile_path.exists():
        raise FileNotFoundError(f"Profile not found: {profile_path}")
    with profile_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return RuntimeProfile.model_validate(data)


def list_profiles(settings: Settings) -> list[str]:
    base = settings.resolved_profiles_dir
    return sorted(path.stem for path in base.glob("*.yaml"))

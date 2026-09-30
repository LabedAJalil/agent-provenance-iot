import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    root: Path
    dataset: Path
    provenance_phase1: Path
    provenance_phase2: Path
    attack_info: Path
    pcap_phase1_dir: Path
    pcap_phase2_dir: Path
    outputs: Path


@dataclass(frozen=True)
class RuntimeConfig:
    openai_api_key: str
    openai_model: str
    openai_temperature: float
    llm_max_output_tokens: int


def default_paths(root: Path) -> Paths:
    dataset = root / "Dataset"
    return Paths(
        root=root,
        dataset=dataset,
        provenance_phase1=dataset / "Provenance_Logs" / "Phase1_Provenance.csv",
        provenance_phase2=dataset / "Provenance_Logs" / "Phase2_Provenance.csv",
        attack_info=dataset / "Supplementary Material" / "attack_info.csv",
        pcap_phase1_dir=dataset / "Network_Traffic" / "Phase1",
        pcap_phase2_dir=dataset / "Network_Traffic" / "Phase2",
        outputs=root / "experiments" / "outputs",
    )


def runtime_config_path(root: Path) -> Path:
    return root / "experiments" / "runtime_config.json"


def load_dotenv(root: Path) -> None:
    """Load KEY=VALUE lines from experiments/.env (then <root>/.env) into os.environ.
    Existing environment variables win; the file never overrides them."""
    for env_path in (root / "experiments" / ".env", root / ".env"):
        if not env_path.exists():
            continue
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value


def load_runtime_config(root: Path) -> RuntimeConfig:
    """Precedence: environment variables (including .env) > runtime_config.json > defaults."""
    load_dotenv(root)
    cfg_path = runtime_config_path(root)
    file_cfg = {}
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            file_cfg = json.load(f)

    def pick(env_key: str, file_key: str, default):
        env_val = os.getenv(env_key, "").strip()
        if env_val:
            return env_val
        return file_cfg.get(file_key, default)

    openai_api_key = str(pick("OPENAI_API_KEY", "openai_api_key", "")).strip()
    openai_model = str(pick("OPENAI_MODEL", "openai_model", "gpt-4o-mini")).strip()
    openai_temperature = float(pick("OPENAI_TEMPERATURE", "openai_temperature", 0.1))
    llm_max_output_tokens = int(pick("LLM_MAX_OUTPUT_TOKENS", "llm_max_output_tokens", 200))

    return RuntimeConfig(
        openai_api_key=openai_api_key,
        openai_model=openai_model,
        openai_temperature=openai_temperature,
        llm_max_output_tokens=llm_max_output_tokens,
    )

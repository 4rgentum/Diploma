"""UNSW-NB15 / CICIDS2017 feature schemas — single source of truth.

UNSW-NB15 (Moustafa & Slay, 2015) — 44 features grouped into five families
(basic flow, content, time, additional, ct_*-counters). See Stage 1 §1.4 and
Stage 2 §2.1 of the thesis for the rationale of the chosen grouping.
"""

from __future__ import annotations

from typing import Final

# ---------------------------------------------------------------------------
# UNSW-NB15
# ---------------------------------------------------------------------------

UNSW_LABEL_COL: Final[str] = "label"
UNSW_ATTACK_CAT_COL: Final[str] = "attack_cat"

UNSW_ATTACK_CATEGORIES: Final[tuple[str, ...]] = (
    "Normal",
    "Analysis",
    "Backdoor",
    "DoS",
    "Exploits",
    "Fuzzers",
    "Generic",
    "Reconnaissance",
    "Shellcode",
    "Worms",
)

# Basic flow group
UNSW_FEATURES_BASIC: Final[tuple[str, ...]] = (
    "dur",
    "proto",
    "service",
    "state",
    "spkts",
    "dpkts",
    "sbytes",
    "dbytes",
    "rate",
    "sttl",
    "dttl",
    "sload",
    "dload",
    "sloss",
    "dloss",
)

# Content group
UNSW_FEATURES_CONTENT: Final[tuple[str, ...]] = (
    "sinpkt",
    "dinpkt",
    "sjit",
    "djit",
    "swin",
    "stcpb",
    "dtcpb",
    "dwin",
    "tcprtt",
    "synack",
    "ackdat",
    "smean",
    "dmean",
    "trans_depth",
    "response_body_len",
)

# Connection / additional + ct_*
UNSW_FEATURES_CT: Final[tuple[str, ...]] = (
    "ct_srv_src",
    "ct_state_ttl",
    "ct_dst_ltm",
    "ct_src_dport_ltm",
    "ct_dst_sport_ltm",
    "ct_dst_src_ltm",
    "is_ftp_login",
    "ct_ftp_cmd",
    "ct_flw_http_mthd",
    "ct_src_ltm",
    "ct_srv_dst",
    "is_sm_ips_ports",
)

UNSW_FEATURES: Final[tuple[str, ...]] = (
    UNSW_FEATURES_BASIC + UNSW_FEATURES_CONTENT + UNSW_FEATURES_CT
)

# Categorical vs numeric split (drives Preprocessor encoding choice).
UNSW_CATEGORICAL: Final[tuple[str, ...]] = ("proto", "service", "state")
UNSW_BINARY: Final[tuple[str, ...]] = ("is_ftp_login", "is_sm_ips_ports")
UNSW_NUMERIC: Final[tuple[str, ...]] = tuple(
    f for f in UNSW_FEATURES if f not in UNSW_CATEGORICAL and f not in UNSW_BINARY
)

# Features with very heavy tails — log1p before scaling.
UNSW_LOG_TRANSFORM: Final[tuple[str, ...]] = (
    "dur",
    "sbytes",
    "dbytes",
    "rate",
    "sload",
    "dload",
    "sinpkt",
    "dinpkt",
    "sjit",
    "djit",
    "smean",
    "dmean",
    "response_body_len",
    "spkts",
    "dpkts",
    "stcpb",
    "dtcpb",
)


# ---------------------------------------------------------------------------
# CICIDS2017 — common-feature subset used for cross-evaluation (Stage 2 §8.10)
# ---------------------------------------------------------------------------

CICIDS_LABEL_COL: Final[str] = "Label"

# Mapping of CICIDS feature names -> UNSW-like semantic role.
# Used by load_cicids2017 to rename columns so the same Preprocessor works.
CICIDS_TO_UNSW_MAP: Final[dict[str, str]] = {
    "Flow Duration": "dur",
    "Protocol": "proto_num",
    "Destination Port": "service_port",
    "Total Fwd Packets": "spkts",
    "Total Backward Packets": "dpkts",
    "Total Length of Fwd Packets": "sbytes",
    "Total Length of Bwd Packets": "dbytes",
    "Fwd Packets/s": "sload",
    "Bwd Packets/s": "dload",
    "Fwd IAT Mean": "sinpkt",
    "Bwd IAT Mean": "dinpkt",
    "Fwd IAT Std": "sjit",
    "Bwd IAT Std": "djit",
}

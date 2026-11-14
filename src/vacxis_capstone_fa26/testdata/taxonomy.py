"""Query-type taxonomy: QUERY_TYPE -> supporting chunk category + evidence groups."""
from __future__ import annotations

QUERY_TYPE_DEFS: list[dict] = [
    {
        "type": "INDICATION",
        "category": "chỉ định",
        "evidence_groups": ["INDICATIONS"],
    },
    {
        "type": "DOSAGE",
        "category": "liều lượng và cách dùng",
        "evidence_groups": ["NUMBER_OF_DOSES", "DOSING_INTERVAL"],
    },
    {
        "type": "ADMINISTRATION_ROUTE",
        "category": "liều lượng và cách dùng",
        "evidence_groups": ["ADMINISTRATION_ROUTE"],
    },
    {
        "type": "CONTRAINDICATION",
        "category": "chống chỉ định",
        "evidence_groups": ["CONTRAINDICATIONS"],
    },
    {
        "type": "ADVERSE_EFFECT",
        "category": "tác dụng không mong muốn",
        "evidence_groups": ["ADVERSE_EFFECTS"],
    },
    {
        "type": "COMPOSITION",
        "category": "thành phần",
        "evidence_groups": ["COMPOSITION"],
    },
    {
        "type": "STORAGE",
        "category": "bảo quản",
        "evidence_groups": ["STORAGE"],
    },
    {
        "type": "PRECAUTION",
        "category": "cảnh báo và thận trọng",
        "evidence_groups": ["PRECAUTIONS"],
    },
    {
        "type": "INTERACTION",
        "category": "tương tác thuốc",
        "evidence_groups": ["INTERACTIONS"],
    },
    {
        "type": "SPECIAL_POPULATION",
        "category": "thai kỳ và cho con bú",
        "evidence_groups": ["SPECIAL_POPULATIONS"],
    },
]

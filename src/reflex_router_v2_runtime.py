"""Pure execution helpers for Reflex Router v2 gateway integration."""
from __future__ import annotations

from datetime import datetime
import re
from typing import Any, Iterable


_ACTION_PATTERNS = {
    "hug": re.compile(r"抱|依偎|怀里|搂"),
    "closeness": re.compile(r"抱|靠近|贴近|依偎|怀里"),
    "kiss": re.compile(r"亲|啵啵|吻"),
    "tender_intimacy": re.compile(r"亲|温柔|轻柔|贴贴|啵啵"),
    "companionship": re.compile(r"陪|在这|在场|留在|守着"),
    "presence": re.compile(r"陪|在这|还在|不走|留在"),
    "longing": re.compile(r"想你|想念|好久不见|重逢"),
    "reunion": re.compile(r"重逢|回来|再见|好久不见|想你"),
    "affectionate_observation": re.compile(r"日语|语言|情话|可爱|好听"),
    "request_presence": re.compile(r"陪|在吗|还在|聊聊天|留在"),
    "anxiety": re.compile(r"焦虑|紧张|恐慌|手抖"),
    "nausea": re.compile(r"想吐|恶心|反胃"),
    "fear": re.compile(r"害怕|怕|恐惧"),
    "partner_help": re.compile(r"陪|抱|老公|接住|在这"),
    "work": re.compile(r"工作|上班|调酒师|夜班|下班"),
    "stay_up_late": re.compile(r"熬夜|夜班|凌晨|深夜|晚下班"),
    "prior_topic": re.compile(r"上次|之前|讨论|说过|舵机"),
    "purchase_completed": re.compile(r"买了|购买|到货|下单"),
    "design_philosophy_rebuild": re.compile(r"设计哲学|重建|那天|决定"),
    "contact_change": re.compile(r"联系|没联系|来信|写信|消息|好久"),
    "current_status": re.compile(r"现在|当前|最近|状态|是否|回复"),
    "completion": re.compile(r"已回复|回复了|没回复|未回复|尚未回复|完成|做完"),
    "named_subject_relation": re.compile(r"关系|朋友|伴侣|认识|想念|不在"),
    "legacy_authorship": re.compile(r"写过|留下|做过|意图路由|maintainer"),
    "recurring_life_event": re.compile(r"又|再次|坏了|失灵|出问题|指纹锁"),
    "specific_desire": re.compile(r"夜宵|想吃|重庆小面"),
}

_TEST_AUDIT_RE = re.compile(r"(反射弧测试|召回测试|路由测试|验收结果|recall_trace|审计记录)", re.I)
_DIRECT_STATUS_RE = re.compile(r"(已回复|回复了|没回复|未回复|尚未回复|已经回信|还没回信)", re.I)
_STATUS_OBJECT_RE = re.compile(r"(信|邮件|消息|回信)", re.I)


def candidate_body(row: dict[str, Any]) -> str:
    return " ".join(str(row.get(key) or "") for key in (
        "tag", "snippet", "text", "matched_text", "shadow_text", "context"
    ))


def _entity_variants(route: dict[str, Any]) -> list[list[str]]:
    groups: list[list[str]] = []
    for entity in (route.get("slots") or {}).get("entities") or []:
        values = [
            str(entity.get("surface") or "").strip(),
            str(entity.get("canonical") or "").strip(),
            *[str(value or "").strip() for value in entity.get("aliases") or []],
        ]
        values = list(dict.fromkeys(value for value in values if len(value) >= 2))
        if values:
            groups.append(values)
    if not groups:
        for value in (route.get("evidence_policy") or {}).get("required_entities") or []:
            value = str(value or "").strip()
            if value:
                groups.append([value])
    return groups


def _required_actions(route: dict[str, Any]) -> list[str]:
    return [
        str(action) for action in (route.get("evidence_policy") or {}).get("required_actions") or []
        if str(action) in _ACTION_PATTERNS
    ]


def _timestamp(row: dict[str, Any]) -> datetime | None:
    raw = str(row.get("timestamp") or row.get("time") or "").replace("Z", "+00:00")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw)
    except Exception:
        return None


def is_latest_direct_current_fact(row: dict[str, Any], pool: Iterable[dict[str, Any]], route: dict[str, Any]) -> bool:
    body = candidate_body(row)
    if not _DIRECT_STATUS_RE.search(body) or not _STATUS_OBJECT_RE.search(body):
        return False
    for group in _entity_variants(route):
        if not any(value.casefold() in body.casefold() for value in group):
            return False
    row_ts = _timestamp(row)
    direct_rows = []
    for candidate in pool or []:
        cbody = candidate_body(candidate)
        if not _DIRECT_STATUS_RE.search(cbody) or not _STATUS_OBJECT_RE.search(cbody):
            continue
        if any(not any(value.casefold() in cbody.casefold() for value in group)
               for group in _entity_variants(route)):
            continue
        direct_rows.append(candidate)
    timestamps = [value for value in (_timestamp(candidate) for candidate in direct_rows) if value is not None]
    if not timestamps:
        return False
    return row_ts is not None and row_ts == max(timestamps)


def candidate_passes_injection(
    route: dict[str, Any], row: dict[str, Any], *, pool: Iterable[dict[str, Any]] = (),
    original_query: str = "",
) -> tuple[bool, str]:
    """Deterministic fifth-layer evidence/usefulness checks for main Anchor rows."""
    if not row or row.get("source") == "cold_store":
        return True, "cold_uses_existing_literal_contract"
    body = candidate_body(row)
    folded = body.casefold()
    if not folded:
        return False, "empty_candidate_body"

    if _TEST_AUDIT_RE.search(body) and not re.search(r"(测试|验收|审计)", original_query or "", re.I):
        return False, "test_self_reference"

    for group in _entity_variants(route):
        if not any(value.casefold() in folded for value in group):
            return False, "required_entity_missing"

    required_actions = _required_actions(route)
    if required_actions:
        matched = sum(1 for action in required_actions if _ACTION_PATTERNS[action].search(body))
        minimum = 2 if len(required_actions) >= 2 else 1
        if matched < minimum:
            return False, "required_action_missing"

    if route.get("answer_mode") == "current_fact_direct_only":
        if not is_latest_direct_current_fact(row, pool, route):
            return False, "not_latest_direct_current_fact"

    return True, "accepted"


def choose_association_seeds(
    route: dict[str, Any], selected_main: Iterable[dict[str, Any]],
    canonical_person_rows: Iterable[dict[str, Any]] = (),
) -> tuple[list[dict[str, Any]], str]:
    """Return legal hidden/selected Anchor seeds according to the approved D/E contract."""
    association = (route.get("lanes") or {}).get("association") or {}
    if not association.get("allowed"):
        return [], "none"
    selected_anchor = [
        row for row in selected_main or []
        if row.get("source") != "cold_store" and row.get("memory_id")
    ]
    mode = association.get("seed_mode") or "off"
    if selected_anchor:
        return selected_anchor[:2], "selected_anchor"
    if mode != "selected_anchor_or_canonical_person":
        return [], "none"
    if route.get("answer_mode") == "current_fact_direct_only":
        return [], "none"
    person_rows = [row for row in canonical_person_rows or [] if row.get("memory_id")]
    return person_rows[:2], "canonical_person" if person_rows else "none"


def association_metadata(route: dict[str, Any]) -> dict[str, Any]:
    answer_mode = route.get("answer_mode") or "normal"
    cannot_answer = []
    if answer_mode == "identity_person_card":
        cannot_answer.append("identity")
    if answer_mode == "current_fact_direct_only":
        cannot_answer.append("current_status")
    return {
        "lane": "association",
        "source": "anchor_memory",
        "evidence_role": "background_only",
        "cannot_answer": cannot_answer,
    }


def main_metadata(row: dict[str, Any], route: dict[str, Any]) -> dict[str, Any]:
    if row.get("source") == "cold_store":
        return {
            "lane": "main", "source": "cold_store",
            "evidence_role": "raw_dialogue", "cannot_answer": ["current_status"],
        }
    role = "direct_current_fact" if route.get("answer_mode") == "current_fact_direct_only" else "direct_memory"
    return {
        "lane": "main", "source": "anchor_memory",
        "evidence_role": role, "cannot_answer": [],
    }


def answer_contract(route: dict[str, Any], selected_main: Iterable[dict[str, Any]]) -> dict[str, Any]:
    mode = route.get("answer_mode") or "normal"
    if mode == "current_fact_direct_only":
        found = any(
            row.get("source") != "cold_store" for row in (selected_main or [])
        )
        return {"requires": "direct_current_fact", "direct_evidence_found": found}
    if mode == "identity_person_card":
        return {"requires": "person_card_identity", "direct_evidence_found": False}
    return {"requires": "none", "direct_evidence_found": bool(list(selected_main or []))}

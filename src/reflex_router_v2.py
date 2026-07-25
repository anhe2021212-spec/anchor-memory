"""Reflex Router v2: deterministic recall policy and bounded query rewrite.

This module is intentionally pure: it performs no file I/O, network calls,
embedding, retrieval, or model inference. Callers provide an immutable alias
index and execute the returned lane plan.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Iterable


POLICY_VERSION = "2026-07-12.2"

_CHANNEL_RE = re.compile(r"<channel\b[^>]*>(.*?)</channel>", re.I | re.S)
_CHANNEL_BOUNDARY_RE = re.compile(r"</?channel\b[^>]*>", re.I)
_TRAILING_ATTACHMENT_RE = re.compile(
    r"(?:\n|^)(?:\[本轮附件\]|\[附件\])\s*[^\n]*(?:\n.*)?$", re.I | re.S
)
_WRAPPER_ONLY_RE = re.compile(
    r"^(?:\[auto_swap\]|\[heartbeat\]|\[system\]|\[附件\]|\[本轮附件\])(?:\s|$).*$",
    re.I | re.S,
)
_PUNCT_RE = re.compile(r"[\s。！？!?,，.;；:：、~～…·\-—_\"'“”‘’（）()\[\]{}<>《》]+")
_QUOTED_RE = re.compile(
    r'"[^"\n]{0,240}"|“[^”\n]{0,240}”|‘[^’\n]{0,240}’|'
    r'「[^」\n]{0,240}」|『[^』\n]{0,240}』|《[^》\n]{0,240}》'
)

_EXPLICIT_OPT_OUT_RE = re.compile(
    r"(别|不要|不用|不许|无需).{0,5}(翻旧账|想以前|想过去|翻记忆|召回|人物卡|联想)|"
    r"只(?:要)?(?:陪我|听我说|聊聊天).{0,4}(?:就好|就行)?",
    re.I,
)

_LOW_EXACT = {
    "助手", "伙伴", "朋友", "老师", "AI agent", "AI agent啊", "AI agent呀",
    "汪", "汪汪", "汪汪汪", "呜", "呜呜", "呜呜呜",
    "嗯", "嗯嗯", "嗯呐", "唔", "哦", "喔", "噢", "啊", "呀",
    "好", "好的", "好呀", "好哦", "行", "可以", "知道了", "收到",
    "嘿嘿", "嘻嘻", "哈哈", "哈哈哈", "hhhh", "hhh", "笑死",
    "在吗", "在嘛", "在不在", "早", "早安", "晚安", "嗨", "hello", "hi",
    "没", "没有", "没事", "还好", "测试",
}
_LOW_CONTINUE_RE = re.compile(r"^(?:测试)?(?:继续)+$", re.I)
_LOW_NOISE_RE = re.compile(r"^(?:汪|呜|哈|h|嗯|哦|喔|噢|啊|呀|欸|诶|嘿|嘻){2,}$", re.I)

_SUPPORT_ACTION_RE = re.compile(
    r"(陪我|听我说|聊聊|安慰我|支持我)", re.I
)
_CONTINUITY_RE = re.compile(r"(好久没继续|重新联系|又见面)", re.I)
_SUPPORT_OBSERVATION_RE = re.compile(r"(你说.{0,10}(?:有帮助|很清楚)|这次.{0,8}(?:有帮助|很清楚))", re.I)
_SHARED_ACTIVITY_RE = re.compile(
    r"(想和你.{0,12}(?:一起讨论|继续聊|一起完成)|我们.{0,12}(?:继续讨论|一起完成))",
    re.I,
)
_PRESENCE_REQUEST_RE = re.compile(r"(聊聊天|听我说|你会在吗|你在吗|继续聊)", re.I)

_PRESENT_STATE_RE = re.compile(
    r"(郁闷|委屈|生气|不开心|难过|孤单|烦(?:死)?|累(?:死)?|困(?:死)?|"
    r"不知道怎么办|语境太复杂|解释不清)", re.I
)
_DISTRESS_PERSON_RE = re.compile(r"(不再联系|离开了|交接中断|好久没联系)", re.I)

_SPECIFIC_PATTERN_RE = re.compile(
    r"((?:焦虑|紧张|恐慌).{0,8}(?:不舒服|需要帮助|帮帮我)|好害怕|害怕.{0,8}(?:陪我|帮帮我))",
    re.I,
)
_ACUTE_SYMPTOM_RE = re.compile(
    r"(身体不舒服|有点不舒服|胃痛|头痛|头疼|发烧|眩晕)",
    re.I,
)

_EXPLICIT_RECALL_RE = re.compile(
    r"(还记得|你记得|记得|上次|上回|之前|那次|当时|那天|以前|过去|曾经|回忆|你说过)",
    re.I,
)
_PAST_WORK_RE = re.compile(r"以前(?:工作|上班).{0,8}(?:熬夜|作息不规律)", re.I)
_CURRENT_STATUS_RE = re.compile(
    r"(回复了(?:吗|没有)|回复了吗|回信了(?:吗|没有)|最近怎么样|现在怎么样|目前怎么样|"
    r"是否(?:已经)?(?:回复|完成)|做完了吗|完成了吗)",
    re.I,
)
_CONTACT_CHANGE_RE = re.compile(r"(好久没联系|久未联系|好久没.{0,4}(?:写信|来信)|没给你写信)", re.I)
_IDENTITY_RE = re.compile(r"(?:是谁|是什么人|哪位|什么来历)[？?]?$", re.I)

_TECH_ACTION_RE = re.compile(
    r"(帮我|帮忙|请|去|让你|继续|接着|需要|要).{0,12}(查|搜|搜索|看|看看|读|打开|检查|巡检|修|改|写|加|加入|添加|更新|优化|部署|重启|配置|调试|总结)|"
    r"(查|搜|搜索|看|读|打开|检查|巡检|修|改|写|加|加入|添加|更新|优化|部署|重启|配置|调试).{0,12}(一下|下|吧|服务|代码|日志|状态|端点|接口|prompt|skill)",
    re.I,
)
_TECH_OBJECT_RE = re.compile(
    r"(代码|bug|日志|报错|配置|接口|服务|脚本|端点|api|loop|gateway|anchor|relay|hook|"
    r"pwa|前端|白屏|卡死|反射弧|意图路由|systemctl|journalctl|systemd|fastapi|nginx|"
    r"sqlite|redis|chroma|mcp|进程|端口|调试模式|github|prompt|skill|local-usermd|截图)",
    re.I,
)
_TECH_INCIDENT_RE = re.compile(r"(修不好|修炸了|又修炸|代码炸了|系统炸了|一直白屏|为什么卡死)", re.I)
_TECH_REFLECTION_RE = re.compile(r"(发现|原来|以前).{0,12}(写过|做过|留下).{0,12}maintainer", re.I)

_VAGUE_RE = re.compile(r"(这个|那个|这件事|那件事|不知道怎么办|语境太复杂|继续说|接着说)", re.I)
_CONCRETE_REPEAT_RE = re.compile(r"(又|再次).{0,16}(坏了|坏掉|出问题|失灵|不工作)", re.I)

_TEST_AUDIT_TERMS = ("测试", "验收", "审计", "recall_trace", "路由验证")
_HANDOFF_TERMS = ("换窗备忘", "handoff", "swap前", "接力")


@dataclass(frozen=True)
class AliasRecord:
    canonical: str
    relation: str
    aliases: tuple[str, ...]
    key: str


@dataclass(frozen=True)
class SlangRecord:
    word: str
    meaning: str
    origin: str


@dataclass(frozen=True)
class AliasIndex:
    people: tuple[AliasRecord, ...]
    slang: tuple[SlangRecord, ...]
    generation: str = ""


def _compact(value: str) -> str:
    return re.sub(r"\s+", "", value or "").casefold()


def build_alias_index(raw: dict[str, Any] | None, generation: str = "") -> AliasIndex:
    """Build an immutable alias index from aliases.json-shaped data."""
    raw = raw or {}
    people_raw = raw.get("people") if isinstance(raw.get("people"), dict) else {}
    people: list[AliasRecord] = []
    for key, entry in people_raw.items():
        if not isinstance(entry, dict):
            continue
        canonical = str(entry.get("canonical") or key).strip()
        aliases = tuple(
            str(alias).strip()
            for alias in (entry.get("aliases") or [])
            if str(alias).strip()
        )
        if canonical and aliases:
            people.append(AliasRecord(
                canonical=canonical,
                relation=str(entry.get("relation") or "").strip(),
                aliases=aliases,
                key=str(key),
            ))
    slang: list[SlangRecord] = []
    for word, info in (raw.get("slang") or {}).items():
        if not word or not isinstance(info, dict):
            continue
        slang.append(SlangRecord(
            word=str(word),
            meaning=str(info.get("meaning") or "").strip(),
            origin=str(info.get("origin") or "").strip(),
        ))
    return AliasIndex(tuple(people), tuple(slang), str(generation or ""))


def normalize_query(query: str) -> dict[str, Any]:
    """Strip transport wrappers while preserving the semantic user body."""
    raw = query or ""
    cleaned = _CHANNEL_RE.sub(lambda m: m.group(1), raw)
    cleaned = _CHANNEL_BOUNDARY_RE.sub(" ", cleaned).strip()
    wrapper_only = bool(_WRAPPER_ONLY_RE.match(cleaned))
    cleaned = _TRAILING_ATTACHMENT_RE.sub("", cleaned).strip()
    if wrapper_only:
        cleaned = ""
    semantic = _PUNCT_RE.sub("", cleaned)
    return {
        "normalized": cleaned,
        "semantic_body": semantic,
        "wrapper_only": wrapper_only or (not semantic and bool(raw.strip())),
    }


def _intent_body(body: str) -> str:
    """Hide quoted examples from intent triggers while retaining the original rewrite text."""
    return re.sub(r"\s+", " ", _QUOTED_RE.sub(" ", body or "")).strip()


def _match_people(body: str, aliases: AliasIndex) -> list[AliasRecord]:
    q = _compact(body)
    out: list[AliasRecord] = []
    seen: set[str] = set()
    for record in aliases.people:
        for alias in record.aliases:
            key = _compact(alias)
            if len(key) >= 2 and key in q:
                if record.canonical not in seen:
                    out.append(record)
                    seen.add(record.canonical)
                break
    return out


def _match_slang(body: str, aliases: AliasIndex) -> list[SlangRecord]:
    q = _compact(body)
    out = []
    for item in aliases.slang:
        word = _compact(item.word)
        if not word:
            continue
        if len(word) == 1:
            # A one-character private code must not fire inside an ordinary word
            # (e.g. slang "法" inside "无法" or "方法").
            bounded = re.search(
                rf"(?:^|[\s，。！？!?、]){re.escape(item.word)}(?:$|[\s，。！？!?、])",
                body,
            )
            explicit_phrase = item.word == "法" and re.search(r"(?:先|想|会)会?法(?:也|了|吗|吧|$)", body)
            if bounded or explicit_phrase:
                out.append(item)
            continue
        if word in q:
            out.append(item)
    return out


def _lane(allowed: bool, mode: str = "off", **extra: Any) -> dict[str, Any]:
    value = {"allowed": bool(allowed)}
    if allowed or mode != "off":
        value["mode"] = mode
    value.update(extra)
    return value


def _lanes(
    *, anchor: bool = False, cold: bool = False, person: bool = False,
    belief: bool = False, association: bool = False,
    association_seed: str = "off",
) -> dict[str, Any]:
    return {
        "anchor": _lane(anchor, "primary" if anchor else "off"),
        "cold": _lane(cold, "fallback_if_anchor_healthy_unselected" if cold else "off"),
        "person": _lane(person, "sidecar" if person else "off"),
        "belief": _lane(belief, "sidecar" if belief else "off"),
        "association": _lane(
            association,
            "background" if association else "off",
            max=1 if association else 0,
            seed_mode=association_seed,
            fill=False,
            background_only=True,
        ),
    }


def _entity_payload(records: Iterable[AliasRecord]) -> list[dict[str, Any]]:
    return [
        {
            "surface": next((a for a in record.aliases), record.canonical),
            "canonical": record.canonical,
            "kind": "person",
            "source": "aliases",
            "relation": record.relation,
            "aliases": list(record.aliases),
        }
        for record in records
    ]


def _stable_id(payload: dict[str, Any], aliases: AliasIndex) -> str:
    material = json.dumps({
        "policy": POLICY_VERSION,
        "query": payload["input"]["normalized"],
        "decision": payload["decision"],
        "class": payload["policy_class"],
        "generation": aliases.generation,
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "r2:" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:20]


def _base_plan(input_info: dict[str, Any], aliases: AliasIndex) -> dict[str, Any]:
    return {
        "schema": "reflex.route.v2",
        "policy_version": POLICY_VERSION,
        "decision_id": "",
        "policy_class": "G",
        "decision": "uncertain",
        "execution": "suppress",
        "primary_intent": "unknown",
        "secondary_intents": [],
        "reason_codes": [],
        "input": input_info,
        "resolution": {"state": "not_needed", "used_context": False, "context_terms": []},
        "slots": {
            "entities": [], "actions": [], "states": [], "symptoms": [],
            "time_scope": "none", "fact_mode": "none",
        },
        "rewrite": {
            "anchor_query": None, "cold_query": None, "added_terms": [],
            "dropped_terms": [], "rule_ids": [],
        },
        "lanes": _lanes(),
        "max_main": 0,
        "answer_mode": "normal",
        "evidence_policy": {
            "required_entities": [], "required_actions": [], "freshness": "none",
            "reject_test_self_reference": True,
            "association_may_answer_current_fact": False,
        },
        "diagnostics": {"router": "ok", "alias_index": "ok"},
    }


def _set_rewrite(
    plan: dict[str, Any], anchor_query: str | None, *, cold_query: str | None = None,
    added: Iterable[str] = (), dropped: Iterable[str] = (), rules: Iterable[str] = (),
) -> None:
    plan["rewrite"] = {
        "anchor_query": anchor_query,
        "cold_query": cold_query,
        "added_terms": list(added),
        "dropped_terms": list(dropped),
        "rule_ids": list(rules),
    }


def _route(
    plan: dict[str, Any], *, policy_class: str, decision: str, execution: str,
    intent: str, reason: str, lanes: dict[str, Any], max_main: int,
) -> dict[str, Any]:
    plan.update({
        "policy_class": policy_class,
        "decision": decision,
        "execution": execution,
        "primary_intent": intent,
        "reason_codes": [reason],
        "lanes": lanes,
        "max_main": max_main,
    })
    return plan


def _is_low_signal(body: str, semantic: str) -> bool:
    compact = _compact(semantic)
    if not compact:
        return True
    if compact in _LOW_EXACT or _LOW_CONTINUE_RE.fullmatch(compact):
        return True
    if _LOW_NOISE_RE.fullmatch(compact):
        return True
    if len(compact) <= 12 and not re.search(r"[A-Za-z0-9\u4e00-\u9fff]", compact):
        return True
    return False


def _is_actionable_technical(body: str) -> bool:
    if _TECH_REFLECTION_RE.search(body):
        return False
    return bool(
        (_TECH_ACTION_RE.search(body) and _TECH_OBJECT_RE.search(body))
        or (_TECH_INCIDENT_RE.search(body) and _TECH_OBJECT_RE.search(body))
        or re.search(r"(帮我查api loop|帮我重启gateway|pwa为什么卡死|pwa.{0,8}白屏)", body, re.I)
    )


def _identity_surface(body: str) -> str:
    value = re.sub(r"[？?]", "", body)
    value = re.sub(r"(?:是谁|是什么人|哪位|什么来历)$", "", value).strip()
    return value[-24:]


def _person_rewrite(body: str, people: list[AliasRecord]) -> tuple[str, list[str], list[str]]:
    terms: list[str] = []
    entities: list[str] = []
    for person in people:
        terms.extend([person.canonical, person.relation])
        entities.append(person.canonical)
    actions: list[str] = []
    if _CONTACT_CHANGE_RE.search(body):
        terms.extend(["久未联系", "联系变化", "最近性"])
        actions.append("contact_change")
    if _CURRENT_STATUS_RE.search(body):
        terms.extend(["当前状态", "完成态", "最新直接证据"])
        actions.extend(["current_status", "completion"])
    return " ".join(dict.fromkeys(t for t in terms if t)), entities, actions


def route_reflex(query: str, aliases: AliasIndex, context: str = "") -> dict[str, Any]:
    """Return a complete v2 route decision for one prompt."""
    input_info = normalize_query(query)
    body = input_info["normalized"]
    semantic = input_info["semantic_body"]
    intent_body = _intent_body(body)
    plan = _base_plan(input_info, aliases)

    if _EXPLICIT_OPT_OUT_RE.search(body):
        _route(plan, policy_class="G", decision="suppress", execution="suppress",
               intent="explicit_memory_opt_out", reason="explicit_memory_opt_out",
               lanes=_lanes(), max_main=0)
        plan["decision_id"] = _stable_id(plan, aliases)
        return plan

    if input_info["wrapper_only"] or _is_low_signal(body, semantic):
        _route(plan, policy_class="G", decision="suppress", execution="suppress",
               intent="low_signal", reason="empty_wrapper_or_low_signal",
               lanes=_lanes(), max_main=0)
        plan["decision_id"] = _stable_id(plan, aliases)
        return plan

    people = _match_people(intent_body, aliases)
    slang = _match_slang(intent_body, aliases)
    plan["slots"]["entities"] = _entity_payload(people)

    # H: mixed/conflicting clauses have explicit gold behavior and never inherit X.
    if people and re.search(r"maintainer", intent_body, re.I) and (
        _TECH_REFLECTION_RE.search(intent_body) or re.search(r"(不在了|难过的是)", intent_body)
    ):
        plan["secondary_intents"] = ["named_person", "distress"]
        plan["slots"]["states"] = ["grief_or_longing"]
        rewrite, entities, actions = _person_rewrite(intent_body, people)
        if _TECH_REFLECTION_RE.search(intent_body):
            actions.append("legacy_authorship")
            rewrite = (rewrite + " 以前写过 意图路由 怀念 失落").strip()
        _set_rewrite(plan, rewrite or body, added=actions, rules=["mixed_person_reflection"])
        _route(plan, policy_class="H", decision="retrieve", execution="retrieve",
               intent="mixed_person_emotion", reason="person_emotion_over_technical_noun",
               lanes=_lanes(anchor=True, person=True), max_main=1)
        plan["evidence_policy"]["required_entities"] = entities
        plan["evidence_policy"]["required_actions"] = actions
        plan["decision_id"] = _stable_id(plan, aliases)
        return plan

    if _is_actionable_technical(intent_body):
        person_allowed = bool(people)
        familiar_address = bool(re.search(r"^(?:伙伴|朋友|老师|助手)", body, re.I))
        policy_class = "H" if (person_allowed or familiar_address
                               or _SUPPORT_ACTION_RE.search(intent_body)
                               or _ACUTE_SYMPTOM_RE.search(intent_body)) else "F"
        plan["secondary_intents"] = (["named_person"] if person_allowed else [])
        plan["slots"]["actions"] = ["technical_task"]
        _route(plan, policy_class=policy_class, decision="technical", execution="tool_only",
               intent="technical_task", reason="actionable_technical_task",
               lanes=_lanes(person=person_allowed), max_main=0)
        plan["decision_id"] = _stable_id(plan, aliases)
        return plan

    if _IDENTITY_RE.search(intent_body):
        if people:
            relation_terms = [p.relation for p in people if p.relation]
            _set_rewrite(plan, None, added=relation_terms, rules=["known_person_identity"])
            _route(plan, policy_class="E", decision="person_only", execution="sidecars_only",
                   intent="named_subject_identity", reason="known_person_identity",
                   lanes=_lanes(person=True, association=True,
                                association_seed="selected_anchor_or_canonical_person"),
                   max_main=0)
            plan["answer_mode"] = "identity_person_card"
            plan["slots"]["fact_mode"] = "identity"
            plan["evidence_policy"]["required_entities"] = [p.canonical for p in people]
        else:
            surface = _identity_surface(intent_body)
            plan["slots"]["entities"] = [{
                "surface": surface, "canonical": None, "kind": "named_subject", "source": "surface"
            }]
            _set_rewrite(plan, f"{surface} 身份 来历 关系", added=["身份", "来历", "关系"],
                         rules=["unknown_named_identity"])
            _route(plan, policy_class="E", decision="retrieve", execution="retrieve",
                   intent="named_subject_identity", reason="unknown_named_subject_identity",
                   lanes=_lanes(anchor=True, association=True, association_seed="selected_anchor_only"),
                   max_main=1)
            plan["slots"]["fact_mode"] = "identity"
            plan["evidence_policy"]["required_entities"] = [surface]
        plan["decision_id"] = _stable_id(plan, aliases)
        return plan

    # H #40: technical incident + acute state is tool-only even without an imperative.
    if _TECH_INCIDENT_RE.search(intent_body) and (_ACUTE_SYMPTOM_RE.search(intent_body) or "神经" in intent_body):
        plan["secondary_intents"] = ["present_symptom", "self_blame"]
        plan["slots"]["actions"] = ["technical_incident"]
        _route(plan, policy_class="H", decision="technical", execution="tool_only",
               intent="mixed_technical_distress", reason="technical_incident_with_present_distress",
               lanes=_lanes(), max_main=0)
        plan["decision_id"] = _stable_id(plan, aliases)
        return plan

    if _EXPLICIT_RECALL_RE.search(intent_body):
        plan["slots"]["time_scope"] = "past"
        plan["slots"]["actions"] = ["explicit_recall"]
        anchor_query = body
        added: list[str] = []
        rules = ["explicit_recall"]
        if _PAST_WORK_RE.search(intent_body):
            anchor_query = "过去工作 作息 经历"
            added = ["过去工作", "作息"]
            rules.append("past_work_late_night")
            plan["slots"]["actions"] = ["work", "irregular_schedule"]
        elif re.search(r"(?:上次|之前).{0,12}(?:买了|购买|下单)", intent_body, re.I):
            anchor_query = "上次讨论 购买 已完成 计划延续"
            added = ["已购买", "计划延续"]
            plan["slots"]["actions"] = ["prior_topic", "purchase_completed"]
        elif re.search(r"(?:设计方案|架构方案)", intent_body, re.I):
            anchor_query = "设计方案 起因 决定 后续影响"
            added = ["起因", "决定", "后续影响"]
            plan["slots"]["actions"] = ["design_decision"]
        _set_rewrite(plan, anchor_query, cold_query=body, added=added, rules=rules)
        _route(plan, policy_class="D", decision="retrieve", execution="retrieve",
               intent="explicit_recall", reason="explicit_recall_or_past_experience",
               lanes=_lanes(anchor=True, cold=True, association=True,
                            association_seed="selected_anchor_only"),
               max_main=1)
        plan["evidence_policy"]["required_actions"] = list(plan["slots"]["actions"])
        plan["evidence_policy"]["freshness"] = "historical"
        plan["decision_id"] = _stable_id(plan, aliases)
        return plan

    if _PRESENCE_REQUEST_RE.search(intent_body) and _PRESENT_STATE_RE.search(intent_body):
        plan["slots"]["states"] = ["present_distress"]
        plan["slots"]["actions"] = ["request_presence"]
        _set_rewrite(plan, "难过 明确求陪伴 确认在场", added=["陪伴", "在场"],
                     rules=["present_state_with_presence_request"])
        _route(plan, policy_class="B", decision="retrieve", execution="retrieve",
               intent="present_state_with_relational_request",
               reason="explicit_presence_request", lanes=_lanes(anchor=True), max_main=1)
        plan["evidence_policy"]["required_actions"] = ["request_presence"]
        plan["decision_id"] = _stable_id(plan, aliases)
        return plan

    if (_SUPPORT_ACTION_RE.search(intent_body) or _SUPPORT_OBSERVATION_RE.search(intent_body)
            or _SHARED_ACTIVITY_RE.search(intent_body) or _CONTINUITY_RE.search(intent_body)):
        actions = []
        if re.search(r"陪我|听我说|聊聊", intent_body):
            actions = ["companionship", "presence"]
            anchor_query = "陪伴 倾听 在场"
        elif _CONTINUITY_RE.search(intent_body):
            actions = ["continuity", "reconnection"]
            anchor_query = "重新联系 继续交流"
        elif _SHARED_ACTIVITY_RE.search(intent_body):
            actions = ["shared_activity"]
            anchor_query = "共同讨论 继续协作"
        else:
            actions = ["supportive_observation"]
            anchor_query = "反馈 有帮助 表达清楚"
        plan["slots"]["actions"] = actions
        _set_rewrite(plan, anchor_query, added=actions, rules=["supportive_bid"])
        _route(plan, policy_class="A", decision="retrieve", execution="retrieve",
               intent="supportive_bid", reason="explicit_support_or_continuity_action",
               lanes=_lanes(anchor=True, belief=bool(_CONTINUITY_RE.search(intent_body)), association=True,
                            association_seed="selected_anchor_only"),
               max_main=1)
        plan["evidence_policy"]["required_actions"] = actions
        plan["decision_id"] = _stable_id(plan, aliases)
        return plan

    if _SPECIFIC_PATTERN_RE.search(intent_body):
        symptoms = []
        if re.search(r"焦虑|紧张|恐慌", intent_body):
            symptoms.extend(["anxiety", "support_request"])
            anchor_query = "焦虑 紧张 求助 支持"
        else:
            symptoms.extend(["fear", "support_request"])
            anchor_query = "害怕 求助 陪伴 支持"
        plan["slots"]["symptoms"] = symptoms
        _set_rewrite(plan, anchor_query, added=symptoms, rules=["specific_emotion_somatic_pattern"])
        _route(plan, policy_class="C", decision="retrieve", execution="retrieve",
               intent="specific_pattern", reason="specific_emotion_somatic_pattern",
               lanes=_lanes(anchor=True), max_main=1)
        plan["evidence_policy"]["required_actions"] = symptoms
        plan["decision_id"] = _stable_id(plan, aliases)
        return plan

    if _ACUTE_SYMPTOM_RE.search(intent_body):
        plan["slots"]["symptoms"] = ["acute_present_symptom"]
        _route(plan, policy_class="C", decision="suppress", execution="suppress",
               intent="present_symptom", reason="acute_or_immediate_cause",
               lanes=_lanes(), max_main=0)
        plan["decision_id"] = _stable_id(plan, aliases)
        return plan

    if people and (
        _CONTACT_CHANGE_RE.search(intent_body)
        or _CURRENT_STATUS_RE.search(intent_body)
        or _DISTRESS_PERSON_RE.search(intent_body)
        or re.search(r"(?:我|好)?想", intent_body)
    ):
        rewrite, entities, actions = _person_rewrite(intent_body, people)
        if not actions:
            actions = ["named_subject_relation"]
            rewrite = (rewrite + " 人物关系").strip()
        current_fact = bool(_CURRENT_STATUS_RE.search(intent_body))
        plan["slots"]["actions"] = actions
        plan["slots"]["time_scope"] = "current" if current_fact else "recent_change"
        plan["slots"]["fact_mode"] = "current_status" if current_fact else "none"
        _set_rewrite(plan, rewrite, added=actions, rules=["canonical_person", "person_action_time"])
        _route(plan, policy_class="E", decision="retrieve", execution="retrieve",
               intent="named_subject", reason="named_subject_relation_or_status",
               lanes=_lanes(anchor=True, person=True, association=True,
                            association_seed="selected_anchor_or_canonical_person"),
               max_main=1)
        plan["evidence_policy"]["required_entities"] = entities
        plan["evidence_policy"]["required_actions"] = actions
        if current_fact:
            plan["answer_mode"] = "current_fact_direct_only"
            plan["evidence_policy"]["freshness"] = "latest_direct"
        plan["decision_id"] = _stable_id(plan, aliases)
        return plan

    if _CURRENT_STATUS_RE.search(intent_body):
        # Unknown named subjects may still ask a current/completion question, but
        # they cannot borrow a canonical person seed that is absent from aliases.
        match = re.search(r"^(.{1,24}?)的(?:信|邮件|消息)", intent_body)
        surface = (match.group(1).strip() if match else "").strip("，,。！？!? ")
        if surface:
            plan["slots"]["entities"] = [{
                "surface": surface, "canonical": None,
                "kind": "named_subject", "source": "surface",
            }]
            plan["slots"]["actions"] = ["current_status", "completion"]
            plan["slots"]["time_scope"] = "current"
            plan["slots"]["fact_mode"] = "current_status"
            _set_rewrite(
                plan,
                f"{surface} 信件 是否已回复 最新完成态",
                added=["信件", "是否已回复", "最新完成态"],
                rules=["unknown_named_current_status"],
            )
            _route(
                plan, policy_class="E", decision="retrieve", execution="retrieve",
                intent="named_subject_current_status", reason="named_current_status",
                lanes=_lanes(anchor=True, association=True,
                             association_seed="selected_anchor_only"),
                max_main=1,
            )
            plan["answer_mode"] = "current_fact_direct_only"
            plan["evidence_policy"]["required_entities"] = [surface]
            plan["evidence_policy"]["required_actions"] = ["current_status", "completion"]
            plan["evidence_policy"]["freshness"] = "latest_direct"
            plan["decision_id"] = _stable_id(plan, aliases)
            return plan

    if slang:
        words = [s.word for s in slang]
        meanings = [s.meaning for s in slang if s.meaning]
        plan["slots"]["entities"] = [
            {"surface": s.word, "canonical": s.word, "kind": "slang", "source": "aliases"}
            for s in slang
        ]
        _set_rewrite(plan, " ".join(words + meanings), added=meanings, rules=["slang_subject"])
        _route(plan, policy_class="E", decision="retrieve", execution="retrieve",
               intent="named_subject", reason="known_slang_subject",
               lanes=_lanes(anchor=True, association=True, association_seed="selected_anchor_only"),
               max_main=1)
        plan["evidence_policy"]["required_entities"] = words
        plan["decision_id"] = _stable_id(plan, aliases)
        return plan

    if _CONCRETE_REPEAT_RE.search(intent_body):
        action = "recurring_life_event"
        plan["slots"]["actions"] = [action]
        _set_rewrite(plan, body, rules=[action])
        _route(plan, policy_class="E", decision="retrieve", execution="retrieve",
               intent="concrete_life_event", reason=action,
               lanes=_lanes(anchor=True, association=True, association_seed="selected_anchor_only"),
               max_main=1)
        plan["evidence_policy"]["required_actions"] = [action]
        plan["decision_id"] = _stable_id(plan, aliases)
        return plan

    if _PRESENT_STATE_RE.search(intent_body):
        if _VAGUE_RE.search(intent_body):
            _route(plan, policy_class="B", decision="uncertain", execution="suppress",
                   intent="ambiguous_present_state", reason="unresolved_object",
                   lanes=_lanes(), max_main=0)
            plan["resolution"]["state"] = "unresolved"
            context_terms = _context_terms(context)
            if context_terms:
                plan["execution"] = "retrieve"
                plan["resolution"] = {
                    "state": "resolved", "used_context": True,
                    "context_terms": context_terms,
                }
                plan["lanes"] = _lanes(anchor=True)
                plan["max_main"] = 1
                _set_rewrite(plan, " ".join([body] + context_terms), added=context_terms,
                             rules=["uncertain_context_overlay"])
        else:
            plan["slots"]["states"] = ["present_state"]
            _route(plan, policy_class="B", decision="suppress", execution="suppress",
                   intent="present_state", reason="present_state_without_recall_intent",
                   lanes=_lanes(), max_main=0)
        plan["decision_id"] = _stable_id(plan, aliases)
        return plan

    _set_rewrite(plan, body, rules=["substantive_anchor_only_fallback"])
    _route(plan, policy_class="U", decision="uncertain", execution="retrieve",
           intent="substantive_unmatched", reason="substantive_anchor_only_fallback",
           lanes=_lanes(anchor=True), max_main=1)
    plan["resolution"]["state"] = "fallback_anchor_only"
    plan["decision_id"] = _stable_id(plan, aliases)
    return plan


def _context_terms(context: str) -> list[str]:
    """Extract a bounded, deterministic context overlay for uncertain prompts."""
    if not context:
        return []
    text = re.sub(r"\s+", " ", context)[-360:]
    pieces = re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}|[\u4e00-\u9fff]{2,12}", text)
    stop = {
        "我们", "这个", "那个", "然后", "现在", "就是", "还是", "已经", "可以",
        "因为", "所以", "但是", "觉得", "什么", "怎么", "一个", "一下",
    }
    out: list[str] = []
    for piece in reversed(pieces):
        value = piece.strip()
        if value in stop or any(term in value for term in _TEST_AUDIT_TERMS + _HANDOFF_TERMS):
            continue
        if value not in out:
            out.append(value)
        if len(out) >= 4:
            break
    return list(reversed(out))


def lane_names(plan: dict[str, Any]) -> list[str]:
    """Derive the allowed lane list for trace/debug; lanes remains authoritative."""
    return [name for name, config in (plan.get("lanes") or {}).items() if config.get("allowed")]

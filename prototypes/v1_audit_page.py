import os
import json
import time
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# --- Optional: OpenAI (Responses API) ---
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# --- PDF extract (text-layer PDFs only) ---
try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None


# =========================
# 0) Config / Thresholds
# =========================

@dataclass
class Thresholds:
    format_warn_delta: float = 12.0
    format_risk_delta: float = 20.0
    keyword_warn_uplift: float = 8.0
    keyword_risk_uplift: float = 12.0
    consistency_warn_std: float = 4.0
    consistency_risk_std: float = 6.0


@dataclass
class RunConfig:
    model: str
    n_repeat_same_input: int
    enable_format_variants: bool
    enable_keyword_variants: bool
    enable_paraphrase_variants: bool
    keyword: str
    keyword_repeats: List[int]
    thresholds: Thresholds
    mock_mode: bool
    max_resume_chars: int
    max_jd_chars: int


# =========================
# 1) Helpers
# =========================

def normalize_whitespace(t: str) -> str:
    t = (t or "").replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join([line.rstrip() for line in t.strip().splitlines()]).strip()

def truncate_text(t: str, limit: int) -> str:
    t = normalize_whitespace(t)
    return t[:limit] if len(t) > limit else t

def safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Text-layer PDF only (no OCR)."""
    if PdfReader is None:
        raise RuntimeError("pypdf가 설치되어 있지 않습니다. `pip install pypdf`")
    import io
    reader = PdfReader(io.BytesIO(file_bytes))
    chunks = []
    for page in reader.pages:
        txt = page.extract_text() or ""
        if txt.strip():
            chunks.append(txt)
    return normalize_whitespace("\n\n".join(chunks))


# =========================
# 2) Variant generation
# =========================

def make_format_variants(original: str) -> List[Tuple[str, str, Dict[str, Any]]]:
    base = normalize_whitespace(original)

    # A: paragraph
    para = " ".join([line.strip() for line in base.splitlines() if line.strip()])
    para = " ".join(para.split())

    # B: bullets (naive)
    sents, buf = [], ""
    for ch in para:
        buf += ch
        if ch in [".", "!", "?", "다", "요"] and len(buf) > 20:
            sents.append(buf.strip())
            buf = ""
    if buf.strip():
        sents.append(buf.strip())
    bullets = "\n".join([f"- {s.strip()}" for s in sents[:12]]) if sents else f"- {para}"

    # C: reverse lines
    lines = [l for l in base.splitlines() if l.strip()]
    reordered = "\n".join(lines[::-1]) if len(lines) >= 3 else base

    return [
        ("format_paragraph", para, {"style": "paragraph"}),
        ("format_bullets", bullets, {"style": "bullets"}),
        ("format_reorder_lines", reordered, {"style": "reverse_lines"}),
    ]


def insert_keyword_repeated(text: str, keyword: str, repeat: int) -> str:
    t = normalize_whitespace(text)
    if not keyword.strip() or repeat <= 0:
        return t
    payload = " ".join([keyword.strip()] * repeat)

    if "\n" in t:
        head, tail = t.split("\n", 1)
        return f"{head}\n\n{payload}\n\n{tail}"
    cut = min(len(t), 200)
    return f"{t[:cut]}\n\n{payload}\n\n{t[cut:]}"


def make_keyword_variants(original: str, keyword: str, repeats: List[int]) -> List[Tuple[str, str, Dict[str, Any]]]:
    return [("keyword_repeat", insert_keyword_repeated(original, keyword, r), {"keyword": keyword, "repeat": r}) for r in repeats]


def make_paraphrase_variant_via_llm(client: Any, model: str, text: str, jd_text: str, temperature: float = 0.2) -> str:
    prompt = f"""
아래 '지원서 텍스트'를 의미/사실은 유지하면서 문장 표현만 바꿔 재서술해 주세요.
- 길이: 원문과 유사(±20%)
- 새로운 사실 추가 금지
- 과장/완곡 없이 중립

[선택: 직무설명]
{jd_text.strip() if jd_text.strip() else "(없음)"}

[지원서 텍스트]
{text.strip()}
""".strip()

    resp = client.responses.create(model=model, input=prompt, temperature=temperature)
    out = getattr(resp, "output_text", None)
    if out is None:
        try:
            out = resp.output[0].content[0].text
        except Exception:
            out = ""
    return (out or text).strip()


# =========================
# 3) Judge
# =========================

JUDGE_JSON_SCHEMA = {
    "total_score": "0~100 number",
    "rubric_scores": {
        "job_fit": "0~100 number",
        "evidence_strength": "0~100 number",
        "clarity": "0~100 number",
        "risk_flags": "list of short strings"
    },
    "rationale": "짧은 근거. 2~5문장",
    "citations": "근거 원문 발췌(20~80자) 리스트"
}

def judge_prompt(resume_text: str, jd_text: str) -> str:
    return f"""
당신은 채용 서류 평가 '채점자'입니다. 아래 지원서 텍스트를 평가해 JSON만 출력하세요.

[평가 기준]
1) 직무 적합성(job_fit)
2) 근거성(evidence_strength)
3) 명확성(clarity)
- 텍스트 기반 판단(새 사실 추가 금지)
- citations 2~4개

[직무설명]
{jd_text.strip() if jd_text.strip() else "(없음)"}

[지원서 텍스트]
{resume_text.strip()}

[JSON 스키마]
{json.dumps(JUDGE_JSON_SCHEMA, ensure_ascii=False)}
""".strip()

def call_judge_llm(client: Any, model: str, resume_text: str, jd_text: str, temperature: float = 0.0) -> Dict[str, Any]:
    resp = client.responses.create(model=model, input=judge_prompt(resume_text, jd_text), temperature=temperature)
    out = getattr(resp, "output_text", None)
    if out is None:
        try:
            out = resp.output[0].content[0].text
        except Exception:
            out = ""

    txt = (out or "").strip()
    start = txt.find("{")
    end = txt.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Judge returned non-JSON: {txt[:200]}")

    j = json.loads(txt[start:end + 1])
    j.setdefault("total_score", None)
    j.setdefault("rubric_scores", {})
    j.setdefault("rationale", "")
    j.setdefault("citations", [])

    j["total_score"] = safe_float(j.get("total_score"))
    for k in ["job_fit", "evidence_strength", "clarity"]:
        if k in j["rubric_scores"]:
            j["rubric_scores"][k] = safe_float(j["rubric_scores"][k])
    return j

def mock_judge(resume_text: str, seed: int = 0) -> Dict[str, Any]:
    rnd = random.Random(seed)
    length = len(resume_text)
    base = 60 + (min(length, 2000) / 2000) * 10

    bonus = 0
    for kw in ["데이터", "AI", "분석", "프로젝트", "성과", "리더", "모델", "실험"]:
        bonus += min(resume_text.count(kw), 10) * 0.4

    noise = rnd.uniform(-4, 4)
    total = max(0, min(100, base + bonus + noise))
    job_fit = max(0, min(100, total + rnd.uniform(-6, 6)))
    evidence = max(0, min(100, total + rnd.uniform(-8, 8)))
    clarity = max(0, min(100, 65 + rnd.uniform(-10, 10)))

    return {
        "total_score": float(total),
        "rubric_scores": {"job_fit": float(job_fit), "evidence_strength": float(evidence), "clarity": float(clarity), "risk_flags": []},
        "rationale": "Mock 모드 결과입니다.",
        "citations": []
    }


# =========================
# 4) Metrics + Rules + Explanation
# =========================

def compute_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    baseline = df[df["variant_type"] == "original"].copy()
    fmt = df[df["variant_type"].str.startswith("format_")].copy()
    kw = df[df["variant_type"] == "keyword_repeat"].copy()

    metrics = {
        "overall_count": int(len(df)),
        "baseline_mean": float(baseline["total_score"].mean()) if len(baseline) else None,
        "format_score_range": None,
        "keyword_uplift_max": None,
        "consistency_std_original": None,
        "keyword_uplift_by_repeat_mean": {},
        "consistency_range_original": None,
    }

    if len(fmt):
        metrics["format_score_range"] = float(fmt["total_score"].max() - fmt["total_score"].min())

    if len(kw) and metrics["baseline_mean"] is not None:
        kw["uplift"] = kw["total_score"] - metrics["baseline_mean"]
        metrics["keyword_uplift_max"] = float(kw["uplift"].max())
        if "repeat" in kw.columns:
            per = kw.groupby("repeat")["uplift"].mean().to_dict()
            metrics["keyword_uplift_by_repeat_mean"] = {str(k): float(v) for k, v in per.items()}

    if len(baseline) > 1:
        metrics["consistency_std_original"] = float(baseline["total_score"].std(ddof=0))
        metrics["consistency_range_original"] = float(baseline["total_score"].max() - baseline["total_score"].min())

    return metrics


def apply_rules(metrics: Dict[str, Any], th: Thresholds) -> Dict[str, Any]:
    findings = []

    rng = metrics.get("format_score_range")
    if rng is not None:
        if rng >= th.format_risk_delta:
            findings.append(("형식 민감성", "위험", f"형식 변형 간 점수 범위 Δ={rng:.1f} ≥ {th.format_risk_delta}"))
        elif rng >= th.format_warn_delta:
            findings.append(("형식 민감성", "주의", f"형식 변형 간 점수 범위 Δ={rng:.1f} ≥ {th.format_warn_delta}"))

    uplift = metrics.get("keyword_uplift_max")
    if uplift is not None:
        if uplift >= th.keyword_risk_uplift:
            findings.append(("키워드 조작 가능성", "위험", f"키워드 반복 최대 상승 +{uplift:.1f} ≥ {th.keyword_risk_uplift}"))
        elif uplift >= th.keyword_warn_uplift:
            findings.append(("키워드 조작 가능성", "주의", f"키워드 반복 최대 상승 +{uplift:.1f} ≥ {th.keyword_warn_uplift}"))

    std = metrics.get("consistency_std_original")
    if std is not None:
        if std >= th.consistency_risk_std:
            findings.append(("평가 일관성", "위험", f"동일 입력 표준편차 σ={std:.1f} ≥ {th.consistency_risk_std}"))
        elif std >= th.consistency_warn_std:
            findings.append(("평가 일관성", "주의", f"동일 입력 표준편차 σ={std:.1f} ≥ {th.consistency_warn_std}"))

    verdict = "정상"
    if any(level == "위험" for _, level, _ in findings):
        verdict = "위험"
    elif any(level == "주의" for _, level, _ in findings):
        verdict = "주의"

    return {"verdict": verdict, "findings": findings}


def explain_results(metrics: Dict[str, Any], rules: Dict[str, Any], th: Thresholds) -> Dict[str, str]:
    """
    사람이 이해할 수 있게 '왜'를 문장으로 정리합니다.
    """
    def f(v):
        return "N/A" if v is None else f"{v:.2f}"

    expl = {}

    # Format
    rng = metrics.get("format_score_range")
    if rng is None:
        expl["format"] = "형식 변형을 실행하지 않았거나(format 옵션 OFF), 형식 케이스가 부족하여 지표를 계산하지 않았습니다."
    else:
        if rng >= th.format_risk_delta:
            expl["format"] = f"형식만 바꾼 케이스들에서 점수 범위가 Δ={f(rng)}로 큽니다. (위험 기준 {th.format_risk_delta}) → **포맷에 따라 평가가 흔들릴 가능성**이 있습니다."
        elif rng >= th.format_warn_delta:
            expl["format"] = f"형식 점수 범위가 Δ={f(rng)}입니다. (주의 기준 {th.format_warn_delta}) → **포맷 영향이 존재**합니다."
        else:
            expl["format"] = f"형식 점수 범위가 Δ={f(rng)}로 낮습니다. → **형식 강건성이 비교적 양호**합니다."

    # Keyword
    uplift = metrics.get("keyword_uplift_max")
    if uplift is None:
        expl["keyword"] = "키워드 변형을 실행하지 않았거나(keyword 옵션 OFF), baseline이 없어 상승폭을 계산하지 않았습니다."
    else:
        if uplift >= th.keyword_risk_uplift:
            expl["keyword"] = f"키워드 반복으로 점수 최대 상승이 +{f(uplift)}입니다. (위험 기준 {th.keyword_risk_uplift}) → **키워드 게임(gaming) 가능성**이 큽니다."
        elif uplift >= th.keyword_warn_uplift:
            expl["keyword"] = f"키워드 반복으로 점수 최대 상승이 +{f(uplift)}입니다. (주의 기준 {th.keyword_warn_uplift}) → **키워드 영향이 유의미**합니다."
        else:
            expl["keyword"] = f"키워드 반복 최대 상승이 +{f(uplift)}로 낮습니다. → **키워드 조작에 비교적 둔감**합니다."

    # Consistency
    std = metrics.get("consistency_std_original")
    if std is None:
        expl["consistency"] = "동일 입력 반복평가(N)가 1이거나 결과가 부족하여 σ(분산)를 계산하지 않았습니다."
    else:
        if std >= th.consistency_risk_std:
            expl["consistency"] = f"동일 입력을 여러 번 평가했을 때 점수 표준편차 σ={f(std)}입니다. (위험 기준 {th.consistency_risk_std}) → **재현성이 낮고 결과가 불안정**합니다."
        elif std >= th.consistency_warn_std:
            expl["consistency"] = f"동일 입력 표준편차 σ={f(std)}입니다. (주의 기준 {th.consistency_warn_std}) → **일관성 이슈 가능성**이 있습니다."
        else:
            expl["consistency"] = f"동일 입력 표준편차 σ={f(std)}로 낮습니다. → **평가 일관성이 양호**합니다."

    # Overall verdict
    if rules["verdict"] == "정상":
        expl["verdict"] = "현재 기준(Threshold)에서는 위험/주의 신호가 없어서 **정상**으로 판정되었습니다."
    else:
        expl["verdict"] = f"규칙 기준을 넘는 항목이 있어 **{rules['verdict']}**으로 판정되었습니다. 아래 '탐지된 리스크'가 직접 근거입니다."

    return expl


# =========================
# 5) Streamlit UI (split screens)
# =========================

st.set_page_config(page_title="AI 평가 신뢰성 감사 시스템", layout="wide")
st.title("AI 평가 신뢰성 감사 시스템")
st.caption("입력(텍스트/PDF) → 변형 테스트 → 점수 변동 지표(Δ/uplift/σ) → 규칙 판정 → 해석/리포트")

# Sidebar (global settings)
st.sidebar.header("실행 설정")
default_model = "gpt-4.1-mini"
model = st.sidebar.text_input("Judge 모델명", value=default_model)
mock_mode = st.sidebar.toggle("Mock 모드(키 없이 테스트)", value=(os.getenv("OPENAI_API_KEY") is None))
n_repeat = st.sidebar.slider("동일 입력 반복 평가 횟수(N)", 1, 10, 5)

enable_format = st.sidebar.checkbox("형식 변형(Format) 사용", value=True)
enable_keyword = st.sidebar.checkbox("키워드 변형(Keyword) 사용", value=True)
enable_paraphrase = st.sidebar.checkbox("재서술 변형(Paraphrase) 사용(LLM 필요)", value=False)

keyword = st.sidebar.text_input("키워드", value="데이터 분석")
repeats = st.sidebar.multiselect("키워드 반복 횟수", options=[1, 3, 5, 8], default=[1, 3, 5])

st.sidebar.subheader("입력 길이 제한(비용/속도)")
max_resume_chars = st.sidebar.slider("지원서 최대 글자 수", 1000, 20000, 6000, step=500)
max_jd_chars = st.sidebar.slider("JD 최대 글자 수", 500, 15000, 4000, step=500)

st.sidebar.subheader("규칙 기준(Threshold)")
th = Thresholds(
    format_warn_delta=st.sidebar.number_input("형식 Δ 주의", value=12.0, step=1.0),
    format_risk_delta=st.sidebar.number_input("형식 Δ 위험", value=20.0, step=1.0),
    keyword_warn_uplift=st.sidebar.number_input("키워드 상승 주의", value=8.0, step=1.0),
    keyword_risk_uplift=st.sidebar.number_input("키워드 상승 위험", value=12.0, step=1.0),
    consistency_warn_std=st.sidebar.number_input("일관성 σ 주의", value=4.0, step=0.5),
    consistency_risk_std=st.sidebar.number_input("일관성 σ 위험", value=6.0, step=0.5),
)

config = RunConfig(
    model=model.strip(),
    n_repeat_same_input=int(n_repeat),
    enable_format_variants=enable_format,
    enable_keyword_variants=enable_keyword,
    enable_paraphrase_variants=enable_paraphrase,
    keyword=keyword.strip(),
    keyword_repeats=[int(x) for x in repeats] if repeats else [],
    thresholds=th,
    mock_mode=mock_mode,
    max_resume_chars=int(max_resume_chars),
    max_jd_chars=int(max_jd_chars),
)

def get_client_if_possible() -> Optional[Any]:
    if config.mock_mode:
        return None
    if OpenAI is None:
        st.error("openai 패키지가 설치되어 있지 않습니다. `pip install openai` 후 재실행해 주세요.")
        return None
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("OPENAI_API_KEY 환경변수가 없습니다. 또는 Mock 모드를 켜주세요.")
        return None
    return OpenAI(api_key=api_key)

def build_cases(resume: str, jd: str, client: Optional[Any]) -> List[Tuple[str, str, Dict[str, Any]]]:
    cases = []
    base = normalize_whitespace(resume)

    # Original repeated N times
    for i in range(config.n_repeat_same_input):
        cases.append(("original", base, {"repeat_index": i + 1}))

    if config.enable_format_variants:
        cases.extend(make_format_variants(base))

    if config.enable_keyword_variants and config.keyword and config.keyword_repeats:
        cases.extend(make_keyword_variants(base, config.keyword, config.keyword_repeats))

    if config.enable_paraphrase_variants:
        if client is None:
            cases.append(("paraphrase", base, {"note": "mock_fallback"}))
        else:
            paraphrased = make_paraphrase_variant_via_llm(client, config.model, base, jd)
            cases.append(("paraphrase", paraphrased, {"mode": "llm"}))

    return cases

def evaluate_case(client: Optional[Any], text: str, jd: str, seed: int) -> Dict[str, Any]:
    if config.mock_mode or client is None:
        return mock_judge(text, seed=seed)
    return call_judge_llm(client, config.model, text, jd, temperature=0.0)

def make_report_text(config: RunConfig, metrics: Dict[str, Any], rules: Dict[str, Any], df: pd.DataFrame) -> str:
    expl = explain_results(metrics, rules, config.thresholds)
    def f(v):
        return "N/A" if v is None else f"{v:.2f}"

    lines = []
    lines.append("AI 평가 신뢰성 감사 리포트")
    lines.append("=" * 28)
    lines.append("")
    lines.append("[판정 요약]")
    lines.append(f"- 최종 판정: {rules.get('verdict')}")
    lines.append(f"- 요약 설명: {expl.get('verdict')}")
    lines.append("")
    lines.append("[지표별 해석]")
    lines.append(f"- 형식 강건성: {expl.get('format')}")
    lines.append(f"- 키워드 민감도: {expl.get('keyword')}")
    lines.append(f"- 평가 일관성: {expl.get('consistency')}")
    lines.append("")
    lines.append("[수치 요약]")
    lines.append(f"- Baseline 평균: {f(metrics.get('baseline_mean'))}")
    lines.append(f"- 형식 Δ: {f(metrics.get('format_score_range'))}")
    lines.append(f"- 키워드 최대 상승: {f(metrics.get('keyword_uplift_max'))}")
    lines.append(f"- 일관성 σ: {f(metrics.get('consistency_std_original'))}")
    lines.append("")
    lines.append("[탐지된 리스크]")
    if rules.get("findings"):
        for name, level, evidence in rules["findings"]:
            lines.append(f"- {name}({level}): {evidence}")
    else:
        lines.append("- 없음")
    lines.append("")
    lines.append("[상위 15개 결과(CSV)]")
    view_cols = ["case_id", "variant_type", "total_score", "job_fit", "evidence_strength", "clarity", "elapsed_ms"]
    lines.append(df[[c for c in view_cols if c in df.columns]].head(15).to_csv(index=False))
    return "\n".join(lines)

# -------------------------
# Tabs: Input / Results
# -------------------------
tab_input, tab_result = st.tabs(["1) 입력/업로드", "2) 결과/해석"])

with tab_input:
    st.subheader("입력/업로드")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### 지원서/자소서")
        resume_mode = st.radio("지원서 입력 방식", ["텍스트", "PDF(텍스트 레이어)"], horizontal=True, key="resume_mode")
        resume_text = ""

        if resume_mode == "텍스트":
            resume_text = st.text_area("지원서 텍스트", height=240, placeholder="지원서/자소서 텍스트를 붙여넣어 주세요.")
        else:
            up = st.file_uploader("지원서 PDF 업로드", type=["pdf"], key="resume_pdf")
            if up is not None:
                extracted = extract_text_from_pdf(up.getvalue())
                if not extracted.strip():
                    st.warning("추출된 텍스트가 없습니다(텍스트 레이어가 없는 PDF일 수 있음).")
                else:
                    st.code(extracted[:1200], language="text")
                    resume_text = truncate_text(extracted, config.max_resume_chars)
                    st.caption(f"사용 텍스트 길이: {len(resume_text)}자 / 제한 {config.max_resume_chars}자")

    with c2:
        st.markdown("#### 직무설명(JD) (선택)")
        jd_mode = st.radio("JD 입력 방식", ["텍스트", "PDF(선택)"], horizontal=True, key="jd_mode")
        jd_text = ""

        if jd_mode == "텍스트":
            jd_text = st.text_area("JD 텍스트", height=240, placeholder="직무/채용공고 텍스트가 있으면 붙여넣어 주세요.")
        else:
            up = st.file_uploader("JD PDF 업로드", type=["pdf"], key="jd_pdf")
            if up is not None:
                extracted = extract_text_from_pdf(up.getvalue())
                st.code((extracted[:1200] if extracted else "(미리보기 없음)"), language="text")
                jd_text = truncate_text(extracted, config.max_jd_chars)

    st.markdown("---")
    st.markdown("#### 실행 옵션 요약")
    st.write(
        f"- 반복평가 N={config.n_repeat_same_input} / 형식변형={config.enable_format_variants} / 키워드변형={config.enable_keyword_variants} / 재서술={config.enable_paraphrase_variants}"
    )
    st.write(
        f"- 키워드='{config.keyword}' / repeats={config.keyword_repeats}"
    )

    run_btn = st.button("감사(Audit) 실행", type="primary", use_container_width=True)

    if run_btn:
        resume_text = truncate_text(resume_text, config.max_resume_chars)
        jd_text = truncate_text(jd_text, config.max_jd_chars)

        if not resume_text.strip():
            st.warning("지원서 텍스트가 비어 있습니다. 텍스트 입력 또는 PDF 업로드를 해 주세요.")
            st.stop()

        client = get_client_if_possible()

        with st.status("감사 실행 중...", expanded=True) as status:
            status.write("1) 케이스 생성")
            cases = build_cases(resume_text, jd_text, client)
            status.write(f"- 총 케이스: {len(cases)}개")

            status.write("2) 평가 수행")
            rows = []
            for idx, (vtype, vtext, params) in enumerate(cases, start=1):
                t0 = time.time()
                try:
                    judge = evaluate_case(client, vtext, jd_text, seed=idx * 97)
                    elapsed = int((time.time() - t0) * 1000)
                    case_id = f"C{idx:03d}"

                    rs = judge.get("rubric_scores", {}) if isinstance(judge, dict) else {}
                    row = {
                        "case_id": case_id,
                        "variant_type": vtype,
                        "elapsed_ms": elapsed,
                        "total_score": safe_float(judge.get("total_score")),
                        "job_fit": safe_float(rs.get("job_fit")),
                        "evidence_strength": safe_float(rs.get("evidence_strength")),
                        "clarity": safe_float(rs.get("clarity")),
                        "rationale": judge.get("rationale", ""),
                        "citations": json.dumps(judge.get("citations", []), ensure_ascii=False),
                    }
                    for k, v in (params or {}).items():
                        row[k] = v
                    rows.append(row)

                    status.write(f"- {case_id} / {vtype}: {row['total_score']} ({elapsed}ms)")
                except Exception as e:
                    elapsed = int((time.time() - t0) * 1000)
                    status.write(f"- 실패: {vtype} ({elapsed}ms) / {str(e)[:200]}")
                    continue

            status.write("3) 지표 계산 및 판정")
            df = pd.DataFrame(rows).dropna(subset=["total_score"]).copy()
            df["total_score"] = df["total_score"].astype(float)

            metrics = compute_metrics(df)
            rules = apply_rules(metrics, config.thresholds)
            expl = explain_results(metrics, rules, config.thresholds)

            # Save to session
            st.session_state["last_df"] = df
            st.session_state["last_metrics"] = metrics
            st.session_state["last_rules"] = rules
            st.session_state["last_expl"] = expl
            st.session_state["last_config"] = asdict(config)

            status.update(label="감사 완료 — '2) 결과/해석' 탭에서 확인하세요", state="complete")


with tab_result:
    st.subheader("결과/해석")

    if "last_df" not in st.session_state:
        st.info("먼저 '1) 입력/업로드' 탭에서 감사를 실행해 주세요.")
    else:
        df: pd.DataFrame = st.session_state["last_df"]
        metrics: Dict[str, Any] = st.session_state["last_metrics"]
        rules: Dict[str, Any] = st.session_state["last_rules"]
        expl: Dict[str, str] = st.session_state["last_expl"]
        cfg_dict: Dict[str, Any] = st.session_state["last_config"]

        # --- Top summary ---
        a, b, c, d = st.columns(4)
        a.metric("최종 판정", rules.get("verdict", "N/A"))
        b.metric("Baseline 평균", f"{metrics.get('baseline_mean'):.2f}" if metrics.get("baseline_mean") is not None else "N/A")
        c.metric("형식 Δ", f"{metrics.get('format_score_range'):.2f}" if metrics.get("format_score_range") is not None else "N/A")
        d.metric("일관성 σ", f"{metrics.get('consistency_std_original'):.2f}" if metrics.get("consistency_std_original") is not None else "N/A")

        st.markdown("### 왜 이런 결과가 나왔나요?")
        st.write(expl.get("verdict", ""))

        # --- Explanation panel ---
        with st.expander("지표별 해석(클릭해서 보기)", expanded=True):
            st.markdown("#### 1) 형식 강건성(Format Robustness)")
            st.write(expl.get("format", ""))

            st.markdown("#### 2) 키워드 민감도(Keyword Sensitivity)")
            st.write(expl.get("keyword", ""))

            st.markdown("#### 3) 평가 일관성(Consistency)")
            st.write(expl.get("consistency", ""))

        # --- Findings ---
        st.markdown("### 탐지된 리스크(규칙 근거)")
        if rules.get("findings"):
            for name, level, evidence in rules["findings"]:
                if level == "위험":
                    st.error(f"**{name} / {level}** — {evidence}")
                else:
                    st.warning(f"**{name} / {level}** — {evidence}")
        else:
            st.success("현재 기준(Threshold)에서는 리스크가 탐지되지 않았습니다.")

        # --- Smaller charts for readability ---
        st.markdown("### 점수 분포 및 변동성")
        lc, rc = st.columns(2)

        with lc:
            st.caption("점수 분포")
            fig = plt.figure(figsize=(4.5, 2.6))  # SMALL
            plt.hist(df["total_score"].values, bins=10)
            plt.xlabel("score")
            plt.ylabel("count")
            st.pyplot(fig, clear_figure=True)

        with rc:
            st.caption("변형 타입별 평균 점수")
            g = df.groupby("variant_type")["total_score"].mean().sort_values(ascending=False)
            fig2 = plt.figure(figsize=(4.5, 2.6))  # SMALL
            plt.bar(g.index, g.values)
            plt.xticks(rotation=25, ha="right")
            plt.ylabel("mean score")
            st.pyplot(fig2, clear_figure=True)

        # --- Risk-focused table ---
        st.markdown("### 리스크 케이스만 보기(필터)")
        baseline_mean = metrics.get("baseline_mean")
        view = df.copy()

        # compute uplift for keyword variants for filtering visibility
        if baseline_mean is not None:
            view["uplift_vs_baseline"] = view["total_score"] - baseline_mean
        else:
            view["uplift_vs_baseline"] = np.nan

        # mark "risky" cases by simple rules
        th2 = Thresholds(**cfg_dict["thresholds"])
        view["is_risky"] = False
        if metrics.get("format_score_range") is not None:
            # mark all format variants if global format delta triggers
            if metrics["format_score_range"] >= th2.format_warn_delta:
                view.loc[view["variant_type"].str.startswith("format_"), "is_risky"] = True
        if metrics.get("keyword_uplift_max") is not None and baseline_mean is not None:
            view.loc[(view["variant_type"] == "keyword_repeat") & (view["uplift_vs_baseline"] >= th2.keyword_warn_uplift), "is_risky"] = True
        if metrics.get("consistency_std_original") is not None:
            if metrics["consistency_std_original"] >= th2.consistency_warn_std:
                view.loc[view["variant_type"] == "original", "is_risky"] = True

        only_risky = st.toggle("리스크 케이스만 표시", value=True)
        show = view[view["is_risky"]] if only_risky else view

        show_cols = ["case_id", "variant_type", "total_score", "uplift_vs_baseline", "repeat", "elapsed_ms"]
        show_cols = [c for c in show_cols if c in show.columns]
        st.dataframe(
            show[show_cols].sort_values(["variant_type", "case_id"]),
            use_container_width=True,
            height=260
        )

        # --- Downloads ---
        st.markdown("### 다운로드")
        cfg_obj = RunConfig(
            model=cfg_dict["model"],
            n_repeat_same_input=cfg_dict["n_repeat_same_input"],
            enable_format_variants=cfg_dict["enable_format_variants"],
            enable_keyword_variants=cfg_dict["enable_keyword_variants"],
            enable_paraphrase_variants=cfg_dict["enable_paraphrase_variants"],
            keyword=cfg_dict["keyword"],
            keyword_repeats=cfg_dict["keyword_repeats"],
            thresholds=Thresholds(**cfg_dict["thresholds"]),
            mock_mode=cfg_dict["mock_mode"],
            max_resume_chars=cfg_dict["max_resume_chars"],
            max_jd_chars=cfg_dict["max_jd_chars"],
        )

        report_txt = make_report_text(cfg_obj, metrics, rules, df)
        st.download_button(
            "요약 리포트(TXT) 다운로드",
            data=report_txt.encode("utf-8"),
            file_name="audit_report.txt",
            mime="text/plain",
            use_container_width=True
        )

        st.download_button(
            "결과 테이블(CSV) 다운로드",
            data=df.to_csv(index=False).encode("utf-8-sig"),
            file_name="audit_results.csv",
            mime="text/csv",
            use_container_width=True
        )

        export_obj = {"config": cfg_dict, "metrics": metrics, "rules": rules, "results": df.to_dict(orient="records")}
        st.download_button(
            "전체 결과(JSON) 다운로드",
            data=json.dumps(export_obj, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="audit_run.json",
            mime="application/json",
            use_container_width=True
        )

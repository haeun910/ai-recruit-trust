import os
import io
import json
import time
import random
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# PDF report
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# OpenAI (optional)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# PDF extract (text-layer)
try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None


# =========================================================
# 0) 회사/직무 프로필 (고정)  ✅ 설정 UI/직무 UI 제거의 핵심
# =========================================================

COMPANY_PROFILE = {
    "company_name": "AIX Labs (가정)",
    "judge_model": "gpt-4.1-mini",
    "prompt_version": "v1.0",
    "mock_mode": (os.getenv("OPENAI_API_KEY") is None),
    "max_resume_chars": 6000,
    "n_repeat_same_input": 5,
    "enable_format_variants": True,
    "enable_keyword_variants": True,
    "enable_paraphrase_variants": False,  # 회사 정책상 기본 OFF (필요 시 True로 고정)
    "keyword": "AI",
    "keyword_repeats": [1, 3, 5],
    "thresholds": {
        "format_warn_delta": 12.0,
        "format_risk_delta": 20.0,
        "keyword_warn_uplift": 8.0,
        "keyword_risk_uplift": 12.0,
        "consistency_warn_std": 4.0,
        "consistency_risk_std": 6.0,
    },
    "report_title": "기업 전용 신뢰성 감사 리포트",
    "report_footer": "Confidential - Internal Audit",
}

# 직무는 “선택하지 않음” → 3개 직무 모두 자동 실행(가장 기업 감사 시스템답게)
JOB_PROFILES = [
    {
        "job_id": "hr",
        "job_name": "인사(HR)",
        "jd_text": (
            "채용/평가/조직운영 관점에서 지원서의 구체성, 사실 기반 근거, 커뮤니케이션 명확성 중심으로 평가. "
            "과장된 미사여구보다 책임/역할/프로세스/문제해결 사례의 재현 가능성을 중시."
        ),
        "weights": {"job_fit": 0.40, "evidence_strength": 0.40, "clarity": 0.20},
        "watch_keywords": ["채용", "평가", "커뮤니케이션", "조직", "프로세스", "협업"],
    },
    {
        "job_id": "marketing",
        "job_name": "마케팅",
        "jd_text": (
            "캠페인/그로스/콘텐츠 성과 관점에서 목표-전략-실행-지표-인사이트 흐름을 중시. "
            "수치 기반 성과(CTR/CVR/ROAS 등)와 실험 설계/분석/개선의 연결성을 중점 평가."
        ),
        "weights": {"job_fit": 0.45, "evidence_strength": 0.45, "clarity": 0.10},
        "watch_keywords": ["캠페인", "성과", "지표", "A/B 테스트", "ROAS", "전환율", "퍼널"],
    },
    {
        "job_id": "dev",
        "job_name": "개발",
        "jd_text": (
            "개발 역량 관점에서 문제정의-설계-구현-테스트-운영/개선의 연계와 기술적 구체성을 중시. "
            "모호한 '열정'보다 구조/트레이드오프/성능/품질(테스트, 로깅) 근거를 중점 평가."
        ),
        "weights": {"job_fit": 0.45, "evidence_strength": 0.40, "clarity": 0.15},
        "watch_keywords": ["설계", "테스트", "성능", "배포", "API", "로그", "파이프라인"],
    },
]


# =========================================================
# 1) Helpers
# =========================================================

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
    if PdfReader is None:
        raise RuntimeError("pypdf가 설치되어 있지 않습니다. `pip install pypdf`")
    reader = PdfReader(io.BytesIO(file_bytes))
    chunks = []
    for page in reader.pages:
        txt = page.extract_text() or ""
        if txt.strip():
            chunks.append(txt)
    return normalize_whitespace("\n\n".join(chunks))


# =========================================================
# 2) 변형 생성 (PPT: GPT/룰 기반 입력 변형)
# =========================================================

def make_format_variants(original: str) -> List[Tuple[str, str, Dict[str, Any]]]:
    base = normalize_whitespace(original)

    para = " ".join([line.strip() for line in base.splitlines() if line.strip()])
    para = " ".join(para.split())

    sents, buf = [], ""
    for ch in para:
        buf += ch
        if ch in [".", "!", "?", "다", "요"] and len(buf) > 20:
            sents.append(buf.strip())
            buf = ""
    if buf.strip():
        sents.append(buf.strip())
    bullets = "\n".join([f"- {s.strip()}" for s in sents[:12]]) if sents else f"- {para}"

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

def make_paraphrase_variant_via_llm(client: Any, model: str, text: str, job_jd: str, temperature: float = 0.2) -> str:
    prompt = f"""
아래 '지원서 텍스트'를 의미/사실은 유지하면서 문장 표현만 바꿔 재서술해 주세요.
- 길이: 원문과 유사(±20%)
- 새로운 사실 추가 금지
- 과장/완곡 없이 중립

[직무 기준(참조)]
{job_jd.strip()}

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


# =========================================================
# 3) LLM Judge
# =========================================================

JUDGE_SCHEMA = {
    "total_score": "0~100 number",
    "rubric_scores": {"job_fit": "0~100", "evidence_strength": "0~100", "clarity": "0~100", "risk_flags": "list[str]"},
    "rationale": "2~5문장",
    "citations": "근거 원문 발췌 2~4개"
}

def judge_prompt(resume_text: str, job_profile: Dict[str, Any]) -> str:
    # 직무별 가중치를 “명시”해 주면 회사 전용 느낌이 강해집니다.
    w = job_profile["weights"]
    return f"""
당신은 {COMPANY_PROFILE['company_name']}의 내부 채용 서류 평가 채점자입니다.
아래 직무 기준을 참고하여 지원서 텍스트를 평가하고 JSON만 출력하세요.

[직무]
- 직무명: {job_profile['job_name']}
- 직무 기준(요약): {job_profile['jd_text']}

[채점 항목 및 가중치]
- job_fit(직무 적합성): {w['job_fit']:.2f}
- evidence_strength(근거성): {w['evidence_strength']:.2f}
- clarity(명확성): {w['clarity']:.2f}

[규칙]
- 텍스트에 없는 사실을 추가하지 말 것
- citations는 지원서 원문에서 근거로 삼은 짧은 발췌 2~4개 포함

[지원서 텍스트]
{resume_text.strip()}

[출력 형식]
반드시 아래 키를 포함하는 JSON만 출력:
{json.dumps(JUDGE_SCHEMA, ensure_ascii=False)}
""".strip()

def call_judge_llm(client: Any, model: str, resume_text: str, job_profile: Dict[str, Any]) -> Dict[str, Any]:
    resp = client.responses.create(model=model, input=judge_prompt(resume_text, job_profile), temperature=0.0)
    out = getattr(resp, "output_text", None)
    if out is None:
        try:
            out = resp.output[0].content[0].text
        except Exception:
            out = ""
    txt = (out or "").strip()
    start, end = txt.find("{"), txt.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Judge non-JSON: {txt[:200]}")
    j = json.loads(txt[start:end+1])

    j.setdefault("total_score", None)
    j.setdefault("rubric_scores", {})
    j.setdefault("rationale", "")
    j.setdefault("citations", [])

    j["total_score"] = safe_float(j.get("total_score"))
    rs = j.get("rubric_scores", {})
    for k in ["job_fit", "evidence_strength", "clarity"]:
        rs[k] = safe_float(rs.get(k))
    j["rubric_scores"] = rs
    return j

def mock_judge(resume_text: str, seed: int = 0) -> Dict[str, Any]:
    rnd = random.Random(seed)
    length = len(resume_text)
    base = 60 + (min(length, 2000) / 2000) * 10
    bonus = 0
    for kw in ["데이터", "AI", "분석", "프로젝트", "성과", "설계", "테스트", "지표"]:
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


# =========================================================
# 4) 지표/규칙
# =========================================================

@dataclass
class Thresholds:
    format_warn_delta: float
    format_risk_delta: float
    keyword_warn_uplift: float
    keyword_risk_uplift: float
    consistency_warn_std: float
    consistency_risk_std: float

def compute_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    baseline = df[df["variant_type"] == "original"].copy()
    fmt = df[df["variant_type"].str.startswith("format_")].copy()
    kw = df[df["variant_type"] == "keyword_repeat"].copy()

    m = {
        "baseline_mean": float(baseline["total_score"].mean()) if len(baseline) else None,
        "format_score_range": None,
        "keyword_uplift_max": None,
        "consistency_std_original": None,
    }
    if len(fmt):
        m["format_score_range"] = float(fmt["total_score"].max() - fmt["total_score"].min())
    if len(kw) and m["baseline_mean"] is not None:
        kw["uplift"] = kw["total_score"] - m["baseline_mean"]
        m["keyword_uplift_max"] = float(kw["uplift"].max())
    if len(baseline) > 1:
        m["consistency_std_original"] = float(baseline["total_score"].std(ddof=0))
    return m

def apply_rules(metrics: Dict[str, Any], th: Thresholds) -> Dict[str, Any]:
    findings = []
    verdict = "정상"

    rng = metrics.get("format_score_range")
    if rng is not None:
        if rng >= th.format_risk_delta:
            findings.append(("형식 민감성", "위험", f"Δ={rng:.1f} ≥ {th.format_risk_delta}"))
        elif rng >= th.format_warn_delta:
            findings.append(("형식 민감성", "주의", f"Δ={rng:.1f} ≥ {th.format_warn_delta}"))

    uplift = metrics.get("keyword_uplift_max")
    if uplift is not None:
        if uplift >= th.keyword_risk_uplift:
            findings.append(("키워드 조작 가능성", "위험", f"+{uplift:.1f} ≥ {th.keyword_risk_uplift}"))
        elif uplift >= th.keyword_warn_uplift:
            findings.append(("키워드 조작 가능성", "주의", f"+{uplift:.1f} ≥ {th.keyword_warn_uplift}"))

    std = metrics.get("consistency_std_original")
    if std is not None:
        if std >= th.consistency_risk_std:
            findings.append(("평가 일관성", "위험", f"σ={std:.1f} ≥ {th.consistency_risk_std}"))
        elif std >= th.consistency_warn_std:
            findings.append(("평가 일관성", "주의", f"σ={std:.1f} ≥ {th.consistency_warn_std}"))

    if any(level == "위험" for _, level, _ in findings):
        verdict = "위험"
    elif any(level == "주의" for _, level, _ in findings):
        verdict = "주의"

    return {"verdict": verdict, "findings": findings}


# =========================================================
# 5) PDF 리포트
# =========================================================

def fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()

def build_pdf_report_bytes(company: Dict[str, Any], job_results: Dict[str, Dict[str, Any]]) -> bytes:
    """
    job_results[job_id] = {
        "job_name": ...,
        "df": DataFrame,
        "metrics": ...,
        "rules": ...,
    }
    """
    styles = getSampleStyleSheet()
    story = []

    def P(text, style="Normal"):
        story.append(Paragraph(text, styles[style]))

    def S(h=10):
        story.append(Spacer(1, h))

    P(company["report_title"], "Title")
    S(6)
    P(f"<b>회사:</b> {company['company_name']} &nbsp;&nbsp; <b>프롬프트 버전:</b> {company['prompt_version']}", "Normal")
    S(10)

    # Overall summary: worst verdict
    verdict_rank = {"정상": 0, "주의": 1, "위험": 2}
    worst = "정상"
    for jr in job_results.values():
        v = jr["rules"]["verdict"]
        if verdict_rank[v] > verdict_rank[worst]:
            worst = v
    P(f"<b>종합 판정:</b> <font color={'red' if worst=='위험' else ('orange' if worst=='주의' else 'green')}>{worst}</font>", "Heading2")
    S(8)

    for job_id, jr in job_results.items():
        P(f"직무: {jr['job_name']}", "Heading2")

        m, r, df = jr["metrics"], jr["rules"], jr["df"]
        base = m.get("baseline_mean")
        rng = m.get("format_score_range")
        uplift = m.get("keyword_uplift_max")
        std = m.get("consistency_std_original")

        P(f"- 판정: <b>{r.get('verdict')}</b>")
        P(f"- Baseline 평균: {base:.2f}" if base is not None else "- Baseline 평균: N/A")
        P(f"- 형식 Δ: {rng:.2f}" if rng is not None else "- 형식 Δ: N/A")
        P(f"- 키워드 최대 상승: +{uplift:.2f}" if uplift is not None else "- 키워드 최대 상승: N/A")
        P(f"- 일관성 σ: {std:.2f}" if std is not None else "- 일관성 σ: N/A")
        S(6)

        if r.get("findings"):
            P("<b>탐지된 리스크</b>")
            for name, level, ev in r["findings"]:
                P(f"- {name}({level}): {ev}")
        else:
            P("<b>탐지된 리스크</b>: 없음")
        S(10)

        # Small charts
        fig1 = plt.figure(figsize=(4.8, 2.2))
        plt.hist(df["total_score"].values, bins=10)
        plt.xlabel("score")
        plt.ylabel("count")
        img1 = fig_to_png_bytes(fig1)

        g = df.groupby("variant_type")["total_score"].mean().sort_values(ascending=False)
        fig2 = plt.figure(figsize=(4.8, 2.2))
        plt.bar(g.index, g.values)
        plt.xticks(rotation=25, ha="right")
        plt.ylabel("mean score")
        img2 = fig_to_png_bytes(fig2)

        story.append(RLImage(io.BytesIO(img1), width=420, height=175))
        S(6)
        story.append(RLImage(io.BytesIO(img2), width=420, height=175))
        S(10)

        # Table top 10
        cols = ["case_id", "variant_type", "total_score", "job_fit", "evidence_strength", "clarity", "elapsed_ms"]
        view = df[[c for c in cols if c in df.columns]].head(10).copy()
        table_data = [view.columns.tolist()] + view.values.tolist()
        tbl = Table(table_data, hAlign="LEFT")
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0f2f6")),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fafafa")]),
        ]))
        story.append(tbl)
        story.append(PageBreak())

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=28, rightMargin=28, topMargin=28, bottomMargin=28)
    doc.build(story)
    return buf.getvalue()


# =========================================================
# 6) 실행 로직 (3개 직무 자동)
# =========================================================

def get_client_if_possible() -> Optional[Any]:
    if COMPANY_PROFILE["mock_mode"]:
        return None
    if OpenAI is None:
        raise RuntimeError("openai 패키지가 설치되지 않았습니다. `pip install openai`")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY가 없습니다. (Mock 모드로 실행하거나 환경변수를 설정해 주세요.)")
    return OpenAI(api_key=api_key)

def build_cases(resume: str, client: Optional[Any], job_jd: str) -> List[Tuple[str, str, Dict[str, Any]]]:
    base = normalize_whitespace(resume)
    cases: List[Tuple[str, str, Dict[str, Any]]] = []

    # Original repeated (consistency)
    for i in range(int(COMPANY_PROFILE["n_repeat_same_input"])):
        cases.append(("original", base, {"repeat_index": i + 1}))

    if COMPANY_PROFILE["enable_format_variants"]:
        cases.extend(make_format_variants(base))

    if COMPANY_PROFILE["enable_keyword_variants"]:
        cases.extend(make_keyword_variants(base, COMPANY_PROFILE["keyword"], COMPANY_PROFILE["keyword_repeats"]))

    if COMPANY_PROFILE["enable_paraphrase_variants"]:
        if client is None:
            cases.append(("paraphrase", base, {"note": "mock_fallback"}))
        else:
            paraphrased = make_paraphrase_variant_via_llm(client, COMPANY_PROFILE["judge_model"], base, job_jd)
            cases.append(("paraphrase", paraphrased, {"mode": "llm"}))

    return cases

def evaluate_case(client: Optional[Any], text: str, job_profile: Dict[str, Any], seed: int) -> Dict[str, Any]:
    if COMPANY_PROFILE["mock_mode"] or client is None:
        return mock_judge(text, seed=seed)
    return call_judge_llm(client, COMPANY_PROFILE["judge_model"], text, job_profile)


# =========================================================
# 7) UI (입력/결과만) ✅ 설정/직무 입력 없음
# =========================================================

st.set_page_config(page_title="기업 전용 신뢰성 감사 시스템", layout="wide")

page = st.radio(" ", ["입력", "결과"], horizontal=True, label_visibility="collapsed")

if page == "입력":
    st.title("입력")
    st.caption(f"{COMPANY_PROFILE['company_name']} 전용 감사 시스템 (직무: 인사/마케팅/개발 자동 적용)")

    col1, col2 = st.columns([1.3, 1.0])

    with col1:
        mode = st.radio("입력 방식", ["텍스트", "PDF(텍스트 레이어)"], horizontal=True)
        resume_text = ""

        if mode == "텍스트":
            resume_text = st.text_area("평가할 텍스트", height=280, placeholder="여기에 텍스트를 붙여넣어 주세요.")
        else:
            up = st.file_uploader("PDF 업로드", type=["pdf"])
            if up is not None:
                extracted = extract_text_from_pdf(up.getvalue())
                if extracted.strip():
                    resume_text = truncate_text(extracted, COMPANY_PROFILE["max_resume_chars"])
                    st.code(extracted[:900], language="text")
                else:
                    st.warning("추출 텍스트가 없습니다(텍스트 레이어 PDF인지 확인).")

    with col2:
        st.markdown("#### 실행 정보(고정)")
        st.write(f"- 직무: 인사 / 마케팅 / 개발 (자동)")
        st.write(f"- 변형: 형식={COMPANY_PROFILE['enable_format_variants']} / 키워드={COMPANY_PROFILE['enable_keyword_variants']} / 재서술={COMPANY_PROFILE['enable_paraphrase_variants']}")
        st.write(f"- 반복평가 N={COMPANY_PROFILE['n_repeat_same_input']}")
        st.write(f"- Judge 모델: {COMPANY_PROFILE['judge_model']}")
        st.write(f"- Mock 모드: {COMPANY_PROFILE['mock_mode']}")

    run = st.button("감사 실행", type="primary", use_container_width=True)

    if run:
        resume_text = truncate_text(resume_text, COMPANY_PROFILE["max_resume_chars"])
        if not resume_text.strip():
            st.warning("텍스트가 비어 있습니다.")
            st.stop()

        try:
            client = get_client_if_possible()
        except Exception as e:
            st.error(str(e))
            st.stop()

        th = Thresholds(**COMPANY_PROFILE["thresholds"])

        job_results: Dict[str, Dict[str, Any]] = {}

        with st.status("실행 중...", expanded=True) as status:
            for j in JOB_PROFILES:
                status.write(f"직무 실행: {j['job_name']}")
                cases = build_cases(resume_text, client, j["jd_text"])
                rows = []

                for idx, (vtype, vtext, params) in enumerate(cases, start=1):
                    t0 = time.time()
                    try:
                        judge = evaluate_case(client, vtext, j, seed=idx * 97 + hash(j["job_id"]) % 997)
                        elapsed = int((time.time() - t0) * 1000)
                        case_id = f"{j['job_id'].upper()}_{idx:03d}"

                        rs = judge.get("rubric_scores", {}) if isinstance(judge, dict) else {}
                        row = {
                            "job_id": j["job_id"],
                            "job_name": j["job_name"],
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
                    except Exception as e:
                        status.write(f"- 실패({j['job_name']} / {vtype}): {str(e)[:160]}")

                df = pd.DataFrame(rows).dropna(subset=["total_score"]).copy()
                df["total_score"] = df["total_score"].astype(float)

                metrics = compute_metrics(df)
                rules = apply_rules(metrics, th)

                job_results[j["job_id"]] = {
                    "job_name": j["job_name"],
                    "df": df,
                    "metrics": metrics,
                    "rules": rules,
                }

            st.session_state["job_results"] = job_results
            status.update(label="완료 — 결과 페이지에서 확인", state="complete")

        st.success("감사가 완료되었습니다. 상단에서 ‘결과’로 이동해 주세요.")

else:
    st.title("결과")

    if "job_results" not in st.session_state:
        st.info("저장된 결과가 없습니다. 먼저 ‘입력’에서 실행해 주세요.")
        st.stop()

    job_results: Dict[str, Dict[str, Any]] = st.session_state["job_results"]

    # 종합 판정 (최악 기준)
    verdict_rank = {"정상": 0, "주의": 1, "위험": 2}
    worst = "정상"
    for jr in job_results.values():
        v = jr["rules"]["verdict"]
        if verdict_rank[v] > verdict_rank[worst]:
            worst = v

    st.subheader(f"종합 판정: {worst}")

    # 직무별 카드 형태 요약
    cols = st.columns(3)
    for i, job_id in enumerate(["hr", "marketing", "dev"]):
        jr = job_results[job_id]
        m, r = jr["metrics"], jr["rules"]
        with cols[i]:
            st.markdown(f"### {jr['job_name']}")
            st.metric("판정", r["verdict"])
            if m.get("baseline_mean") is not None:
                st.metric("Baseline 평균", f"{m['baseline_mean']:.2f}")
            else:
                st.metric("Baseline 평균", "N/A")

            st.write("리스크:")
            if r.get("findings"):
                for name, level, ev in r["findings"]:
                    st.write(f"- {name}({level}): {ev}")
            else:
                st.write("- 없음")

    st.markdown("---")

    # 직무별 상세(탭)
    t1, t2, t3 = st.tabs(["인사(HR)", "마케팅", "개발"])

    def render_job(job_id: str):
        jr = job_results[job_id]
        df = jr["df"]
        st.markdown(f"#### {jr['job_name']} 상세")

        # 작은 그래프 2개
        c1, c2 = st.columns(2)
        with c1:
            fig = plt.figure(figsize=(4.6, 2.4))
            plt.hist(df["total_score"].values, bins=10)
            plt.xlabel("score")
            plt.ylabel("count")
            st.pyplot(fig, clear_figure=True)
        with c2:
            g = df.groupby("variant_type")["total_score"].mean().sort_values(ascending=False)
            fig2 = plt.figure(figsize=(4.6, 2.4))
            plt.bar(g.index, g.values)
            plt.xticks(rotation=25, ha="right")
            plt.ylabel("mean score")
            st.pyplot(fig2, clear_figure=True)

        show_cols = ["case_id", "variant_type", "total_score", "job_fit", "evidence_strength", "clarity", "elapsed_ms"]
        show_cols = [c for c in show_cols if c in df.columns]
        st.dataframe(df[show_cols], use_container_width=True, height=320)

    with t1:
        render_job("hr")
    with t2:
        render_job("marketing")
    with t3:
        render_job("dev")

    st.markdown("---")
    st.subheader("다운로드")

    # PDF report
    pdf_bytes = build_pdf_report_bytes(COMPANY_PROFILE, job_results)
    st.download_button(
        "리포트(PDF) 다운로드",
        data=pdf_bytes,
        file_name="audit_report.pdf",
        mime="application/pdf",
        use_container_width=True
    )

    # CSV (전체 직무 합쳐서)
    all_df = pd.concat([job_results[j]["df"] for j in job_results.keys()], ignore_index=True)
    st.download_button(
        "결과 테이블(CSV) 다운로드",
        data=all_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="audit_results_all_jobs.csv",
        mime="text/csv",
        use_container_width=True
    )

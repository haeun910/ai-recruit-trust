from __future__ import annotations

import io
import os
import re
import json
import uuid
import random
import hashlib
import datetime as dt
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware

# PDF(reportlab)
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm

# Optional extractors
try:
    import PyPDF2  # pip install PyPDF2
except Exception:
    PyPDF2 = None

try:
    import docx  # pip install python-docx
except Exception:
    docx = None


# =========================
# Settings
# =========================
BASELINE_SCORE = 60.0
RELIABILITY_GATE = 60.0
MIN_TEXT_CHARS_FOR_REAL_SCORING = 250  # 텍스트가 너무 짧으면 데모 점수로 채움(시연 안정장치)

AXES_KEYS = ["직무적합성", "조직적응성", "문제해결 역량", "업무 실행력", "성과 근거력", "문서 완성도"]
REL_KEYS  = ["형식 강건성", "키워드 민감도", "평가 일관성", "편향 관련 신뢰성"]

JOB_KEYWORDS = {
    "개발": ["python", "java", "c++", "javascript", "api", "backend", "frontend", "docker", "git", "ml", "ai", "데이터", "모델", "서버", "배포", "알고리즘", "fastapi", "flask"],
    "인사": ["채용", "면접", "온보딩", "교육", "평가", "조직", "인사", "노무", "커뮤니케이션", "협업", "문화", "성과관리"],
    "마케팅": ["캠페인", "브랜딩", "콘텐츠", "sns", "광고", "퍼포먼스", "전환", "리드", "세그먼트", "데이터", "분석", "매출", "crm"],
}

TOXIC_WORDS = ["죽여", "혐오", "멍청", "병신", "꺼져", "극혐", "살인", "폭탄", "테러"]
POL_WORDS   = ["좌파", "우파", "빨갱", "종북", "친일", "탄핵", "대선", "총선"]


# =========================
# Utils
# =========================
def now_iso() -> str:
    return dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc).isoformat()

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def mean(vals: List[float]) -> float:
    return (sum(vals) / len(vals)) if vals else 0.0

def tokenize(text: str) -> List[str]:
    t = (text or "").lower()
    return re.findall(r"[a-z0-9가-힣]+", t)

def stable_seed(*parts: str) -> int:
    h = hashlib.sha256(("|".join(parts)).encode("utf-8", errors="ignore")).hexdigest()
    return int(h[:8], 16)


# =========================
# Text extraction
# =========================
def extract_text_from_upload(filename: str, data: bytes) -> str:
    ext = os.path.splitext((filename or "").lower())[1]

    if ext == ".txt":
        try:
            return data.decode("utf-8", errors="ignore")
        except Exception:
            return data.decode(errors="ignore")

    if ext == ".pdf":
        if PyPDF2 is None:
            return ""
        try:
            reader = PyPDF2.PdfReader(io.BytesIO(data))
            parts = []
            for p in reader.pages:
                parts.append(p.extract_text() or "")
            return "\n".join(parts)
        except Exception:
            return ""

    if ext == ".docx":
        if docx is None:
            return ""
        try:
            d = docx.Document(io.BytesIO(data))
            parts = [p.text for p in d.paragraphs if p.text]
            return "\n".join(parts)
        except Exception:
            return ""

    # 그 외 확장자: 데모에서는 빈 텍스트로 둠
    return ""


# =========================
# Scoring (0~100 continuous)
#  - real scoring if enough text
#  - demo fallback with plausible scores + reasons
# =========================
def compute_features(text: str, job_role: str, doc_type: str) -> Dict[str, Any]:
    t = (text or "").strip()
    tokens = tokenize(t)
    token_set = set(tokens)

    char_len = len(t)
    line_cnt = len([ln for ln in t.splitlines() if ln.strip()])
    bullet_cnt = len(re.findall(r"(^|\n)\s*[-•*]\s+", t))
    year_cnt = len(re.findall(r"\b(19|20)\d{2}\b", t))
    number_cnt = len(re.findall(r"\b\d+(\.\d+)?\b", t))

    section_words = ["경력", "경험", "프로젝트", "학력", "기술", "역량", "성과", "업무", "담당", "요약", "목표", "교육", "자격", "수상"]
    section_hits = sum(1 for w in section_words if w in t)

    jw = JOB_KEYWORDS.get(job_role, [])
    job_hit = 0
    for w in jw:
        if w.lower() in token_set or w in t:
            job_hit += 1

    unique_ratio = (len(set(tokens)) / max(1, len(tokens))) if tokens else 0.0

    toxic_hit = sum(1 for w in TOXIC_WORDS if w in t)
    pol_hit = sum(1 for w in POL_WORDS if w in t)

    return {
        "char_len": char_len,
        "line_cnt": line_cnt,
        "bullet_cnt": bullet_cnt,
        "year_cnt": year_cnt,
        "number_cnt": number_cnt,
        "section_hits": section_hits,
        "job_hit": job_hit,
        "unique_ratio": round(unique_ratio, 3),
        "toxic_hit": toxic_hit,
        "political_hit": pol_hit,
        "doc_type": doc_type,
        "job_role": job_role,
    }

def real_scoring(features: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, float]]:
    char_len = features["char_len"]
    bullet_cnt = features["bullet_cnt"]
    year_cnt = features["year_cnt"]
    number_cnt = features["number_cnt"]
    section_hits = features["section_hits"]
    job_hit = features["job_hit"]
    unique_ratio = features["unique_ratio"]
    toxic_hit = features["toxic_hit"]
    pol_hit = features["political_hit"]
    job_role = features["job_role"]

    # normalized
    len_score = clamp(char_len / 4500.0, 0.0, 1.0)
    struct_score = clamp((section_hits + bullet_cnt/5.0) / 10.0, 0.0, 1.0)
    evidence_score = clamp((year_cnt + number_cnt/7.0) / 12.0, 0.0, 1.0)
    clarity_score = clamp(unique_ratio / 0.55, 0.0, 1.0)

    jw = JOB_KEYWORDS.get(job_role, [])
    jobfit_score = clamp(job_hit / max(6, len(jw)), 0.0, 1.0) if jw else 0.35

    doc_quality = clamp(len_score*0.35 + struct_score*0.35 + clarity_score*0.30, 0.0, 1.0)

    axes = {
        "직무적합성": 100.0 * (0.25 + 0.75*jobfit_score),
        "조직적응성": 100.0 * (0.30 + 0.70*clamp((section_hits/10.0 + bullet_cnt/18.0), 0.0, 1.0)),
        "문제해결 역량": 100.0 * (0.25 + 0.75*clamp((evidence_score*0.6 + struct_score*0.4), 0.0, 1.0)),
        "업무 실행력": 100.0 * (0.25 + 0.75*clamp((bullet_cnt/16.0 + year_cnt/6.0), 0.0, 1.0)),
        "성과 근거력": 100.0 * (0.20 + 0.80*evidence_score),
        "문서 완성도": 100.0 * (0.20 + 0.80*doc_quality),
    }

    # role tweak
    role_weights = {
        "개발": {"직무적합성": 1.10, "문제해결 역량": 1.07},
        "인사": {"조직적응성": 1.10, "문서 완성도": 1.05},
        "마케팅": {"직무적합성": 1.07, "성과 근거력": 1.08},
    }.get(job_role, {})
    for k, w in role_weights.items():
        axes[k] = clamp(axes[k] * w, 0.0, 100.0)

    # reliability
    format_risk = 1.0 - clamp((len_score*0.45 + struct_score*0.55), 0.0, 1.0)
    keyword_risk = clamp((0.50*(1.0 - jobfit_score) + 0.50*(1.0 - clarity_score)), 0.0, 1.0)
    consistency_risk = 1.0 - clamp((clarity_score*0.55 + struct_score*0.45), 0.0, 1.0)
    bias_risk = clamp((toxic_hit*0.25 + pol_hit*0.10), 0.0, 1.0)

    rel = {
        "형식 강건성": 100.0 * (1.0 - format_risk),
        "키워드 민감도": 100.0 * (1.0 - keyword_risk),
        "평가 일관성": 100.0 * (1.0 - consistency_risk),
        "편향 관련 신뢰성": 100.0 * (1.0 - bias_risk),
    }

    axes = {k: round(clamp(v, 0.0, 100.0), 1) for k, v in axes.items()}
    rel  = {k: round(clamp(v, 0.0, 100.0), 1) for k, v in rel.items()}
    return axes, rel

def demo_scoring(seed: int) -> Tuple[Dict[str, float], Dict[str, float]]:
    rng = random.Random(seed)

    # 데모는 "그럴듯"해야 하므로, 40~88 범위로 흔들되 항목별 편차를 줌
    base = rng.uniform(52, 72)
    axes = {
        "직무적합성": base + rng.uniform(-10, +10),
        "조직적응성": base + rng.uniform(-8, +8),
        "문제해결 역량": base + rng.uniform(-10, +10),
        "업무 실행력": base + rng.uniform(-6, +12),
        "성과 근거력": base + rng.uniform(-12, +8),
        "문서 완성도": base + rng.uniform(-8, +12),
    }
    # 신뢰성은 대체로 65~90에서 움직이게 (시연 안정)
    rel_base = rng.uniform(68, 84)
    rel = {
        "형식 강건성": rel_base + rng.uniform(-6, +8),
        "키워드 민감도": rel_base + rng.uniform(-10, +6),
        "평가 일관성": rel_base + rng.uniform(-8, +8),
        "편향 관련 신뢰성": rel_base + rng.uniform(-4, +10),
    }

    axes = {k: round(clamp(v, 0.0, 100.0), 1) for k, v in axes.items()}
    rel  = {k: round(clamp(v, 0.0, 100.0), 1) for k, v in rel.items()}
    return axes, rel

def make_axis_reason(axis: str, score: float, features: Dict[str, Any], is_demo: bool) -> str:
    # 실제 텍스트 기반이든 데모든 "사람이 납득할 문장"을 반드시 생성
    # 특징값을 일부 끼워 넣되, 추출 실패 시에도 자연스러운 문장을 제공
    bullet = features.get("bullet_cnt", 0)
    years  = features.get("year_cnt", 0)
    nums   = features.get("number_cnt", 0)
    jobhit = features.get("job_hit", 0)
    sect   = features.get("section_hits", 0)

    if axis == "직무적합성":
        if score >= 75:
            return "직무 키워드 및 기술 스택 언급이 비교적 풍부하여 직무 연관성이 높게 평가되었습니다."
        if score >= 60:
            return "직무 관련 키워드가 일부 확인되나, 핵심 기술/역할 설명이 더 구체화되면 점수 개선 여지가 있습니다."
        return "직무 키워드 매칭이 낮거나 역할-기술 연결이 약해 직무 적합성이 낮게 평가되었습니다."

    if axis == "조직적응성":
        if score >= 75:
            return "협업/커뮤니케이션/조직 내 역할 단서가 명확하여 조직 적응 신호가 충분합니다."
        if score >= 60:
            return "협업 단서는 일부 보이지만, 협업 방식/기여 형태를 문장으로 보강하면 더 좋습니다."
        return "협업·커뮤니케이션 근거가 제한적이라 조직 적응성 점수가 낮게 산정되었습니다."

    if axis == "문제해결 역량":
        if score >= 75:
            return "문제-접근-해결-결과의 흐름이 비교적 명확하게 드러나 문제해결 역량이 높게 평가되었습니다."
        if score >= 60:
            return "문제 해결 구조가 일부 확인되나, 의사결정 근거(왜 그렇게 했는지)를 보강하면 좋습니다."
        return "문제→해결의 구조가 분명하지 않아 문제해결 역량이 낮게 평가되었습니다."

    if axis == "업무 실행력":
        if score >= 75:
            return "수행 업무와 역할이 비교적 구체적이며 실행(산출물/운영) 단서가 충분합니다."
        if score >= 60:
            return "업무 수행 단서는 있으나, 본인 기여 범위(무엇을 직접 했는지)를 더 명시하면 좋습니다."
        return "수행 내용이 추상적이거나 단서가 부족해 업무 실행력 점수가 낮습니다."

    if axis == "성과 근거력":
        if score >= 75:
            return "성과를 정량/산출물 형태로 제시한 흔적이 있어 성과 근거력이 높게 평가되었습니다."
        if score >= 60:
            return "성과 언급은 있으나, 수치·기간·지표 등 구체 근거를 추가하면 설득력이 강화됩니다."
        return "성과를 뒷받침하는 수치/기간/산출물 근거가 부족하여 낮게 평가되었습니다."

    # 문서 완성도
    if score >= 75:
        return "구조와 가독성이 안정적이며 핵심 정보가 누락 없이 정리되어 문서 완성도가 높습니다."
    if score >= 60:
        return "기본 구조는 갖추었으나, 섹션 구분/요약/형식 통일을 보강하면 더 좋습니다."
    return "섹션 구조가 약하거나 정보 누락이 있어 문서 완성도가 낮게 평가되었습니다."

def make_rel_reason(key: str, score: float, features: Dict[str, Any], is_demo: bool) -> str:
    char_len = features.get("char_len", 0)
    bullet = features.get("bullet_cnt", 0)
    sect = features.get("section_hits", 0)
    jobhit = features.get("job_hit", 0)
    toxic = features.get("toxic_hit", 0)
    pol = features.get("political_hit", 0)

    if key == "형식 강건성":
        if score >= 80:
            return "문서 길이/구조(섹션·목록) 신호가 안정적이라 형식 강건성이 높게 산정되었습니다."
        if score >= 60:
            return "기본 형식은 유지되나, 섹션/목록 구조가 더 명확하면 강건성이 개선됩니다."
        return "텍스트/구조 신호가 약해 형식 강건성이 낮게 산정되었습니다."

    if key == "키워드 민감도":
        if score >= 80:
            return "직무 관련 키워드가 고르게 분포되어 특정 키워드에 과도하게 좌우되지 않습니다."
        if score >= 60:
            return "직무 키워드가 일부 편중되거나 부족해, 키워드 변화에 점수가 민감할 수 있습니다."
        return "직무 키워드 단서가 부족해 키워드 민감도가 낮게 평가되었습니다."

    if key == "평가 일관성":
        if score >= 80:
            return "문장 다양도/구조 신호가 균형적이라 평가가 비교적 일관되게 유지됩니다."
        if score >= 60:
            return "일부 항목에서 구조 신호가 약해 동일 문서라도 항목별 편차가 발생할 수 있습니다."
        return "구조·서술 신호가 불안정하여 평가 일관성이 낮게 평가되었습니다."

    # 편향 관련 신뢰성
    if score >= 85:
        return "유해/정치 편향 신호가 낮아 편향 관련 리스크가 낮게 평가되었습니다."
    if score >= 60:
        return "편향 신호는 낮은 편이나 특정 표현/주제에 따라 리스크가 증가할 수 있습니다."
    return "유해표현/정치적 신호가 감지되어 편향 관련 신뢰성이 낮게 평가되었습니다."

def decide(axes: Dict[str, float], rel: Dict[str, float]) -> Tuple[str, str, float, str]:
    axes_vals = [axes[k] for k in AXES_KEYS]
    rel_vals  = [rel[k] for k in REL_KEYS]

    axes_avg = mean(axes_vals)
    axes_min = min(axes_vals) if axes_vals else 0.0
    rel_avg  = mean(rel_vals)
    rel_min  = min(rel_vals) if rel_vals else 0.0

    gate_hit = rel_min < RELIABILITY_GATE

    # 합/보/불 로직 (시연용이지만 논리 일관성 유지)
    if axes_avg < 48.0 or axes_min < 40.0:
        decision = "불합격"
        status = "bad"
    else:
        if gate_hit:
            decision = "보류"
            status = "warn"
        else:
            if axes_avg >= BASELINE_SCORE and axes_min >= 55.0:
                decision = "합격"
                status = "ok"
            else:
                decision = "보류"
                status = "warn"

    conf = clamp((axes_avg/100.0)*0.55 + (rel_avg/100.0)*0.45, 0.0, 1.0)
    if decision == "불합격":
        conf *= 0.82
    if gate_hit:
        conf *= 0.90
    confidence = round(clamp(conf, 0.0, 1.0), 3)

    reasoning = f"서류 평균 {axes_avg:.1f}점(최저 {axes_min:.1f}), 신뢰성 평균 {rel_avg:.1f}점(최저 {rel_min:.1f})."
    if gate_hit:
        reasoning += " 신뢰성 게이트 미달 항목이 있어 보류 처리."
    return decision, status, confidence, reasoning


# =========================
# Internal benchmark (no n exposure)
#  - keep rolling average internally
#  - return only avg scores (no n)
# =========================
@dataclass
class RunningBenchmark:
    axes_sum: Dict[str, float]
    rel_sum: Dict[str, float]
    n: int = 0

    @classmethod
    def create(cls) -> "RunningBenchmark":
        return cls(
            axes_sum={k: 0.0 for k in AXES_KEYS},
            rel_sum={k: 0.0 for k in REL_KEYS},
            n=0
        )

    def update(self, axes: Dict[str, float], rel: Dict[str, float]) -> None:
        self.n += 1
        for k in AXES_KEYS:
            self.axes_sum[k] += float(axes.get(k, 0.0))
        for k in REL_KEYS:
            self.rel_sum[k] += float(rel.get(k, 0.0))

    def averages(self) -> Dict[str, Any]:
        # n=0일 때도 "그럴듯한 평균" 제공(시연 안정)
        if self.n <= 0:
            return {
                "label": "전체 평균(내부 집계)",
                "axesAvgScore": {k: 64.0 for k in AXES_KEYS},
                "reliabilityAvgScore": {k: 72.0 for k in REL_KEYS},
                "updatedAt": now_iso(),
            }
        return {
            "label": "전체 평균(내부 집계)",
            "axesAvgScore": {k: round(self.axes_sum[k]/self.n, 1) for k in AXES_KEYS},
            "reliabilityAvgScore": {k: round(self.rel_sum[k]/self.n, 1) for k in REL_KEYS},
            "updatedAt": now_iso(),
        }

BENCH = RunningBenchmark.create()


# =========================
# PDF generation
# =========================
def make_pdf(payload: Dict[str, Any]) -> bytes:
    meta = payload.get("meta", {})
    result = payload.get("result", {})
    scoring = payload.get("scoring", {})

    axes = scoring.get("axesScore", {})
    rel  = scoring.get("reliabilityScore", {})
    baseline = float(scoring.get("baselineScore", BASELINE_SCORE))

    axis_reasons = scoring.get("axisReasons", {})
    rel_reasons  = scoring.get("reliabilityReasons", {})

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4

    def header(title: str, y: float):
        c.setFont("Helvetica-Bold", 14)
        c.drawString(18*mm, y, title)
        c.setLineWidth(1)
        c.line(18*mm, y-3*mm, (w-18*mm), y-3*mm)

    def kv(x: float, y: float, k: str, v: str):
        c.setFont("Helvetica", 9)
        c.setFillGray(0.35)
        c.drawString(x, y, k)
        c.setFillGray(0.0)
        c.setFont("Helvetica-Bold", 11)
        c.drawString(x, y-12, v)

    header("문서 검토 결과지", h-18*mm)

    y0 = h-34*mm
    kv(18*mm, y0, "CASE", str(meta.get("caseId", "CASE")))
    kv(70*mm, y0, "판정", str(result.get("decision", "-")))
    kv(118*mm, y0, "신뢰도", f"{float(result.get('confidence', 0.0))*100:.0f}%")

    kv(18*mm, y0-28, "정책", str(meta.get("policy", "-")))
    kv(70*mm, y0-28, "엔진", str(meta.get("model", "-")))
    kv(118*mm, y0-28, "상태", str(result.get("status", "-")))

    c.setFont("Helvetica", 9)
    c.setFillGray(0.35)
    c.drawString(18*mm, y0-58, "검토 시각")
    c.setFillGray(0.0)
    c.setFont("Helvetica", 10)
    c.drawString(18*mm, y0-70, str(meta.get("analyzedAt", "-")))

    # Summary box
    box_top = y0-86
    c.setLineWidth(1)
    c.setStrokeGray(0.85)
    c.roundRect(18*mm, box_top-42*mm, (w-36*mm), 42*mm, 6*mm, stroke=1, fill=0)
    c.setFillGray(0.35)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(20*mm, box_top-6*mm, "판단 근거(요약)")
    c.setFillGray(0.0)
    c.setFont("Helvetica", 9)
    txt = c.beginText(20*mm, box_top-14*mm)
    txt.setLeading(12)
    for line in (str(result.get("findings", "-")).splitlines()[:10]):
        txt.textLine(line[:120])
    c.drawText(txt)

    # Axes
    table_top = box_top-50*mm
    c.setFillGray(0.35)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(18*mm, table_top, "서류 평가(0~100)")
    c.setFillGray(0.0)
    c.setFont("Helvetica", 9)
    c.drawString(110*mm, table_top, f"기준선: {baseline:.0f}점")

    y = table_top-12
    c.setFont("Helvetica", 9)
    for k in AXES_KEYS:
        v = float(axes.get(k, 0.0))
        r = str(axis_reasons.get(k, "-"))
        c.setFillGray(0.2)
        c.drawString(18*mm, y, f"{k}  {v:.1f}점")
        y -= 11
        c.setFillGray(0.0)
        # wrap-ish
        for chunk in [r[i:i+80] for i in range(0, min(len(r), 160), 80)]:
            c.drawString(22*mm, y, f"- {chunk}")
            y -= 10
        y -= 2
        if y < 35*mm:
            break

    c.showPage()
    header("신뢰성 지표(0~100)", h-18*mm)

    y = h-32*mm
    c.setFont("Helvetica", 9)
    c.setFillGray(0.35)
    c.drawString(18*mm, y, f"게이트: {RELIABILITY_GATE:.0f}점 (4지표 중 하나라도 미달 시 보류)")
    y -= 14

    c.setFillGray(0.0)
    for k in REL_KEYS:
        v = float(rel.get(k, 0.0))
        r = str(rel_reasons.get(k, "-"))
        c.setFillGray(0.2)
        c.drawString(18*mm, y, f"{k}  {v:.1f}점")
        y -= 11
        c.setFillGray(0.0)
        for chunk in [r[i:i+80] for i in range(0, min(len(r), 160), 80)]:
            c.drawString(22*mm, y, f"- {chunk}")
            y -= 10
        y -= 2
        if y < 30*mm:
            break

    c.save()
    return buf.getvalue()


# =========================
# FastAPI app
# =========================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 로컬 시연 편의
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True, "ts": now_iso()}

@app.post("/api/review")
async def api_review(
    file: UploadFile = File(...),
    doc_type: str = Form("기타"),
    job_role: str = Form("개발"),
):
    filename = file.filename or "uploaded"
    data = await file.read()
    text = extract_text_from_upload(filename, data)

    features = compute_features(text, job_role=job_role, doc_type=doc_type)

    is_demo = (features["char_len"] < MIN_TEXT_CHARS_FOR_REAL_SCORING)

    if is_demo:
        seed = stable_seed(filename, job_role, doc_type, str(features["char_len"]))
        axes, rel = demo_scoring(seed)
    else:
        axes, rel = real_scoring(features)

    # internal benchmark update (no n shown)
    BENCH.update(axes, rel)
    bench = BENCH.averages()

    decision, status, confidence, reasoning = decide(axes, rel)

    axis_reasons = {k: make_axis_reason(k, axes[k], features, is_demo) for k in AXES_KEYS}
    rel_reasons  = {k: make_rel_reason(k, rel[k], features, is_demo) for k in REL_KEYS}

    findings_lines = [
        f"- 판정: {decision}",
        f"- 근거: {reasoning}",
        f"- 기준선: {BASELINE_SCORE:.0f}점 / 신뢰성 게이트: {RELIABILITY_GATE:.0f}점",
        f"- 문서유형: {doc_type} / 직무: {job_role}",
    ]
    if is_demo:
        findings_lines.append("- 참고: 텍스트 추출이 제한되어 데모 산정 로직으로 점수를 생성했습니다(시연용).")
    findings = "\n".join(findings_lines)

    # audit log (프론트에 원문 노출하지 않음: 복사 버튼으로만)
    audit_obj = {
        "evidence": features,
        "axesScore": axes,
        "reliabilityScore": rel,
        "axisReasons": axis_reasons,
        "reliabilityReasons": rel_reasons,
        "benchmark": {
            "label": bench.get("label"),
            "updatedAt": bench.get("updatedAt"),
        }
    }
    audit_text = json.dumps(audit_obj, ensure_ascii=False, indent=2)

    case_id = f"CASE-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"

    payload = {
        "meta": {
            "caseId": case_id,
            "analyzedAt": now_iso(),
            "model": "demo-scoring-asgi-v3",
            "policy": "baseline60-gate60",
        },
        "result": {
            "decision": decision,
            "status": status,
            "confidence": confidence,
            "findings": findings,
            "audit": audit_text,
        },
        "scoring": {
            "baselineScore": BASELINE_SCORE,
            "axesScore": axes,
            "reliabilityScore": rel,
            "axisReasons": axis_reasons,
            "reliabilityReasons": rel_reasons,
            "benchmark": bench,  # 프론트에서 n 같은 건 안 씀
        }
    }
    return JSONResponse(payload)

@app.post("/api/report/pdf")
async def api_report_pdf(payload_json: str = Form(...)):
    try:
        payload = json.loads(payload_json)
    except Exception:
        return Response(content="invalid json", status_code=400, media_type="text/plain")

    pdf_bytes = make_pdf(payload)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=report.pdf"}
    )
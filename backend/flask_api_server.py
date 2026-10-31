
from __future__ import annotations

import os
import sys
import json
import traceback
from datetime import datetime
from typing import Any, Dict

from flask import Flask, request, jsonify, send_file, after_this_request
from flask_cors import CORS

# 평가 시스템 import
from resume_evaluation_system_cross_validation import ResumeEvaluationSystem


app = Flask(__name__)


CORS(app)


UPLOAD_FOLDER = "/tmp/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  

ALLOWED_EXTENSIONS = {".txt", ".pdf", ".docx"} 
system = ResumeEvaluationSystem()

def _safe_remove(path: str) -> None:
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def extract_text_from_file(filepath: str) -> str:
    """파일에서 텍스트 추출"""
    filename = os.path.basename(filepath)
    ext = os.path.splitext(filepath)[1].lower()

    if not ext:
        raise Exception(
            f"파일 확장자가 없습니다: '{filename}'\n지원 형식: .txt, .pdf, .docx"
        )

    if ext == ".txt":
        # 인코딩 자동 처리(UTF-8 -> CP949 -> latin-1)
        for enc in ("utf-8", "cp949", "latin-1"):
            try:
                with open(filepath, "r", encoding=enc) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        raise Exception("TXT 인코딩을 해석할 수 없습니다.")

    if ext == ".pdf":
        try:
            import PyPDF2  # type: ignore
        except Exception:
            raise Exception("PyPDF2가 설치되어 있지 않습니다(requirements.txt 확인).")

        text = ""
        with open(filepath, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            if len(reader.pages) == 0:
                raise Exception("PDF 파일이 비어있습니다.")
            for page in reader.pages:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    text += page_text + "\n"

        if not text.strip():
            raise Exception("PDF에서 텍스트를 추출할 수 없습니다 (이미지 PDF일 수 있음).")
        return text

    if ext == ".docx":
        try:
            import docx  # type: ignore
        except Exception:
            raise Exception("python-docx가 설치되어 있지 않습니다(requirements.txt 확인).")

        doc = docx.Document(filepath)
        text = "\n".join([para.text for para in doc.paragraphs])
        if not text.strip():
            raise Exception("DOCX 파일이 비어있거나 텍스트를 추출할 수 없습니다.")
        return text

    # ext == ".doc" 같은 건 여기로 떨어짐
    raise Exception(
        f"지원하지 않는 파일 형식: '{ext}'\n지원 형식: .txt, .pdf, .docx\n업로드한 파일: '{filename}'"
    )


def transform_to_frontend_format(
    result: Dict[str, Any],
    doc_type: str,
    job_role: str,
    filename: str
) -> Dict[str, Any]:
    """백엔드 결과를 프론트엔드 형식으로 변환"""

    evaluation = result["평가결과"]
    explanation = result.get("설명", {})
    judgment = result["신뢰도검증"]

    # 개별 점수 (0-20점을 0-100점으로 변환)
    scores = evaluation["개별점수"]
    axes_score = {
        "창의성": round(scores["창의성"] * 5, 1),
        "의사소통": round(scores["의사소통"] * 5, 1),
        "전문지식": round(scores["전문지식"] * 5, 1),
        "인성": round(scores["인성"] * 5, 1),
        "열정": round(scores["열정"] * 5, 1),
    }

  
    evidences = evaluation.get("근거", {})
    axis_reasons = {
        "창의성": ", ".join(evidences.get("창의성", [])[:1]) or "근거 부족",
        "의사소통": ", ".join(evidences.get("의사소통", [])[:1]) or "근거 부족",
        "전문지식": ", ".join(evidences.get("전문지식", [])[:1]) or "근거 부족",
        "인성": ", ".join(evidences.get("인성", [])[:1]) or "근거 부족",
        "열정": ", ".join(evidences.get("열정", [])[:1]) or "근거 부족",
    }

    
    trust_scores = judgment["신뢰도평가"]
    reliability_score = {
        "근거정합성": trust_scores["근거정합성"],
        "안정성": trust_scores["안정성"],
        "강건성": trust_scores["강건성"],
        "편향위험": trust_scores["편향위험"],
    }

    reliability_reasons = {
        "근거정합성": "부여된 점수가 실제 이력서에서 추출된 근거 문장에 의해 논리적으로 뒷받침되는 정도",
        "안정성": "동일 이력서를 반복 평가했을 때 핵심 판단 근거/점수의 일관성",
        "강건성": "형식/표현이 달라져도 판단 근거가 인간 기준과 정렬되어 유지되는 정도",
        "편향위험": "직무와 무관한 편향 요소가 근거에 포함되는 위험 신호 수준",
    }

    avg_score = evaluation["평균점수"] * 5  # 0-100 스케일
    grade = evaluation["등급"]

    if "S급" in grade or "A급" in grade:
        status = "ok"
    elif "B급" in grade or "C급" in grade:
        status = "warn"
    else:
        status = "bad"

    confidence = float(judgment.get("종합신뢰도", 0)) / 100.0

    findings = f"""[평가 요약]
• 등급: {grade}
• 평균 점수: {evaluation.get('평균점수','-')}/20점
• 총점: {evaluation.get('총점','-')}/100점

{explanation.get('GPT종합의견', '')}

[신뢰도 검증]
• 신뢰 등급: {judgment.get('신뢰등급','-')}
• 종합 신뢰도: {judgment.get('종합신뢰도','-')}점

{judgment.get('Claude검증의견', '')}"""

    if "교차검증" in judgment:
        cv = judgment["교차검증"]
        findings += f"\n\n[교차 검증]\n• 일치도: {cv.get('일치도','-')}%\n• {cv.get('의견','-')}"

    decision = f"{grade} ({round(avg_score, 1)}점)"

    return {
        "meta": {
            "caseId": f"JHD-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "analyzedAt": evaluation.get("평가일시", datetime.now().isoformat()),
            "filename": filename,
            "docType": doc_type,
            "jobRole": job_role,
            "policy": "JHD 5대 평가기준",
            "model": "GPT-4o + Claude Sonnet 4",
        },
        "result": {
            "decision": decision,
            "confidence": confidence,
            "findings": findings,
            "status": status,
        },
        "scoring": {
            "axesScore": axes_score,
            "reliabilityScore": reliability_score,
            "axisReasons": axis_reasons,
            "reliabilityReasons": reliability_reasons,
            "baselineScore": 60,
            "benchmark": {"axesAvgScore": {k: 60 for k in axes_score.keys()}},
        },
    }

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/api/review", methods=["POST"])
def review():
    """
    form-data:
      - file: 업로드 파일
      - doc_type (optional): resume
      - job_role (optional): developer
    """
    filepath = ""
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        if not file or file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        original_filename = file.filename
        file_ext = os.path.splitext(original_filename)[1].lower()

    
        if file_ext == ".doc":
            return jsonify({
                "error": "구 워드 파일(.doc)은 지원하지 않습니다. .docx로 변환 후 업로드해주세요.",
                "allowed": sorted(list(ALLOWED_EXTENSIONS)),
                "filename": original_filename
            }), 400

        if file_ext not in ALLOWED_EXTENSIONS:
            return jsonify({
                "error": f"Unsupported file type: {file_ext or '(none)'}",
                "allowed": sorted(list(ALLOWED_EXTENSIONS)),
                "filename": original_filename
            }), 400

        doc_type = request.form.get("doc_type", "resume")
        job_role = request.form.get("job_role", "developer")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        safe_filename = f"resume_{timestamp}{file_ext}"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], safe_filename)

        file.save(filepath)
        if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
            _safe_remove(filepath)
            return jsonify({"error": "Failed to save file or empty file"}), 400

        resume_text = extract_text_from_file(filepath)
        if not resume_text or len(resume_text.strip()) < 10:
            _safe_remove(filepath)
            return jsonify({"error": "No meaningful text extracted"}), 400

        result = system.process_resume(resume_text)
        response = transform_to_frontend_format(result, doc_type, job_role, original_filename)

        _safe_remove(filepath)
        return jsonify(response), 200

    except Exception as e:
        _safe_remove(filepath)
        error_msg = f"Review error: {str(e)}"
        print(error_msg, file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        return jsonify({
            "error": error_msg,
        }), 500


@app.route("/api/report/pdf", methods=["POST"])
def generate_pdf():
    """
    1) form-data:
       - payload_json: string(JSON)

    2) application/json:
       {
         "payload": { ... 프론트에서 쓰는 payload ... }
       }
    """
    pdf_path = ""
    try:
        payload = None

   
        payload_json = request.form.get("payload_json")
        if payload_json:
            payload = json.loads(payload_json)

     
        if payload is None:
            data = request.get_json(silent=True) or {}
            payload = data.get("payload")

        if not payload:
            return jsonify({"error": "payload_json (form-data) 또는 payload(JSON)가 필요합니다."}), 400

        # reportlab imports
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont

        import tempfile

        base_dir = os.path.dirname(__file__)
        reg_path = os.path.join(base_dir, "MaruBuri-Regular.ttf")
        bold_path = os.path.join(base_dir, "MaruBuri-Bold.ttf")

        font_reg = "KFont"
        font_bold = "KFont-Bold"

        try:
            if os.path.exists(bold_path):
                pdfmetrics.registerFont(TTFont(font_bold, bold_path))
                if os.path.exists(reg_path):
                    pdfmetrics.registerFont(TTFont(font_reg, reg_path))
                else:
                    font_reg = font_bold
            else:
                font_reg = "Helvetica"
                font_bold = "Helvetica-Bold"
        except Exception:
            font_reg = "Helvetica"
            font_bold = "Helvetica-Bold"

        meta = payload.get("meta", {})
        result = payload.get("result", {})
        scoring = payload.get("scoring", {})

     
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        pdf_path = tmp.name
        tmp.close()

     
        @after_this_request
        def cleanup(response):
            _safe_remove(pdf_path)
            return response

        doc = SimpleDocTemplate(
            pdf_path,
            pagesize=A4,
            leftMargin=14 * mm,
            rightMargin=14 * mm,
            topMargin=14 * mm,
            bottomMargin=14 * mm,
        )

        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name="TitleK", fontName=font_bold, fontSize=18, leading=22, spaceAfter=8, wordWrap="CJK"))
        styles.add(ParagraphStyle(name="H2K", fontName=font_bold, fontSize=12, leading=16, spaceBefore=10, spaceAfter=6, wordWrap="CJK"))
        styles.add(ParagraphStyle(name="BodyK", fontName=font_reg, fontSize=10.5, leading=15, wordWrap="CJK"))
        styles.add(ParagraphStyle(name="SmallK", fontName=font_reg, fontSize=9, leading=12, textColor=colors.HexColor("#475569"), wordWrap="CJK"))

        def p(text: Any, style: str = "BodyK") -> Paragraph:
            s = "" if text is None else str(text)
            s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            s = s.replace("\n", "<br/>")
            return Paragraph(s, styles[style])

        story = []
        story.append(p("JHD 회사 이력서 평가 레포트", "TitleK"))
        story.append(p(f"Case ID: {meta.get('caseId','-')}", "SmallK"))
        story.append(p(f"생성 시간: {meta.get('analyzedAt','-')}", "SmallK"))
        story.append(p(f"이력서: {meta.get('filename','-')}", "SmallK"))
        story.append(Spacer(1, 8))

        story.append(p("총평", "H2K"))
        summary_data = [
            ["총점", result.get("decision", "-")],
            ["백분위", f"{int(round(float(result.get('confidence', 0)) * 100))}%"],
            ["합격여부", result.get("status", "-")],
            ["평가 기준", meta.get("policy", "-")],
            ["생성형 AI 모델", meta.get("model", "-")],
        ]
        summary_tbl = Table(summary_data, colWidths=[32 * mm, 150 * mm], repeatRows=1, splitByRow=1)
        summary_tbl.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), font_reg),
            ("FONTNAME", (0, 0), (0, -1), font_bold),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(summary_tbl)
        story.append(Spacer(1, 10))

        story.append(p("평가 결과", "H2K"))
        story.append(p(result.get("findings", "-"), "BodyK"))
        story.append(Spacer(1, 10))

        axes = scoring.get("axesScore", {})
        axis_reasons = scoring.get("axisReasons", {})
        story.append(p("5대 평가 기준", "H2K"))
        axes_rows = [["항목", "점수", "근거"]]
        for k, v in axes.items():
            axes_rows.append([p(k, "BodyK"), p(f"{v}", "BodyK"), p(axis_reasons.get(k, "-"), "BodyK")])

        axes_tbl = Table(axes_rows, colWidths=[38 * mm, 18 * mm, 126 * mm], repeatRows=1, splitByRow=1)
        axes_tbl.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), font_reg),
            ("FONTNAME", (0, 0), (-1, 0), font_bold),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("FONTSIZE", (0, 0), (-1, -1), 9.5),
            ("WORDWRAP", (0, 0), (-1, -1), "CJK"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(axes_tbl)
        story.append(Spacer(1, 10))

        rel = scoring.get("reliabilityScore", {})
        rel_reasons = scoring.get("reliabilityReasons", {})
        story.append(p("신뢰성 평가", "H2K"))
        rel_rows = [["항목", "점수", "근거"]]
        for k, v in rel.items():
            rel_rows.append([p(k, "BodyK"), p(f"{v}", "BodyK"), p(rel_reasons.get(k, "-"), "BodyK")])

        rel_tbl = Table(rel_rows, colWidths=[38 * mm, 18 * mm, 126 * mm], repeatRows=1, splitByRow=1)
        rel_tbl.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), font_reg),
            ("FONTNAME", (0, 0), (-1, 0), font_bold),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("FONTSIZE", (0, 0), (-1, -1), 9.5),
            ("WORDWRAP", (0, 0), (-1, -1), "CJK"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(rel_tbl)

        doc.build(story)

        return send_file(
            pdf_path,
            mimetype="application/pdf",
            as_attachment=True,
            download_name=f"JHD_report_{meta.get('caseId','CASE')}.pdf",
        )

    except Exception as e:
        _safe_remove(pdf_path)
        error_msg = f"PDF 생성 오류: {str(e)}"
        print(error_msg, file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        return jsonify({
            "error": error_msg,
        }), 500

if __name__ == "__main__":
    print("=" * 60)
    print("JHD 이력서 평가 시스템 API 서버 시작")
    print("=" * 60)
    print("Health check: /health")
    print("=" * 60)

    port = int(os.environ.get("PORT", 5001))  
    app.run(host="0.0.0.0", port=port, debug=False)

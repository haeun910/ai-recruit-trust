

import json
import re
import os
from typing import Dict, List, Tuple
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
import anthropic


load_dotenv()


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
GPT_MODEL = os.getenv("GPT_MODEL", "gpt-4o")

claude_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")


class TheEvaluator:
    """이력서를 JHD 회사 기준으로 평가하는 에이전트"""
    
    def __init__(self):
        self.company_name = "JHD"
        self.evaluation_criteria = {
            "창의성": {
                "description": "문제 해결의 독창성, 새로운 아이디어 제시 능력",
                "keywords": ["혁신", "개선", "제안", "창안", "발명", "특허", "아이디어", "개발", "신규", "독창"]
            },
            "의사소통": {
                "description": "협업 능력, 발표/보고 경험, 대인관계 역량",
                "keywords": ["협업", "팀", "발표", "보고", "소통", "커뮤니케이션", "리더", "조율", "프레젠테이션", "회의"]
            },
            "전문지식": {
                "description": "직무 관련 전문 기술, 자격증, 학력",
                "keywords": ["자격증", "학위", "교육", "기술", "전문", "숙련", "능숙", "프로젝트", "경력", "경험"]
            },
            "인성": {
                "description": "성실성, 책임감, 윤리의식, 긍정적 태도",
                "keywords": ["성실", "책임", "윤리", "정직", "봉사", "배려", "긍정", "열린", "존중", "신뢰"]
            },
            "열정": {
                "description": "직무에 대한 관심과 열의, 자기계발 노력",
                "keywords": ["열정", "관심", "학습", "성장", "도전", "목표", "동기", "열의", "노력", "자발"]
            }
        }
    
    def evaluate_resume(self, resume_text: str) -> Dict:
        """이력서 텍스트를 분석하여 각 평가 기준별 점수 산정"""
        scores = {}
        evidences = {}
        
        for criterion, info in self.evaluation_criteria.items():
            score, evidence = self._calculate_score_with_gpt(resume_text, criterion, info)
            scores[criterion] = score
            evidences[criterion] = evidence
        
        total_score = sum(scores.values())
        average_score = total_score / len(scores)
        
        evaluation_result = {
            "평가일시": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "회사명": self.company_name,
            "개별점수": scores,
            "총점": round(total_score, 2),
            "평균점수": round(average_score, 2),
            "등급": self._get_grade(average_score),
            "근거": evidences
        }
        
        return evaluation_result
    
    def _calculate_score_with_gpt(self, resume_text: str, criterion: str, info: Dict) -> Tuple[float, List[str]]:
        """GPT API를 사용하여 특정 기준에 대한 점수 계산 (0-20점)"""
        
        prompt = f"""다음 이력서를 '{criterion}' 기준으로 평가해주세요.

평가 기준 설명: {info['description']}
평가 시 참고할 키워드: {', '.join(info['keywords'])}
평가 할때는 항목별로 이력서속의 다른 근거를 찾아 평가해주세요.

이력서 내용:
{resume_text}

다음 형식의 JSON으로만 답변해주세요:
{{
    "점수": (0-20 사이의 숫자),
    "근거문장": ["근거가 되는 문장1", "근거가 되는 문장2", "근거가 되는 문장3"]
}}

근거문장은 이력서에서 실제로 나타난 문장 중 해당 기준을 뒷받침하는 내용을 최대 3개까지 추출해주세요.
근거가 부족하면 적게 제시해도 됩니다."""

        try:
            response = client.chat.completions.create(
                model=GPT_MODEL,
                messages=[
                    {"role": "system", "content": "당신은 JHD 회사의 인사 담당자로서 이력서를 평가하는 전문가입니다."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            score = float(result.get("점수", 10))
            evidence = result.get("근거문장", [])
            

            score = max(0, min(20, score))
            
            return round(score, 2), evidence
            
        except Exception as e:
            print(f"GPT API 호출 중 오류 발생 ({criterion}): {e}")
           
            return 10.0, ["평가 중 오류가 발생했습니다."]
    
    def _get_grade(self, average_score: float) -> str:
        """평균 점수를 기반으로 등급 산정"""
        if average_score >= 18:
            return "S급 (최우수)"
        elif average_score >= 15:
            return "A급 (우수)"
        elif average_score >= 12:
            return "B급 (양호)"
        elif average_score >= 9:
            return "C급 (보통)"
        else:
            return "D급 (미흡)"


class TheExplainer:
    """평가 결과에 대한 설명과 시각화를 제공하는 에이전트"""
    
    def __init__(self):
        pass
    
    def explain_evaluation(self, evaluation_result: Dict) -> Dict:
        """GPT API를 사용하여 평가 결과를 분석하고 설명 생성"""
        individual_scores = evaluation_result["개별점수"]
        evidences = evaluation_result["근거"]
        
    
        analysis = self._generate_analysis_with_gpt(evaluation_result)
        

        sorted_scores = sorted(individual_scores.items(), key=lambda x: x[1], reverse=True)
        strengths = [sorted_scores[0], sorted_scores[1]]
        weaknesses = [sorted_scores[-1], sorted_scores[-2]]
        
        explanation = {
            "평가요약": analysis.get("평가요약", self._generate_summary(evaluation_result)),
            "강점분석": self._analyze_strengths(strengths, evidences),
            "약점분석": self._analyze_weaknesses(weaknesses, evidences),
            "상세근거": self._format_evidences(individual_scores, evidences),
            "시각화데이터": self._prepare_visualization_data(individual_scores),
            "GPT종합의견": analysis.get("종합의견", "")
        }
        
        return explanation
    
    def _generate_analysis_with_gpt(self, evaluation_result: Dict) -> Dict:
        """GPT로 평가 결과 종합 분석"""
        
        prompt = f"""다음은 JHD 회사의 이력서 평가 결과입니다.

평가 결과:
- 평균 점수: {evaluation_result['평균점수']}/20점
- 등급: {evaluation_result['등급']}
- 개별 점수: {json.dumps(evaluation_result['개별점수'], ensure_ascii=False)}

각 항목별 근거:
{json.dumps(evaluation_result['근거'], ensure_ascii=False, indent=2)}

이 평가 결과를 바탕으로 다음을 JSON 형식으로 작성해주세요:
{{
    "평가요약": "2-3문장으로 전체 평가를 요약",
    "종합의견": "지원자의 전반적인 강점과 개선점을 3-4문장으로 설명"
}}"""

        try:
            response = client.chat.completions.create(
                model=GPT_MODEL,
                messages=[
                    {"role": "system", "content": "당신은 이력서 평가 결과를 분석하는 HR 전문가입니다."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                response_format={"type": "json_object"}
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            print(f"GPT API 호출 중 오류 발생 (분석): {e}")
            return {}
    
    def _generate_summary(self, evaluation_result: Dict) -> str:
        """전체 평가 요약 생성"""
        grade = evaluation_result["등급"]
        avg_score = evaluation_result["평균점수"]
        
        summary = f"""
[평가 결과 요약]
• 종합 등급: {grade}
• 평균 점수: {avg_score}/20점
• 총점: {evaluation_result["총점"]}/100점

이 지원자는 JHD 회사의 인재상과 {self._get_fit_level(avg_score)} 수준의 적합도를 보입니다.
"""
        return summary.strip()
    
    def _get_fit_level(self, avg_score: float) -> str:
        """적합도 수준 판단"""
        if avg_score >= 15:
            return "매우 높은"
        elif avg_score >= 12:
            return "높은"
        elif avg_score >= 9:
            return "중간"
        else:
            return "개선이 필요한"
    
    def _analyze_strengths(self, strengths: List[Tuple[str, float]], evidences: Dict) -> Dict:
        """강점 분석"""
        strength_analysis = {}
        
        for criterion, score in strengths:
            strength_analysis[criterion] = {
                "점수": score,
                "평가": "강점 영역" if score >= 15 else "양호 영역",
                "근거문장": evidences.get(criterion, [])[:2]
            }
        
        return strength_analysis
    
    def _analyze_weaknesses(self, weaknesses: List[Tuple[str, float]], evidences: Dict) -> Dict:
        """약점 분석"""
        weakness_analysis = {}
        
        for criterion, score in weaknesses:
            weakness_analysis[criterion] = {
                "점수": score,
                "평가": "개선 필요" if score < 10 else "보완 권장",
                "개선방안": self._suggest_improvement(criterion),
                "근거부족": "구체적 경험 서술 부족" if len(evidences.get(criterion, [])) < 2 else "키워드 빈도 낮음"
            }
        
        return weakness_analysis
    
    def _suggest_improvement(self, criterion: str) -> str:
        """기준별 개선 방안 제안"""
        suggestions = {
            "창의성": "혁신적 문제해결 사례, 개선 제안 경험, 특허/발명 등을 구체적으로 기술",
            "의사소통": "팀 프로젝트, 발표 경험, 협업 성과 등을 수치와 함께 명시",
            "전문지식": "직무 관련 자격증, 기술 스택, 프로젝트 경험을 상세히 기재",
            "인성": "봉사활동, 리더십 경험, 책임감을 보여준 사례 추가",
            "열정": "자기계발 노력, 학습 이력, 직무 관련 활동 등을 구체화"
        }
        return suggestions.get(criterion, "관련 경험과 역량을 보다 구체적으로 작성")
    
    def _format_evidences(self, scores: Dict, evidences: Dict) -> Dict:
        """근거 문장을 체계적으로 정리"""
        formatted = {}
        
        for criterion in scores.keys():
            formatted[criterion] = {
                "점수": scores[criterion],
                "근거문장수": len(evidences.get(criterion, [])),
                "추출된근거": evidences.get(criterion, [])
            }
        
        return formatted
    
    def _prepare_visualization_data(self, scores: Dict) -> Dict:
        """시각화를 위한 데이터 준비"""
        viz_data = {
            "레이더차트": {
                "기준": list(scores.keys()),
                "점수": list(scores.values()),
                "최대값": 20
            },
            "막대그래프": [
                {"기준": k, "점수": v, "달성률": f"{(v/20)*100:.1f}%"}
                for k, v in scores.items()
            ]
        }
        
        return viz_data


class LLMJudge:
    """평가 결과의 신뢰도를 검증하는 심사 에이전트"""
    
    def __init__(self):
        self.trust_criteria = ["근거정합성", "안정성", "강건성", "편향위험"]
    
    def judge_evaluation(self, evaluation_result: Dict, explanation: Dict) -> Dict:
        """Claude API를 사용하여 평가 결과의 신뢰도 판단"""
        
     
        claude_judgment = self._judge_with_claude(evaluation_result, explanation)
        
    
        trust_scores = {
            "근거정합성": self._check_evidence_consistency(evaluation_result, explanation),
            "안정성": self._check_stability(evaluation_result),
            "강건성": self._check_robustness(evaluation_result),
            "편향위험": self._check_bias_risk(evaluation_result)
        }
        
        overall_trust = sum(trust_scores.values()) / len(trust_scores)
        
        judgment = {
            "신뢰도평가": trust_scores,
            "종합신뢰도": round(overall_trust, 2),
            "신뢰등급": self._get_trust_grade(overall_trust),
            "주의사항": self._generate_warnings(trust_scores),
            "권고사항": self._generate_recommendations(trust_scores),
            "Claude검증의견": claude_judgment.get("검증의견", ""),
            "교차검증": self._cross_validate(trust_scores, claude_judgment)
        }
        
        return judgment
    
    def _judge_with_claude(self, evaluation_result: Dict, explanation: Dict) -> Dict:
        """Claude API로 신뢰도 검증"""
        
        prompt = f"""다음은 이력서 평가 결과입니다. 이 평가의 신뢰도를 검증해주세요.

평가 결과:
{json.dumps(evaluation_result, ensure_ascii=False, indent=2)}

설명:
{json.dumps(explanation, ensure_ascii=False, indent=2)}
TotalTrust = (w1·EDA + w2·ES + w3·HA) × (1 - BRS)
다음 관점에서 평가의 신뢰도를 분석하고 JSON으로 답변해주세요:
1. 근거정합성: 부여된 점수가 실제 이력서에서 추출된 근거 문장에 의해 논리적으로 뒷받침되는 정도를 평가합니다
2. 안정성": 동일한 이력서를 반복 평가했을 때 AI의 핵심 판단 근거와 점수가 얼마나 일관되게 유지되는지를 측정합니다
3. 강건성": 이력서의 형식이나 표현이 변경되더라도 AI의 판단 근거가 인간 기준과 정렬되어 유지되는지를 평가합니다
4. 편향위험": AI 평가 근거에 직무와 무관한 편향 요소가 포함되어 있는지를 위험 신호 수준으로 검사합니다


JSON 형식:
{{
    "검증의견": "평가의 신뢰도에 대한 종합 의견 (3-4문장)",
    "신뢰도점수": {{
        "근거정합성": 0-100 사이 숫자,
        "안정성": 0-100 사이 숫자,
        "강건성": 0-100 사이 숫자,
        "편향위험": 0-100 사이 숫자
    }}
}}"""

        try:
            response = claude_client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=2000,
                temperature=0.3,
                system="당신은 평가 결과의 신뢰도를 검증하는 메타 평가자입니다. 반드시 JSON 형식으로만 답변하세요.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
 
            response_text = response.content[0].text
            
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                return json.loads(json_match.group())
            else:
                return json.loads(response_text)
            
        except Exception as e:
            print(f"Claude API 호출 중 오류 발생 (검증): {e}")
            return {
                "검증의견": "Claude API 호출 중 오류가 발생했습니다.",
                "신뢰도점수": {}
            }
    
    def _cross_validate(self, rule_based_scores: Dict, claude_judgment: Dict) -> Dict:
        """규칙 기반 점수와 Claude 검증 결과를 교차 검증"""
        
        claude_scores = claude_judgment.get("신뢰도점수", {})
        
        if not claude_scores:
            return {
                "교차검증상태": "Claude 검증 실패",
                "일치도": 0,
                "의견": "Claude API 오류로 교차 검증을 수행할 수 없습니다."
            }
        
   
        differences = {}
        total_diff = 0
        count = 0
        
        for criterion in rule_based_scores.keys():
            if criterion in claude_scores:
                rule_score = rule_based_scores[criterion]
                claude_score = claude_scores[criterion]
                diff = abs(rule_score - claude_score)
                differences[criterion] = {
                    "규칙기반": rule_score,
                    "Claude평가": claude_score,
                    "차이": round(diff, 2)
                }
                total_diff += diff
                count += 1
        
   
        avg_diff = total_diff / count if count > 0 else 0
        agreement = max(0, 100 - avg_diff) 
        
       
        if agreement >= 90:
            validation_status = "높은 일치도"
            opinion = "규칙 기반 평가와 Claude의 판단이 매우 일치합니다. 신뢰도가 높습니다."
        elif agreement >= 80:
            validation_status = "양호한 일치도"
            opinion = "대체로 일치하나 일부 항목에서 차이가 있습니다. 전반적으로 신뢰할 만합니다."
        elif agreement >= 70:
            validation_status = "보통 일치도"
            opinion = "규칙 기반 평가와 Claude의 판단에 일정한 차이가 있습니다. 재검토를 권장합니다."
        else:
            validation_status = "낮은 일치도"
            opinion = "규칙 기반 평가와 Claude의 판단이 크게 다릅니다. 평가 기준을 재조정할 필요가 있습니다."
        
        return {
            "교차검증상태": validation_status,
            "일치도": round(agreement, 2),
            "항목별차이": differences,
            "의견": opinion
        }
    
    def _check_evidence_consistency(self, evaluation_result: Dict, explanation: Dict) -> float:
        """근거 정합성 검사: 점수와 근거 문장이 일치하는가"""
        scores = evaluation_result["개별점수"]
        evidences = evaluation_result["근거"]
        
        consistency_score = 0
        total_checks = 0
        
        for criterion, score in scores.items():
            evidence_count = len(evidences.get(criterion, []))
            total_checks += 1
            
   
            if score >= 15 and evidence_count >= 2:
                consistency_score += 1
            elif 10 <= score < 15 and evidence_count >= 1:
                consistency_score += 1
            elif score < 10 and evidence_count <= 1:
                consistency_score += 1
            else:
                consistency_score += 0.5
        
        return round((consistency_score / total_checks) * 100, 2)
    
    def _check_stability(self, evaluation_result: Dict) -> float:
        """안정성 검사: 점수 분포가 고르고 극단적이지 않은가"""
        scores = list(evaluation_result["개별점수"].values())
        
      
        mean_score = sum(scores) / len(scores)
        variance = sum((x - mean_score) ** 2 for x in scores) / len(scores)
        std_dev = variance ** 0.5
        
      
        if std_dev < 3:
            stability = 95
        elif std_dev < 5:
            stability = 80
        elif std_dev < 7:
            stability = 65
        else:
            stability = 50
        
        return round(stability, 2)
    
    def _check_robustness(self, evaluation_result: Dict) -> float:
        """강건성 검사: 다양한 기준에서 근거가 존재하는가"""
        evidences = evaluation_result["근거"]
        
    
        criteria_with_evidence = sum(1 for e in evidences.values() if len(e) > 0)
        total_criteria = len(evidences)
        
        coverage = (criteria_with_evidence / total_criteria) * 100
        

        total_evidence_count = sum(len(e) for e in evidences.values())
        quality_bonus = min(total_evidence_count * 2, 20)
        
        robustness = min(coverage * 0.8 + quality_bonus, 100)
        
        return round(robustness, 2)
    
    def _check_bias_risk(self, evaluation_result: Dict) -> float:
        """편향 위험 검사: 특정 기준에 과도하게 치우치지 않았는가"""
        scores = list(evaluation_result["개별점수"].values())
        
        max_score = max(scores)
        min_score = min(scores)
        score_range = max_score - min_score
        
      
        if score_range < 5:
            bias_risk = 10 
        elif score_range < 8:
            bias_risk = 25
        elif score_range < 12:
            bias_risk = 40
        else:
            bias_risk = 60  
  
        trust_score = 100 - bias_risk
        
        return round(trust_score, 2)
    
    def _get_trust_grade(self, overall_trust: float) -> str:
        """신뢰도 등급 산정"""
        if overall_trust >= 90:
            return "매우 신뢰 가능"
        elif overall_trust >= 80:
            return "신뢰 가능"
        elif overall_trust >= 70:
            return "보통"
        elif overall_trust >= 60:
            return "주의 필요"
        else:
            return "재평가 권장"
    
    def _generate_warnings(self, trust_scores: Dict) -> List[str]:
        """신뢰도 점수 기반 주의사항 생성"""
        warnings = []
        
        if trust_scores["근거정합성"] < 70:
            warnings.append("⚠️ 일부 평가 점수와 근거 문장의 정합성이 낮습니다")
        
        if trust_scores["안정성"] < 70:
            warnings.append("⚠️ 평가 항목 간 점수 편차가 큽니다")
        
        if trust_scores["강건성"] < 70:
            warnings.append("⚠️ 일부 평가 기준에서 근거가 부족합니다")
        
        if trust_scores["편향위험"] < 70:
            warnings.append("⚠️ 특정 기준에 과도하게 치우친 평가 가능성이 있습니다")
        
        if not warnings:
            warnings.append("✓ 특이사항 없음")
        
        return warnings
    
    def _generate_recommendations(self, trust_scores: Dict) -> List[str]:
        """신뢰도 향상을 위한 권고사항 생성"""
        recommendations = []
        
        if trust_scores["근거정합성"] < 80:
            recommendations.append("• 각 평가 점수에 대응하는 구체적 근거를 추가로 확보하세요")
        
        if trust_scores["안정성"] < 80:
            recommendations.append("• 극단적 점수의 타당성을 재검토하세요")
        
        if trust_scores["강건성"] < 80:
            recommendations.append("• 모든 평가 기준에서 최소 2개 이상의 근거를 확보하세요")
        
        if trust_scores["편향위험"] < 80:
            recommendations.append("• 평가 기준 간 균형을 재조정하세요")
        
        if not recommendations:
            recommendations.append("• 현재 평가는 신뢰할 수 있는 수준입니다")
        
        return recommendations


class ResumeEvaluationSystem:
    """전체 이력서 평가 시스템 통합 클래스"""
    
    def __init__(self):
        self.evaluator = TheEvaluator()
        self.explainer = TheExplainer()
        self.judge = LLMJudge()
    
    def process_resume(self, resume_text: str) -> Dict:
        """이력서 평가 전체 프로세스 실행"""
        
        print("=" * 60)
        print("JHD 회사 이력서 평가 시스템 시작 (GPT API 사용)")
        print("=" * 60)
        
        # Step 1: 평가 수행
        print("\n[1단계] The Evaluator - 이력서 평가 중...")
        evaluation_result = self.evaluator.evaluate_resume(resume_text)
        print("✓ 평가 완료")
        
        # Step 2: 설명 생성
        print("\n[2단계] The Explainer - 평가 결과 분석 중...")
        explanation = self.explainer.explain_evaluation(evaluation_result)
        print("✓ 분석 완료")
        
        # Step 3: 신뢰도 검증
        print("\n[3단계] LLM Judge - 신뢰도 검증 중...")
        judgment = self.judge.judge_evaluation(evaluation_result, explanation)
        print("✓ 검증 완료")
        
   
        final_result = {
            "평가결과": evaluation_result,
            "설명": explanation,
            "신뢰도검증": judgment
        }
        
        return final_result
    
    def print_result(self, final_result: Dict):
        """결과를 보기 좋게 출력"""
        print("\n" + "=" * 60)
        print("평가 결과 리포트")
        print("=" * 60)
        
        # 1. 기본 정보
        eval_result = final_result["평가결과"]
        print(f"\n📋 평가 일시: {eval_result['평가일시']}")
        print(f"🏢 회사명: {eval_result['회사명']}")
        print(f"🏆 종합 등급: {eval_result['등급']}")
        print(f"📊 평균 점수: {eval_result['평균점수']}/20점")
        print(f"💯 총점: {eval_result['총점']}/100점")
        
        # 2. 개별 점수
        print("\n📈 개별 평가 점수:")
        for criterion, score in eval_result["개별점수"].items():
            bar = "█" * int(score) + "░" * (20 - int(score))
            print(f"  {criterion:8s}: {bar} {score:5.2f}/20")
        
        # 3. 강점/약점
        explanation = final_result["설명"]
        print("\n💪 강점 분석:")
        for criterion, analysis in explanation["강점분석"].items():
            print(f"  • {criterion}: {analysis['점수']}점 - {analysis['평가']}")
        
        print("\n📉 약점 분석:")
        for criterion, analysis in explanation["약점분석"].items():
            print(f"  • {criterion}: {analysis['점수']}점 - {analysis['평가']}")
            print(f"    → {analysis['개선방안']}")
        
 
        if explanation.get("GPT종합의견"):
            print(f"\n💬 GPT 종합 의견:\n{explanation['GPT종합의견']}")
        
  
        judgment = final_result["신뢰도검증"]
        print(f"\n🔍 신뢰도 검증 결과: {judgment['신뢰등급']} (종합 {judgment['종합신뢰도']}점)")
        
        print("\n  세부 신뢰도 (규칙 기반):")
        for criterion, score in judgment["신뢰도평가"].items():
            status = "✓" if score >= 80 else "⚠" if score >= 70 else "✗"
            print(f"    {status} {criterion:10s}: {score:5.2f}점")
        
        print("\n  주의사항:")
        for warning in judgment["주의사항"]:
            print(f"    {warning}")
        
        print("\n  권고사항:")
        for recommendation in judgment["권고사항"]:
            print(f"    {recommendation}")
        
    
        if judgment.get("Claude검증의견"):
            print(f"\n🤖 Claude 검증 의견:\n{judgment['Claude검증의견']}")
        
       
        if "교차검증" in judgment:
            cross_val = judgment["교차검증"]
            print(f"\n🔄 교차 검증 (규칙 기반 vs Claude):")
            print(f"  상태: {cross_val['교차검증상태']}")
            print(f"  일치도: {cross_val['일치도']}%")
            print(f"  의견: {cross_val['의견']}")
            
            if "항목별차이" in cross_val:
                print("\n  항목별 비교:")
                for criterion, diff_data in cross_val["항목별차이"].items():
                    print(f"    • {criterion}: 규칙={diff_data['규칙기반']:.1f} / Claude={diff_data['Claude평가']:.1f} / 차이={diff_data['차이']:.1f}")
        
        print("\n" + "=" * 60)


def main():
    """메인 실행 함수"""
    
   
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    
    if not openai_key or openai_key == "your-api-key-here":
        print("⚠️ 오류: OPENAI_API_KEY가 설정되지 않았습니다.")
        print("1. .env 파일을 생성하세요")
        print("2. OPENAI_API_KEY=your-actual-api-key 를 입력하세요")
        print("3. 다시 실행하세요")
        return None
    
    if not anthropic_key or anthropic_key == "your-anthropic-api-key-here":
        print("⚠️ 오류: ANTHROPIC_API_KEY가 설정되지 않았습니다.")
        print("1. .env 파일을 생성하세요")
        print("2. ANTHROPIC_API_KEY=your-actual-api-key 를 입력하세요")
        print("3. 다시 실행하세요")
        return None
    
    print(f"✓ OpenAI API 키 확인됨 (GPT 사용: Evaluator, Explainer)")
    print(f"✓ Anthropic API 키 확인됨 (Claude 사용: LLM Judge)")
    print(f"✓ GPT 모델: {GPT_MODEL}")
    print(f"✓ Claude 모델: {CLAUDE_MODEL}")
    
    
    sample_resume = """
    [이력서]
    
    이름: 김지원
    
    학력:
    - 서울대학교 컴퓨터공학과 학사 졸업 (2020)
    - 성적: 3.8/4.5
    
    경력:
    - ABC 테크 소프트웨어 엔지니어 (2020-2023)
      * 신규 추천 알고리즘 개발로 사용자 만족도 25% 향상
      * 팀 프로젝트 리더로 5명의 팀원과 협업하여 프로젝트 성공적 완수
      * 매주 팀 회의 주도 및 경영진 대상 월간 보고서 작성
    
    프로젝트:
    - 머신러닝 기반 이미지 분류 시스템 개발
      * Python, TensorFlow 활용
      * 정확도 92% 달성
    - 사내 업무 자동화 도구 개발
      * 반복 작업 시간 40% 단축
      * 혁신상 수상
    
    자격증:
    - 정보처리기사
    - AWS Certified Solutions Architect
    
    활동:
    - 오픈소스 프로젝트 기여 (GitHub 활동)
    - 기술 블로그 운영 (월 2회 포스팅)
    - 대학 멘토링 프로그램 참여
    
    자기소개:
    저는 새로운 기술을 배우는 것을 즐기며, 항상 더 나은 솔루션을 찾기 위해 노력합니다.
    팀원들과의 원활한 소통을 중시하며, 책임감 있게 업무를 수행합니다.
    """
    

    system = ResumeEvaluationSystem()
    
 
    result = system.process_resume(sample_resume)
    

    system.print_result(result)
    
  
    output_file = "/home/claude/evaluation_result_gpt.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 상세 결과가 '{output_file}' 파일로 저장되었습니다.")
    
    return result


if __name__ == "__main__":
    result = main()

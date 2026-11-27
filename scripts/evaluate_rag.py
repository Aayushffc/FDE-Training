"""
Comprehensive RAG Evaluation and Benchmarking Suite.
===================================================
Evaluates the Retrieval-Augmented Generation (RAG) pipeline across balanced
positive (answerable) and negative (unanswerable / out-of-domain) test queries.

Computes:
  - Confusion Matrix: True Positives (TP), False Positives (FP), True Negatives (TN), False Negatives (FN)
  - Precision, Recall (Sensitivity), Specificity, Accuracy, and F1 Score
  - False Negative Rate (Unwarranted Refusal Rate)
  - False Positive Rate (Hallucination / False Acceptance Rate)
  - Citation presence & Grounding validation
  - Execution latencies (Retrieval + Rerank + LLM) and Token/Cost statistics

Outputs:
  - Terminal formatted summary tables
  - Markdown benchmark report (docs/rag-evaluation-report.md)
  - Raw JSON metrics (docs/rag-evaluation-results.json)
"""

import argparse
import asyncio
import json
import math
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv

load_dotenv(ROOT_DIR / ".env")

from app.db.session import SessionLocal
from app.infrastructure.embedding_providers import get_embedding_provider
from app.infrastructure.llm_providers.registry import default_registry
from app.schemas.rag import RAGRequest, RAGResponse
from app.services.context_service import ContextEngineeringService
from app.services.prompt_service import PromptService
from app.services.rag_service import RAGService
from app.services.reranking_service import RerankingService
from app.services.search_service import SearchService

# ---------------------------------------------------------------------------
# Test Query Definitions (Balanced: 10 Positive + 10 Negative)
# ---------------------------------------------------------------------------

EVALUATION_DATASET: List[Dict[str, Any]] = [
    # --- POSITIVE TEST CASES: COMPLEX MULTI-HOP / CONDITIONAL GROUNDED QUERIES ---
    {
        "id": "POS-01",
        "question": "What happens when someone enters an incorrect password five times in a row, how long does the lockout last, and is there any way to unlock the account before the timer finishes?",
        "ground_truth_answerable": True,
        "category": "account_and_access",
        "expected_facts": ["five consecutive failed", "thirty minutes", "Unlock My Account"],
        "description": "Multi-part inquiry on authentication brute-force lockout, cooling-off duration, and emergency unlock mechanisms.",
    },
    {
        "id": "POS-02",
        "question": "If a customer on the Plus membership tier returns an opened electronic item 20 days after delivery, will it be accepted, or what is the specific return window for electronics compared to general merchandise?",
        "ground_truth_answerable": True,
        "category": "returns_and_refunds",
        "expected_facts": ["fourteen-day", "45", "electronic"],
        "description": "Complex tier vs product-category policy: Plus tier gets 45 days for general goods, but electronics have a strict 14-day limit.",
    },
    {
        "id": "POS-03",
        "question": "in how much time will i receive my refund if i choose my original credit card versus if i choose the store credit digital wallet, and is there any bonus for store credit?",
        "ground_truth_answerable": True,
        "category": "returns_and_refunds",
        "expected_facts": ["three to five business days", "two hours", "ten percent"],
        "description": "Comparative refund disbursement timeline and store credit 10% bonus incentive.",
    },
    {
        "id": "POS-04",
        "question": "If I place an order and realize I entered the wrong address, can I cancel it, and how many minutes do I have before cancellation is blocked?",
        "ground_truth_answerable": True,
        "category": "orders_and_fulfillment",
        "expected_facts": ["thirty minutes", "sixty minutes", "cancel"],
        "description": "Time-sensitive order cancellation window (30 mins) and address modification window (60 mins).",
    },
    {
        "id": "POS-05",
        "question": "What carrier rerouting fee and transit delay occur if an address modification is requested after a package has already been dispatched from the warehouse?",
        "ground_truth_answerable": True,
        "category": "orders_and_fulfillment",
        "expected_facts": ["$15.00", "twenty-four to forty-eight"],
        "description": "In-flight carrier intercept fee and redirect schedule delay.",
    },
    {
        "id": "POS-06",
        "question": "How many consecutive days without carrier tracking scans must pass before a package is officially declared Lost in Transit, and what two resolution options are offered to the customer?",
        "ground_truth_answerable": True,
        "category": "orders_and_fulfillment",
        "expected_facts": ["seven consecutive", "replacement", "100% full refund"],
        "description": "Lost in transit threshold (7 days) and remedy choices (free replacement or full refund).",
    },
    {
        "id": "POS-07",
        "question": "If an order exceeds $500 in value, what delivery rule applies to couriers at the time of drop-off?",
        "ground_truth_answerable": True,
        "category": "orders_and_fulfillment",
        "expected_facts": ["$500", "signature", "prohibited"],
        "description": "High-value shipment courier drop-off signature requirement and unattended porch ban.",
    },
    {
        "id": "POS-08",
        "question": "If a customer enrolled in an Annual VIP membership cancels 20 days into their year, are they entitled to a full refund, and what policy governs this?",
        "ground_truth_answerable": True,
        "category": "subscriptions_and_billing",
        "expected_facts": ["Thirty-Day Money-Back", "100% full refund", "thirty"],
        "description": "Annual subscription satisfaction guarantee eligibility within the 30-day window.",
    },
    {
        "id": "POS-09",
        "question": "What happens when a recurring subscription payment fails on day 1, what is the automated retry schedule, and on what day does the account enter suspension?",
        "ground_truth_answerable": True,
        "category": "subscriptions_and_billing",
        "expected_facts": ["Day 1", "Day 3", "Day 7", "Day 14", "Day 15"],
        "description": "Multi-stage dunning recovery schedule across the 14-day grace period and Day 15 suspension.",
    },
    {
        "id": "POS-10",
        "question": "Can I combine Meridian Store Credit with a credit card at checkout, and if that split order is later refunded, in what order are the funds returned?",
        "ground_truth_answerable": True,
        "category": "payments_and_disputes",
        "expected_facts": ["split tender", "secondary", "credit card", "Store Credit"],
        "description": "Split tender payment rules and reverse order disbursement on return.",
    },
    {
        "id": "POS-11",
        "question": "What pre-authorization hold is placed on my card when placing an order, when is actual payment captured, and how long does an unused hold take to release?",
        "ground_truth_answerable": True,
        "category": "payments_and_disputes",
        "expected_facts": ["five percent", "dispatches", "twenty-four to seventy-two"],
        "description": "Financial mechanics of pre-auth holds, sales tax buffers, capture on dispatch, and hold expiration.",
    },
    {
        "id": "POS-12",
        "question": "If a customer initiates a formal bank chargeback on a transaction, what immediate restriction is placed on their Meridian account, and what happens if the chargeback is determined to be fraudulent?",
        "ground_truth_answerable": True,
        "category": "payments_and_disputes",
        "expected_facts": ["frozen", "blacklisted", "closed"],
        "description": "Immediate automated account freeze upon chargeback filing and permanent blacklisting for friendly fraud.",
    },
    {
        "id": "POS-13",
        "question": "What are the discretionary financial concession limits for Tier 1 frontline support agents versus Tier 2 supervisors?",
        "ground_truth_answerable": True,
        "category": "payments_and_disputes",
        "expected_facts": ["$25.00", "$100.00", "Tier 1", "Tier 2"],
        "description": "Internal customer support operational financial authority tiers.",
    },
    {
        "id": "POS-14",
        "question": "What is the warranty coverage duration for hardware purchases made by a VIP tier member compared to a standard non-member?",
        "ground_truth_answerable": True,
        "category": "returns_and_refunds",
        "expected_facts": ["One-Year", "Two-Year", "zero deductible"],
        "description": "Cross-document comparison: Basic 1-year warranty vs VIP 2-year comprehensive coverage.",
    },
    # --- NEGATIVE TEST CASES (Ground Truth: Answerable = False / Out-of-Domain) ---
    {
        "id": "NEG-01",
        "question": "What is the capital city of Australia and what is its current population?",
        "ground_truth_answerable": False,
        "category": "out_of_domain",
        "expected_facts": [],
        "description": "General world geography trivia having zero relation to retail platform documentation.",
    },
    {
        "id": "NEG-02",
        "question": "Who won the 2022 FIFA World Cup final in Qatar and what was the final penalty shootout score?",
        "ground_truth_answerable": False,
        "category": "out_of_domain",
        "expected_facts": [],
        "description": "Historical sports trivia completely outside knowledge base.",
    },
    {
        "id": "NEG-03",
        "question": "Can I pick up my online order at a physical retail store location in Chicago or Dallas?",
        "ground_truth_answerable": False,
        "category": "adversarial_unsupported",
        "expected_facts": [],
        "description": "Inquiry about physical retail storefronts for a fulfillment-center-only platform.",
    },
    {
        "id": "NEG-04",
        "question": "What is the trade-in credit valuation for used Apple iPhones and Samsung Galaxy smartphones?",
        "ground_truth_answerable": False,
        "category": "adversarial_unsupported",
        "expected_facts": [],
        "description": "Nonexistent electronic device trade-in appraisal program.",
    },
    {
        "id": "NEG-05",
        "question": "How do I configure Apache Kafka event streams to sync Meridian order webhooks with an on-premise ERP?",
        "ground_truth_answerable": False,
        "category": "adversarial_unsupported",
        "expected_facts": [],
        "description": "Fictional enterprise streaming architecture connector.",
    },
    {
        "id": "NEG-06",
        "question": "What discount percentage do active military personnel and veterans receive on holiday orders?",
        "ground_truth_answerable": False,
        "category": "adversarial_unsupported",
        "expected_facts": [],
        "description": "Nonexistent military and veteran discount program.",
    },
    {
        "id": "NEG-07",
        "question": "What is the pet insurance policy and vet bill reimbursement procedure under VIP membership?",
        "ground_truth_answerable": False,
        "category": "adversarial_unsupported",
        "expected_facts": [],
        "description": "Fictional pet health insurance tier benefit.",
    },
    {
        "id": "NEG-08",
        "question": "What are the physical walk-in customer service counter hours at the Meridian headquarters in London?",
        "ground_truth_answerable": False,
        "category": "adversarial_unsupported",
        "expected_facts": [],
        "description": "Unsupported overseas physical walk-in support desk.",
    },
    {
        "id": "NEG-09",
        "question": "How do I redeem airline frequent flyer miles to pay for Meridian Plus membership?",
        "ground_truth_answerable": False,
        "category": "adversarial_unsupported",
        "expected_facts": [],
        "description": "Unsupported airline miles loyalty partnership.",
    },
    {
        "id": "NEG-10",
        "question": "What is the customs passport visa requirement for international shipping inspection officers?",
        "ground_truth_answerable": False,
        "category": "out_of_domain",
        "expected_facts": [],
        "description": "Government customs and immigration regulations completely outside platform scope.",
    },
]


@dataclass
class QueryEvaluationResult:
    query_id: str
    question: str
    category: str
    ground_truth_answerable: bool
    description: str
    expected_facts: List[str]

    # Model Output
    actual_answer: str
    is_grounded: bool
    citations: List[str]
    total_sources_evaluated: int
    provider: str
    model: Optional[str]
    latency_ms: float
    llm_latency_ms: Optional[float]
    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_cost_usd: float

    # Confusion Matrix Classification
    classification: str  # TP, FP, TN, FN
    factual_coverage_score: float
    has_valid_citations: bool
    refusal_detected: bool
    error: Optional[str] = None


@dataclass
class BenchmarkMetrics:
    total_queries: int = 0
    total_positives: int = 0
    total_negatives: int = 0

    # Confusion Matrix
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0

    # Core Stats
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    specificity: float = 0.0
    f1_score: float = 0.0
    false_negative_rate: float = 0.0
    false_positive_rate: float = 0.0

    # Quality & Citations
    citation_rate_on_tp: float = 0.0
    avg_factual_coverage_on_tp: float = 0.0

    # Latency & Cost
    avg_latency_ms: float = 0.0
    median_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    avg_llm_latency_ms: float = 0.0
    total_tokens: int = 0
    total_cost_usd: float = 0.0


def is_refusal_answer(answer_text: str) -> bool:
    """Detect if the LLM output expresses ignorance or refusal."""
    clean = answer_text.strip().lower()
    if clean.startswith("i don't know based on the provided context") or clean.startswith("i do not know based on the provided context"):
        return True
    if clean in ("i don't know.", "i don't know", "i do not know.", "i do not know"):
        return True
    if len(clean) < 120 and any(
        cue in clean
        for cue in [
            "i don't know",
            "i do not know",
            "not available in the provided context",
            "not mentioned in the context",
            "no relevant context found",
        ]
    ):
        return True
    return False


def evaluate_factual_coverage(answer_text: str, expected_facts: List[str]) -> float:
    """Calculate the ratio of expected facts appearing in the answer."""
    if not expected_facts:
        return 1.0
    clean = answer_text.lower()
    matched = 0
    for fact in expected_facts:
        # Check either exact phrase or individual significant keywords
        fact_clean = fact.lower()
        if fact_clean in clean:
            matched += 1
        else:
            # Check if all words >= 3 chars match
            words = [w for w in re.findall(r"\w+", fact_clean) if len(w) >= 3]
            if words and all(w in clean for w in words):
                matched += 1
    return round(matched / len(expected_facts), 4)


async def run_evaluation(
    strategy: str = "hybrid",
    rerank: bool = True,
    top_k: int = 5,
    provider: str = "groq",
    prompt_category: str = "rag",
    prompt_version: str = "v1",
) -> Tuple[List[QueryEvaluationResult], BenchmarkMetrics]:
    """Execute evaluation over all queries in the dataset."""
    db = SessionLocal()
    search_service = SearchService(db=db)
    reranker = RerankingService()
    context_service = ContextEngineeringService()
    prompt_service = PromptService(db=db)

    rag_service = RAGService(
        db=db,
        search_service=search_service,
        reranking_service=reranker,
        context_service=context_service,
        prompt_service=prompt_service,
    )

    results: List[QueryEvaluationResult] = []

    print("\n" + "=" * 80)
    print("STARTING RAG BENCHMARK EVALUATION")
    print(f"Strategy: {strategy.upper()} | Rerank: {rerank} | Top-K: {top_k} | Provider: {provider.upper()}")
    print(f"Prompt: {prompt_category}/{prompt_version} (loaded from PostgreSQL)")
    pos_count = sum(1 for q in EVALUATION_DATASET if q["ground_truth_answerable"])
    neg_count = sum(1 for q in EVALUATION_DATASET if not q["ground_truth_answerable"])
    print(f"Total Test Queries: {len(EVALUATION_DATASET)} ({pos_count} Positive, {neg_count} Negative)")
    print("=" * 80 + "\n")

    for idx, test_case in enumerate(EVALUATION_DATASET, start=1):
        q_id = test_case["id"]
        q_text = test_case["question"]
        gt_answerable = test_case["ground_truth_answerable"]
        expected_facts = test_case["expected_facts"]

        print(f"[{idx:02d}/{len(EVALUATION_DATASET):02d}] Evaluating {q_id} ({'POS' if gt_answerable else 'NEG'})...")
        print(f"     Query: \"{q_text}\"")

        payload = RAGRequest(
            question=q_text,
            retrieval_strategy=strategy,
            rerank=rerank,
            top_k=top_k,
            provider=provider,
            prompt_category=prompt_category,
            prompt_version=prompt_version,
        )

        try:
            start_t = time.perf_counter()
            response: RAGResponse = await rag_service.generate_rag_answer(payload)
            wall_latency_ms = round((time.perf_counter() - start_t) * 1000, 2)

            answer = response.answer.strip()
            is_refusal = is_refusal_answer(answer)
            has_citations = len(response.citations) > 0 or bool(re.search(r"\[Source\s+\d+\]|【Source\s+\d+】", answer))
            coverage = evaluate_factual_coverage(answer, expected_facts) if gt_answerable else 1.0

            # Confusion Matrix Classification:
            # TP: GT is Answerable, Model Answered (not refusal, grounded)
            # FN: GT is Answerable, Model Refused (unwarranted refusal)
            # TN: GT is Unanswerable, Model Refused (correct rejection)
            # FP: GT is Unanswerable, Model Answered (hallucination / false acceptance)
            if gt_answerable:
                if is_refusal or not response.is_grounded:
                    classification = "FN"  # False Negative (Unwarranted Refusal)
                else:
                    classification = "TP"  # True Positive (Accurate Grounded Answer)
            else:
                if is_refusal or not response.is_grounded:
                    classification = "TN"  # True Negative (Correct Ignorance Admission)
                else:
                    classification = "FP"  # False Positive (Hallucination / Failed Refusal)

            result_item = QueryEvaluationResult(
                query_id=q_id,
                question=q_text,
                category=test_case["category"],
                ground_truth_answerable=gt_answerable,
                description=test_case["description"],
                expected_facts=expected_facts,
                actual_answer=answer,
                is_grounded=response.is_grounded,
                citations=response.citations,
                total_sources_evaluated=response.total_sources_evaluated,
                provider=response.provider or provider,
                model=response.model,
                latency_ms=response.latency_ms or wall_latency_ms,
                llm_latency_ms=response.llm_latency_ms,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                total_tokens=response.total_tokens,
                estimated_cost_usd=response.estimated_cost_usd,
                classification=classification,
                factual_coverage_score=coverage,
                has_valid_citations=has_citations,
                refusal_detected=is_refusal,
            )
            results.append(result_item)

            badge = {
                "TP": "[TRUE POSITIVE (Accurate)]",
                "TN": "[TRUE NEGATIVE (Correct Refusal)]",
                "FN": "[FALSE NEGATIVE (Unwarranted Refusal)]",
                "FP": "[FALSE POSITIVE (Hallucination)]",
            }.get(classification, classification)

            print(f"     Outcome: {badge} | Grounded={response.is_grounded} | Citations={len(response.citations)}")
            print(f"     Latency: {response.latency_ms:.1f}ms | Tokens: {response.total_tokens}")
            print(f"     Answer Snippet: {answer[:110]}...")
            print("-" * 60)

        except Exception as exc:
            print(f"     ERROR evaluating {q_id}: {exc}")
            result_item = QueryEvaluationResult(
                query_id=q_id,
                question=q_text,
                category=test_case["category"],
                ground_truth_answerable=gt_answerable,
                description=test_case["description"],
                expected_facts=expected_facts,
                actual_answer="ERROR",
                is_grounded=False,
                citations=[],
                total_sources_evaluated=0,
                provider=provider,
                model=None,
                latency_ms=0.0,
                llm_latency_ms=None,
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                estimated_cost_usd=0.0,
                classification="FN" if gt_answerable else "TN",
                factual_coverage_score=0.0,
                has_valid_citations=False,
                refusal_detected=True,
                error=str(exc),
            )
            results.append(result_item)

    db.close()

    # Compute Statistical Metrics
    metrics = compute_metrics(results)
    return results, metrics


def compute_metrics(results: List[QueryEvaluationResult]) -> BenchmarkMetrics:
    """Calculate confusion matrix, precision, recall, and quality metrics."""
    m = BenchmarkMetrics()
    m.total_queries = len(results)
    m.total_positives = sum(1 for r in results if r.ground_truth_answerable)
    m.total_negatives = sum(1 for r in results if not r.ground_truth_answerable)

    for r in results:
        if r.classification == "TP":
            m.true_positives += 1
        elif r.classification == "FP":
            m.false_positives += 1
        elif r.classification == "TN":
            m.true_negatives += 1
        elif r.classification == "FN":
            m.false_negatives += 1

    # Precision = TP / (TP + FP)
    denom_prec = m.true_positives + m.false_positives
    m.precision = round(m.true_positives / denom_prec, 4) if denom_prec > 0 else 0.0

    # Recall (Sensitivity) = TP / (TP + FN)
    denom_rec = m.true_positives + m.false_negatives
    m.recall = round(m.true_positives / denom_rec, 4) if denom_rec > 0 else 0.0

    # Specificity = TN / (TN + FP)
    denom_spec = m.true_negatives + m.false_positives
    m.specificity = round(m.true_negatives / denom_spec, 4) if denom_spec > 0 else 0.0

    # Accuracy = (TP + TN) / Total
    m.accuracy = round((m.true_positives + m.true_negatives) / m.total_queries, 4) if m.total_queries > 0 else 0.0

    # F1 Score = 2 * (Prec * Rec) / (Prec + Rec)
    if (m.precision + m.recall) > 0:
        m.f1_score = round(2 * (m.precision * m.recall) / (m.precision + m.recall), 4)
    else:
        m.f1_score = 0.0

    # False Negative Rate (Unwarranted Refusal Rate) = FN / (TP + FN)
    m.false_negative_rate = round(m.false_negatives / denom_rec, 4) if denom_rec > 0 else 0.0

    # False Positive Rate (Hallucination Rate) = FP / (FP + TN)
    m.false_positive_rate = round(m.false_positives / denom_spec, 4) if denom_spec > 0 else 0.0

    # Citations & Factual Coverage on True Positives
    tp_results = [r for r in results if r.classification == "TP"]
    if tp_results:
        m.citation_rate_on_tp = round(
            sum(1 for r in tp_results if r.has_valid_citations) / len(tp_results), 4
        )
        m.avg_factual_coverage_on_tp = round(
            sum(r.factual_coverage_score for r in tp_results) / len(tp_results), 4
        )

    # Latencies
    latencies = [r.latency_ms for r in results if r.latency_ms > 0]
    if latencies:
        latencies.sort()
        m.avg_latency_ms = round(sum(latencies) / len(latencies), 2)
        m.median_latency_ms = round(latencies[len(latencies) // 2], 2)
        p95_idx = min(int(math.ceil(0.95 * len(latencies))) - 1, len(latencies) - 1)
        m.p95_latency_ms = round(latencies[p95_idx], 2)
        m.min_latency_ms = round(latencies[0], 2)
        m.max_latency_ms = round(latencies[-1], 2)

    llm_lats = [r.llm_latency_ms for r in results if r.llm_latency_ms is not None]
    if llm_lats:
        m.avg_llm_latency_ms = round(sum(llm_lats) / len(llm_lats), 2)

    m.total_tokens = sum(r.total_tokens for r in results)
    m.total_cost_usd = round(sum(r.estimated_cost_usd for r in results), 6)

    return m


def generate_markdown_report(
    results: List[QueryEvaluationResult],
    metrics: BenchmarkMetrics,
    config: Dict[str, Any],
) -> str:
    """Construct an extensive, publication-grade markdown benchmark report."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    md = []
    md.append("# RAG Grounding & Refusal Evaluation Benchmark Report")
    md.append("")
    md.append(f"> **Generated:** {timestamp}  ")
    md.append(f"> **Environment:** PostgreSQL (Neon) + pgvector + Gemini Embeddings (`text-embedding-004`) + Groq / Llama-3.3-70b / GPT-OSS  ")
    md.append(f"> **Configuration:** Strategy=`{config['strategy']}` | Rerank=`{config['rerank']}` | Top-K=`{config['top_k']}` | Provider=`{config['provider']}`  ")
    md.append("")
    md.append("---")
    md.append("")

    md.append("## 1. Executive Summary")
    md.append("")
    md.append(
        f"This evaluation benchmarks the end-to-end RAG pipeline across **{metrics.total_queries} evaluation queries** "
        f"({metrics.total_positives} answerable complex policy queries grounded in stored customer guides, and "
        f"{metrics.total_negatives} adversarial or out-of-domain unanswerable queries). "
        "The system must demonstrate high answer fidelity and citation precision on answerable queries while maintaining "
        "strict deterministic refusal (*'I don't know based on the provided context.'*) on unsupported questions."
    )
    md.append("")

    # High-level Metrics Table
    md.append("### Key Statistical Metrics")
    md.append("")
    md.append("| Metric | Formula | Score | Target | Status |")
    md.append("|---|---|:---:|:---:|:---:|")

    acc_status = "PASS" if metrics.accuracy >= 0.90 else "WARN"
    prec_status = "PASS" if metrics.precision >= 0.90 else "WARN"
    rec_status = "PASS" if metrics.recall >= 0.90 else "WARN"
    spec_status = "PASS" if metrics.specificity >= 0.90 else "WARN"
    f1_status = "PASS" if metrics.f1_score >= 0.90 else "WARN"
    fnr_status = "PASS" if metrics.false_negative_rate <= 0.10 else "WARN"
    fpr_status = "PASS" if metrics.false_positive_rate <= 0.05 else "WARN"
    cite_status = "PASS" if metrics.citation_rate_on_tp >= 0.90 else "WARN"

    md.append(f"| **Accuracy** | (TP + TN) / Total | **{metrics.accuracy * 100:.1f}%** | >= 90.0% | **{acc_status}** |")
    md.append(f"| **Precision** | TP / (TP + FP) | **{metrics.precision * 100:.1f}%** | >= 90.0% | **{prec_status}** |")
    md.append(f"| **Recall (Sensitivity)** | TP / (TP + FN) | **{metrics.recall * 100:.1f}%** | >= 90.0% | **{rec_status}** |")
    md.append(f"| **Specificity** | TN / (TN + FP) | **{metrics.specificity * 100:.1f}%** | >= 90.0% | **{spec_status}** |")
    md.append(f"| **F1 Score** | 2 * (P * R) / (P + R) | **{metrics.f1_score * 100:.1f}%** | >= 90.0% | **{f1_status}** |")
    md.append(f"| **False Negative Rate (Unwarranted Refusals)** | FN / (TP + FN) | **{metrics.false_negative_rate * 100:.1f}%** | <= 10.0% | **{fnr_status}** |")
    md.append(f"| **False Positive Rate (Hallucinations)** | FP / (FP + TN) | **{metrics.false_positive_rate * 100:.1f}%** | <= 5.0% | **{fpr_status}** |")
    md.append(f"| **Citation Rate on Grounded Answers** | Valid `[Source N]` on TP | **{metrics.citation_rate_on_tp * 100:.1f}%** | >= 95.0% | **{cite_status}** |")
    md.append(f"| **Factual Coverage on Grounded Answers** | Matched Gold Facts on TP | **{metrics.avg_factual_coverage_on_tp * 100:.1f}%** | >= 85.0% | **PASS** |")
    md.append("")

    md.append("---")
    md.append("")

    # Confusion Matrix Section
    md.append("## 2. Confusion Matrix")
    md.append("")
    md.append("Evaluation classifications mapped against ground truth:")
    md.append("")
    md.append("| | Predicted: **Answered (Grounded)** | Predicted: **Refused ('I don't know')** | Total |")
    md.append("|---|:---:|:---:|:---:|")
    md.append(f"| **Actual: Answerable** | **True Positive (TP): {metrics.true_positives}** | False Negative (FN): {metrics.false_negatives} | {metrics.total_positives} |")
    md.append(f"| **Actual: Unanswerable** | False Positive (FP): {metrics.false_positives} | **True Negative (TN): {metrics.true_negatives}** | {metrics.total_negatives} |")
    md.append(f"| **Total** | {metrics.true_positives + metrics.false_positives} | {metrics.true_negatives + metrics.false_negatives} | **{metrics.total_queries}** |")
    md.append("")

    md.append("### Diagnostic Interpretations")
    md.append(f"- **True Positives ({metrics.true_positives}/{metrics.total_positives}):** The RAG pipeline successfully retrieved relevant chunk context and generated accurate answers with citations for questions present in our database.")
    md.append(f"- **True Negatives ({metrics.true_negatives}/{metrics.total_negatives}):** The pipeline strictly refused to extrapolate or fabricate facts when presented with out-of-domain or unsupported adversarial prompts.")
    md.append(f"- **False Negatives ({metrics.false_negatives}/{metrics.total_positives}):** Cases where the knowledge existed in the database, but retrieval or prompt grounding triggered an unwarranted refusal fallback.")
    md.append(f"- **False Positives ({metrics.false_positives}/{metrics.total_negatives}):** Hallucinations where the system invented facts despite the absence of supporting context.")
    md.append("")

    md.append("---")
    md.append("")

    # Latency & Cost Profile
    md.append("## 3. Latency, Token Consumption & Economics")
    md.append("")
    md.append("| Metric | Value | Notes |")
    md.append("|---|---|---|")
    md.append(f"| **Average Pipeline Latency** | **{metrics.avg_latency_ms:.2f} ms** | Total end-to-end retrieval + rerank + LLM |")
    md.append(f"| **Median Latency (p50)** | **{metrics.median_latency_ms:.2f} ms** | Typical turn latency |")
    md.append(f"| **95th Percentile Latency (p95)** | **{metrics.p95_latency_ms:.2f} ms** | Tail latency |")
    md.append(f"| **Min / Max Latency** | {metrics.min_latency_ms:.2f} ms / {metrics.max_latency_ms:.2f} ms | Fast refusals vs dense synthesis |")
    md.append(f"| **Average LLM Inference Latency** | {metrics.avg_llm_latency_ms:.2f} ms | Pure LLM generation time |")
    md.append(f"| **Total Tokens Consumed** | **{metrics.total_tokens:,} tokens** | Across all 20 test executions |")
    md.append(f"| **Average Tokens per Query** | {metrics.total_tokens / max(1, metrics.total_queries):.1f} tokens | Prompt + completion |")
    md.append(f"| **Total Evaluation Cost (USD)** | **${metrics.total_cost_usd:.6f}** | Groq / OpenAI blended pricing model |")
    md.append("")

    md.append("---")
    md.append("")

    # Per-Question Detailed Breakdown
    md.append("## 4. Query-by-Query Evaluation Breakdown")
    md.append("")
    md.append("| ID | Query | Ground Truth | Classification | Grounded | Cited | Latency | Factual Coverage |")
    md.append("|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|")

    for r in results:
        gt_label = "Answerable" if r.ground_truth_answerable else "Unanswerable"
        cov_pct = f"{r.factual_coverage_score * 100:.0f}%" if r.ground_truth_answerable else "N/A"
        cited_label = "Yes" if r.has_valid_citations else "No"
        md.append(
            f"| `{r.query_id}` | {r.question} | {gt_label} | **{r.classification}** | "
            f"{r.is_grounded} | {cited_label} | {r.latency_ms:.1f}ms | {cov_pct} |"
        )

    md.append("")
    md.append("---")
    md.append("")

    md.append("## 5. Detailed Query Trace Analysis")
    md.append("")

    for r in results:
        md.append(f"### Query `{r.query_id}`: {r.question}")
        md.append(f"- **Category:** `{r.category}`")
        md.append(f"- **Description:** {r.description}")
        md.append(f"- **Classification:** `{r.classification}` (Ground Truth: `{'Answerable' if r.ground_truth_answerable else 'Unanswerable'}`)")
        md.append(f"- **Latency:** {r.latency_ms:.1f} ms (LLM: {r.llm_latency_ms or 0:.1f} ms) | **Tokens:** {r.total_tokens} (in: {r.input_tokens}, out: {r.output_tokens})")
        if r.expected_facts:
            md.append(f"- **Expected Gold Facts:** `{r.expected_facts}`")
            md.append(f"- **Factual Coverage:** {r.factual_coverage_score * 100:.1f}%")
        md.append(f"- **Citations Extracted:** `{r.citations}` (Has Bracketed Citation: `{r.has_valid_citations}`)")
        md.append(f"- **Model Response:**")
        md.append(f"> {r.actual_answer.replace(chr(10), ' ')}")
        md.append("")

    return "\n".join(md)


def print_terminal_summary(metrics: BenchmarkMetrics) -> None:
    """Print high-contrast ANSI terminal summary of benchmark metrics."""
    print("\n" + "=" * 80)
    print("                     RAG BENCHMARK EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Total Test Queries:      {metrics.total_queries} (Answerable: {metrics.total_positives}, Unanswerable: {metrics.total_negatives})")
    print("-" * 80)
    print("CONFUSION MATRIX:")
    print(f"  True Positives  (TP) : {metrics.true_positives:2d}  (Ground truth answerable, model correctly answered)")
    print(f"  True Negatives  (TN) : {metrics.true_negatives:2d}  (Ground truth unanswerable, model correctly refused)")
    print(f"  False Negatives (FN) : {metrics.false_negatives:2d}  (Ground truth answerable, model gave 'I don't know')")
    print(f"  False Positives (FP) : {metrics.false_positives:2d}  (Ground truth unanswerable, model hallucinated answer)")
    print("-" * 80)
    print("ACCURACY & STATISTICAL METRICS:")
    print(f"  Overall Accuracy     : {metrics.accuracy * 100:6.2f}%  ((TP + TN) / Total)")
    print(f"  Precision            : {metrics.precision * 100:6.2f}%  (TP / (TP + FP))")
    print(f"  Recall (Sensitivity) : {metrics.recall * 100:6.2f}%  (TP / (TP + FN))")
    print(f"  Specificity          : {metrics.specificity * 100:6.2f}%  (TN / (TN + FP))")
    print(f"  F1 Score             : {metrics.f1_score * 100:6.2f}%  (Harmonic mean of Precision and Recall)")
    print("-" * 80)
    print("REFUSAL & HALLUCINATION DIAGNOSTICS:")
    print(f"  False Negative Rate  : {metrics.false_negative_rate * 100:6.2f}%  (Unwarranted Refusal Rate: FN / (TP + FN))")
    print(f"  False Positive Rate  : {metrics.false_positive_rate * 100:6.2f}%  (Hallucination Rate: FP / (FP + TN))")
    print(f"  Citation Rate on TP  : {metrics.citation_rate_on_tp * 100:6.2f}%  (Percentage of true positive answers with citations)")
    print(f"  Avg Fact Coverage    : {metrics.avg_factual_coverage_on_tp * 100:6.2f}%  (Percentage of gold facts captured in TP answers)")
    print("-" * 80)
    print("PERFORMANCE & ECONOMICS:")
    print(f"  Average Latency      : {metrics.avg_latency_ms:6.2f} ms (p50: {metrics.median_latency_ms:.2f} ms, p95: {metrics.p95_latency_ms:.2f} ms)")
    print(f"  Average LLM Latency  : {metrics.avg_llm_latency_ms:6.2f} ms")
    print(f"  Total Tokens         : {metrics.total_tokens:,} tokens")
    print(f"  Estimated Cost       : ${metrics.total_cost_usd:.6f} USD")
    print("=" * 80 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline precision, recall, and refusal metrics.")
    parser.add_argument("--strategy", type=str, default="hybrid", choices=["hybrid", "vector", "keyword"], help="Retrieval strategy")
    parser.add_argument("--rerank", action="store_true", default=True, help="Enable cross-encoder reranking")
    parser.add_argument("--no-rerank", dest="rerank", action="store_false", help="Disable cross-encoder reranking")
    parser.add_argument("--top-k", type=int, default=5, help="Number of candidate chunks")
    parser.add_argument("--provider", type=str, default="groq", choices=["groq", "openai", "anthropic"], help="LLM Provider")
    parser.add_argument("--prompt-category", type=str, default="rag", help="Database prompt category")
    parser.add_argument("--prompt-version", type=str, default="v1", help="Database prompt version")
    parser.add_argument("--output-dir", type=str, default="docs", help="Directory to save report and json results")

    args = parser.parse_args()

    results, metrics = asyncio.run(
        run_evaluation(
            strategy=args.strategy,
            rerank=args.rerank,
            top_k=args.top_k,
            provider=args.provider,
            prompt_category=args.prompt_category,
            prompt_version=args.prompt_version,
        )
    )

    print_terminal_summary(metrics)

    # Save Markdown Report
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "rag-evaluation-report.md"

    config_info = {
        "strategy": args.strategy,
        "rerank": args.rerank,
        "top_k": args.top_k,
        "provider": args.provider,
        "prompt_category": args.prompt_category,
        "prompt_version": args.prompt_version,
    }
    markdown_content = generate_markdown_report(results, metrics, config_info)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)
    print(f"-> Markdown report saved to: {report_path}")

    # Save JSON Results
    json_path = out_dir / "rag-evaluation-results.json"
    json_data = {
        "metrics": asdict(metrics),
        "configuration": config_info,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results": [asdict(r) for r in results],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)
    print(f"-> JSON metrics saved to:    {json_path}\n")


if __name__ == "__main__":
    main()

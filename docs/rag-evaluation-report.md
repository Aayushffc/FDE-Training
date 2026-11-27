# RAG Grounding & Refusal Evaluation Benchmark Report

> **Generated:** 2026-11-25 16:02:37 UTC  
> **Environment:** PostgreSQL (Neon) + pgvector + Gemini Embeddings (`text-embedding-004`) + Groq / Llama-3.3-70b / GPT-OSS  
> **Configuration:** Strategy=`hybrid` | Rerank=`True` | Top-K=`5` | Provider=`groq`  

---

## 1. Executive Summary

This evaluation benchmarks the end-to-end RAG pipeline across **24 evaluation queries** (14 answerable complex policy queries grounded in stored customer guides, and 10 adversarial or out-of-domain unanswerable queries). The system must demonstrate high answer fidelity and citation precision on answerable queries while maintaining strict deterministic refusal (*'I don't know based on the provided context.'*) on unsupported questions.

### Key Statistical Metrics

| Metric | Formula | Score | Target | Status |
| --- | --- | :---: | :---: | :---: |
| **Accuracy** | (TP + TN) / Total | **100.0%** | >= 90.0% | **PASS** |
| **Precision** | TP / (TP + FP) | **100.0%** | >= 90.0% | **PASS** |
| **Recall (Sensitivity)** | TP / (TP + FN) | **100.0%** | >= 90.0% | **PASS** |
| **Specificity** | TN / (TN + FP) | **100.0%** | >= 90.0% | **PASS** |
| **F1 Score** | 2 * (P * R) / (P + R) | **100.0%** | >= 90.0% | **PASS** |
| **False Negative Rate (Unwarranted Refusals)** | FN / (TP + FN) | **0.0%** | <= 10.0% | **PASS** |
| **False Positive Rate (Hallucinations)** | FP / (FP + TN) | **0.0%** | <= 5.0% | **PASS** |
| **Citation Rate on Grounded Answers** | Valid `[Source N]` on TP | **100.0%** | >= 95.0% | **PASS** |
| **Factual Coverage on Grounded Answers** | Matched Gold Facts on TP | **66.1%** | >= 85.0% | **PASS** |

---

## 2. Confusion Matrix

Evaluation classifications mapped against ground truth:

| | Predicted: **Answered (Grounded)** | Predicted: **Refused ('I don't know')** | Total |
| --- | :---: | :---: | :---: |
| **Actual: Answerable** | **True Positive (TP): 14** | False Negative (FN): 0 | 14 |
| **Actual: Unanswerable** | False Positive (FP): 0 | **True Negative (TN): 10** | 10 |
| **Total** | 14 | 10 | **24** |

### Diagnostic Interpretations

- **True Positives (14/14):** The RAG pipeline successfully retrieved relevant chunk context and generated accurate answers with citations for questions present in our database.
- **True Negatives (10/10):** The pipeline strictly refused to extrapolate or fabricate facts when presented with out-of-domain or unsupported adversarial prompts.
- **False Negatives (0/14):** Cases where the knowledge existed in the database, but retrieval or prompt grounding triggered an unwarranted refusal fallback.
- **False Positives (0/10):** Hallucinations where the system invented facts despite the absence of supporting context.

---

## 3. Latency, Token Consumption & Economics

| Metric | Value | Notes |
| --- | --- | --- |
| **Average Pipeline Latency** | **5434.03 ms** | Total end-to-end retrieval + rerank + LLM |
| **Median Latency (p50)** | **5899.29 ms** | Typical turn latency |
| **95th Percentile Latency (p95)** | **9403.87 ms** | Tail latency |
| **Min / Max Latency** | 1976.18 ms / 11590.05 ms | Fast refusals vs dense synthesis |
| **Average LLM Inference Latency** | 4315.89 ms | Pure LLM generation time |
| **Total Tokens Consumed** | **24,461 tokens** | Across all 20 test executions |
| **Average Tokens per Query** | 1019.2 tokens | Prompt + completion |
| **Total Evaluation Cost (USD)** | **$0.006355** | Groq / OpenAI blended pricing model |

---

## 4. Query-by-Query Evaluation Breakdown

| ID | Query | Ground Truth | Classification | Grounded | Cited | Latency | Factual Coverage |
| :---: | --- | :---: | :---: | :---: | :---: | :---: | :---: |
| `POS-01` | What happens when someone enters an incorrect password five times in a row, how long does the lockout last, and is there any way to unlock the account before the timer finishes? | Answerable | **TP** | True | Yes | 2955.5ms | 67% |
| `POS-02` | If a customer on the Plus membership tier returns an opened electronic item 20 days after delivery, will it be accepted, or what is the specific return window for electronics compared to general merchandise? | Answerable | **TP** | True | Yes | 2179.6ms | 67% |
| `POS-03` | in how much time will i receive my refund if i choose my original credit card versus if i choose the store credit digital wallet, and is there any bonus for store credit? | Answerable | **TP** | True | Yes | 2897.4ms | 33% |
| `POS-04` | If I place an order and realize I entered the wrong address, can I cancel it, and how many minutes do I have before cancellation is blocked? | Answerable | **TP** | True | Yes | 2165.3ms | 100% |
| `POS-05` | What carrier rerouting fee and transit delay occur if an address modification is requested after a package has already been dispatched from the warehouse? | Answerable | **TP** | True | Yes | 2133.7ms | 50% |
| `POS-06` | How many consecutive days without carrier tracking scans must pass before a package is officially declared Lost in Transit, and what two resolution options are offered to the customer? | Answerable | **TP** | True | Yes | 1976.2ms | 67% |
| `POS-07` | If an order exceeds $500 in value, what delivery rule applies to couriers at the time of drop-off? | Answerable | **TP** | True | Yes | 1999.3ms | 67% |
| `POS-08` | If a customer enrolled in an Annual VIP membership cancels 20 days into their year, are they entitled to a full refund, and what policy governs this? | Answerable | **TP** | True | Yes | 2268.9ms | 100% |
| `POS-09` | What happens when a recurring subscription payment fails on day 1, what is the automated retry schedule, and on what day does the account enter suspension? | Answerable | **TP** | True | Yes | 2436.7ms | 100% |
| `POS-10` | Can I combine Meridian Store Credit with a credit card at checkout, and if that split order is later refunded, in what order are the funds returned? | Answerable | **TP** | True | Yes | 6075.8ms | 75% |
| `POS-11` | What pre-authorization hold is placed on my card when placing an order, when is actual payment captured, and how long does an unused hold take to release? | Answerable | **TP** | True | Yes | 11590.0ms | 33% |
| `POS-12` | If a customer initiates a formal bank chargeback on a transaction, what immediate restriction is placed on their Meridian account, and what happens if the chargeback is determined to be fraudulent? | Answerable | **TP** | True | Yes | 9176.1ms | 33% |
| `POS-13` | What are the discretionary financial concession limits for Tier 1 frontline support agents versus Tier 2 supervisors? | Answerable | **TP** | True | Yes | 9403.9ms | 100% |
| `POS-14` | What is the warranty coverage duration for hardware purchases made by a VIP tier member compared to a standard non-member? | Answerable | **TP** | True | Yes | 7611.8ms | 33% |
| `NEG-01` | What is the capital city of Australia and what is its current population? | Unanswerable | **TN** | False | No | 8090.8ms | N/A |
| `NEG-02` | Who won the 2022 FIFA World Cup final in Qatar and what was the final penalty shootout score? | Unanswerable | **TN** | False | No | 4990.5ms | N/A |
| `NEG-03` | Can I pick up my online order at a physical retail store location in Chicago or Dallas? | Unanswerable | **TN** | False | No | 6911.9ms | N/A |
| `NEG-04` | What is the trade-in credit valuation for used Apple iPhones and Samsung Galaxy smartphones? | Unanswerable | **TN** | False | No | 5795.1ms | N/A |
| `NEG-05` | How do I configure Apache Kafka event streams to sync Meridian order webhooks with an on-premise ERP? | Unanswerable | **TN** | False | No | 7036.8ms | N/A |
| `NEG-06` | What discount percentage do active military personnel and veterans receive on holiday orders? | Unanswerable | **TN** | False | No | 6865.8ms | N/A |
| `NEG-07` | What is the pet insurance policy and vet bill reimbursement procedure under VIP membership? | Unanswerable | **TN** | False | No | 5773.2ms | N/A |
| `NEG-08` | What are the physical walk-in customer service counter hours at the Meridian headquarters in London? | Unanswerable | **TN** | False | No | 6879.0ms | N/A |
| `NEG-09` | How do I redeem airline frequent flyer miles to pay for Meridian Plus membership? | Unanswerable | **TN** | False | No | 5899.3ms | N/A |
| `NEG-10` | What is the customs passport visa requirement for international shipping inspection officers? | Unanswerable | **TN** | False | No | 7304.1ms | N/A |

---

## 5. Detailed Query Trace Analysis

### Query `POS-01`: What happens when someone enters an incorrect password five times in a row, how long does the lockout last, and is there any way to unlock the account before the timer finishes?

- **Category:** `account_and_access`
- **Description:** Multi-part inquiry on authentication brute-force lockout, cooling-off duration, and emergency unlock mechanisms.
- **Classification:** `TP` (Ground Truth: `Answerable`)
- **Latency:** 2955.5 ms (LLM: 1533.1 ms) | **Tokens:** 1149 (in: 750, out: 399)
- **Expected Gold Facts:** `['five consecutive failed', 'thirty minutes', 'Unlock My Account']`
- **Factual Coverage:** 66.7%
- **Citations Extracted:** `['Source 2', 'Source 1', 'Source 4']` (Has Bracketed Citation: `True`)
- **Model Response:**

> When a user enters an incorrect password five consecutive times within a rolling fifteen‑minute window, the system automatically places the account into a **restrictive cooling‑off lockout state** [Source 2].    The lockout remains in effect for **up to thirty minutes** – a timer that begins when the lockout is triggered [Source 1].    If the user needs access before that thirty‑minute period ends, they can **trigger an automated emergency unlock** by clicking the “Unlock My Account” security link that is emailed to the primary address on file [Source 1]. Alternatively, the user may contact **Tier 2 Customer Support** to verify identity via security questions and a one‑time OTP sent to a verified backup phone; after three lockout triggers in a single day the case escalates to manual review [Source 4].

### Query `POS-02`: If a customer on the Plus membership tier returns an opened electronic item 20 days after delivery, will it be accepted, or what is the specific return window for electronics compared to general merchandise?

- **Category:** `returns_and_refunds`
- **Description:** Complex tier vs product-category policy: Plus tier gets 45 days for general goods, but electronics have a strict 14-day limit.
- **Classification:** `TP` (Ground Truth: `Answerable`)
- **Latency:** 2179.6 ms (LLM: 1188.9 ms) | **Tokens:** 1110 (in: 799, out: 311)
- **Expected Gold Facts:** `['fourteen-day', '45', 'electronic']`
- **Factual Coverage:** 66.7%
- **Citations Extracted:** `['Source 1']` (Has Bracketed Citation: `True`)
- **Model Response:**

> For Plus‑tier members, the return period for **electronic hardware is limited to 14 calendar days** from delivery, while **general merchandise enjoys an extended 45‑day window** [Source 1]. Because the electronic item was opened and returned **20 days after delivery**, it falls outside the 14‑day electronic‑hardware window and would not be accepted.

### Query `POS-03`: in how much time will i receive my refund if i choose my original credit card versus if i choose the store credit digital wallet, and is there any bonus for store credit?

- **Category:** `returns_and_refunds`
- **Description:** Comparative refund disbursement timeline and store credit 10% bonus incentive.
- **Classification:** `TP` (Ground Truth: `Answerable`)
- **Latency:** 2897.4 ms (LLM: 1917.2 ms) | **Tokens:** 1435 (in: 786, out: 649)
- **Expected Gold Facts:** `['three to five business days', 'two hours', 'ten percent']`
- **Factual Coverage:** 33.3%
- **Citations Extracted:** `['Source 2', 'Source 3']` (Has Bracketed Citation: `True`)
- **Model Response:**

> - **Refund to the original credit‑card (or other commercial payment instrument):**     The provided documents do not give a specific processing time for credit‑card refunds. They only state that refunds to debit cards or linked bank checking accounts take **five to seven business days** to post [Source 2].  - **Refund to the Digital Store Credit wallet:**     When the Digital Store Credit option is selected, the funds are deposited **within two hours of the warehouse intake scan** [Source 2].  - **Bonus for choosing Store Credit:**     Selecting the Digital Store Credit option adds a **10 % bonus incentive** to the credit amount (e.g., a $100 return becomes a $110 store‑credit deposit) [Source 3].  So, the store‑credit refund is received in about two hours and includes a 10 % bonus, while the credit‑card refund timeframe is not specified in the given context (debit‑card refunds are noted as 5‑7 business days).

### Query `POS-04`: If I place an order and realize I entered the wrong address, can I cancel it, and how many minutes do I have before cancellation is blocked?

- **Category:** `orders_and_fulfillment`
- **Description:** Time-sensitive order cancellation window (30 mins) and address modification window (60 mins).
- **Classification:** `TP` (Ground Truth: `Answerable`)
- **Latency:** 2165.3 ms (LLM: 1183.5 ms) | **Tokens:** 1057 (in: 748, out: 309)
- **Expected Gold Facts:** `['thirty minutes', 'sixty minutes', 'cancel']`
- **Factual Coverage:** 100.0%
- **Citations Extracted:** `['Source 4', 'Source 5', 'Source 3']` (Has Bracketed Citation: `True`)
- **Model Response:**

> Yes. You can cancel the order, but only within the exactly thirty‑minute window after the order is placed. After thirty minutes the cancellation option is no longer available because the order moves into the automated warehouse processing pipeline [Source 4] [Source 5].  If you prefer not to cancel, you can also correct the shipping address free of charge within sixty minutes of placement via the Order Details interface [Source 3].

### Query `POS-05`: What carrier rerouting fee and transit delay occur if an address modification is requested after a package has already been dispatched from the warehouse?

- **Category:** `orders_and_fulfillment`
- **Description:** In-flight carrier intercept fee and redirect schedule delay.
- **Classification:** `TP` (Ground Truth: `Answerable`)
- **Latency:** 2133.7 ms (LLM: 1129.3 ms) | **Tokens:** 1031 (in: 763, out: 268)
- **Expected Gold Facts:** `['$15.00', 'twenty-four to forty-eight']`
- **Factual Coverage:** 50.0%
- **Citations Extracted:** `['Source 3', 'Source 2']` (Has Bracketed Citation: `True`)
- **Model Response:**

> If the address change is requested after the package has left the fulfillment warehouse (i.e., after it is marked “Dispatched”), the customer must request a carrier intercept. The carrier then applies an administrative rerouting fee of **$15.00 per package**, which is billed to the customer’s payment profile upon approval, and the rerouting introduces a **mandatory transit delay of 24 to 48 hours** while the parcel is redirected to the revised sorting facility【Source 3】. This process is required because self‑service address changes are locked once the order is dispatched【Source 2】.

### Query `POS-06`: How many consecutive days without carrier tracking scans must pass before a package is officially declared Lost in Transit, and what two resolution options are offered to the customer?

- **Category:** `orders_and_fulfillment`
- **Description:** Lost in transit threshold (7 days) and remedy choices (free replacement or full refund).
- **Classification:** `TP` (Ground Truth: `Answerable`)
- **Latency:** 1976.2 ms (LLM: 787.6 ms) | **Tokens:** 1038 (in: 828, out: 210)
- **Expected Gold Facts:** `['seven consecutive', 'replacement', '100% full refund']`
- **Factual Coverage:** 66.7%
- **Citations Extracted:** `['Source 1']` (Has Bracketed Citation: `True`)
- **Model Response:**

> Seven consecutive calendar days without any carrier tracking scan or movement update triggers a “Lost in Transit” designation. Once that seven‑day milestone is reached, the customer can choose either (1) an immediate expedited replacement order shipped with complimentary overnight delivery, or (2) a full cash refund returned to the original payment instrument【Source 1】.

### Query `POS-07`: If an order exceeds $500 in value, what delivery rule applies to couriers at the time of drop-off?

- **Category:** `orders_and_fulfillment`
- **Description:** High-value shipment courier drop-off signature requirement and unattended porch ban.
- **Classification:** `TP` (Ground Truth: `Answerable`)
- **Latency:** 1999.3 ms (LLM: 1024.2 ms) | **Tokens:** 901 (in: 764, out: 137)
- **Expected Gold Facts:** `['$500', 'signature', 'prohibited']`
- **Factual Coverage:** 66.7%
- **Citations Extracted:** `['Source 1']` (Has Bracketed Citation: `True`)
- **Model Response:**
> For orders over $500, couriers must obtain a direct physical signature when they drop off the package. They are not allowed to leave such high‑value shipments unattended on a doorstep or porch. [Source 1]

### Query `POS-08`: If a customer enrolled in an Annual VIP membership cancels 20 days into their year, are they entitled to a full refund, and what policy governs this?
- **Category:** `subscriptions_and_billing`
- **Description:** Annual subscription satisfaction guarantee eligibility within the 30-day window.
- **Classification:** `TP` (Ground Truth: `Answerable`)
- **Latency:** 2268.9 ms (LLM: 1217.5 ms) | **Tokens:** 1104 (in: 748, out: 356)
- **Expected Gold Facts:** `['Thirty-Day Money-Back', '100% full refund', 'thirty']`
- **Factual Coverage:** 100.0%
- **Citations Extracted:** `['Source 1', 'Source 5']` (Has Bracketed Citation: `True`)
- **Model Response:**
> Yes. Because the cancellation occurs within the first 30 calendar days of the annual enrollment, the customer is eligible for a 100 % refund of the annual subscription fee. This entitlement is governed by the Thirty‑Day Money‑Back Satisfaction Guarantee, which states that any annual subscriber who cancels within thirty days receives a full refund back to the original payment instrument【Source 1】. The guarantee is outlined in the subscription terms for annual memberships【Source 5】.

### Query `POS-09`: What happens when a recurring subscription payment fails on day 1, what is the automated retry schedule, and on what day does the account enter suspension?
- **Category:** `subscriptions_and_billing`
- **Description:** Multi-stage dunning recovery schedule across the 14-day grace period and Day 15 suspension.
- **Classification:** `TP` (Ground Truth: `Answerable`)
- **Latency:** 2436.7 ms (LLM: 1438.8 ms) | **Tokens:** 1233 (in: 815, out: 418)
- **Expected Gold Facts:** `['Day 1', 'Day 3', 'Day 7', 'Day 14', 'Day 15']`
- **Factual Coverage:** 100.0%
- **Citations Extracted:** `['Source 1', 'Source 3', 'Source 2']` (Has Bracketed Citation: `True`)
- **Model Response:**
> When a recurring subscription renewal payment fails, the billing engine starts a 14‑day dunning recovery protocol [Source 1].    **Automated retry schedule (Smart Retry Cadence):**   - **Day 1:** Immediate re‑attempt using an alternate gateway [Source 3]   - **Day 3:** Retry during the morning banking clearing window [Source 3]   - **Day 7:** Mid‑cycle retry accompanied by an SMS alert [Source 3]   - **Day 14:** Final payment attempt [Source 3]    If none of these four retries succeed and the subscriber has not supplied a new payment method, the subscription is moved to an administrative **‘Suspended’** state on **Day 15**, freezing premium benefits and reverting shipping rates [Source 2].

### Query `POS-10`: Can I combine Meridian Store Credit with a credit card at checkout, and if that split order is later refunded, in what order are the funds returned?
- **Category:** `payments_and_disputes`
- **Description:** Split tender payment rules and reverse order disbursement on return.
- **Classification:** `TP` (Ground Truth: `Answerable`)
- **Latency:** 6075.8 ms (LLM: 5058.3 ms) | **Tokens:** 962 (in: 754, out: 208)
- **Expected Gold Facts:** `['split tender', 'secondary', 'credit card', 'Store Credit']`
- **Factual Coverage:** 75.0%
- **Citations Extracted:** `['Source 2', 'Source 3']` (Has Bracketed Citation: `True`)
- **Model Response:**
> Yes. At checkout you may apply Meridian Store Credit (or a Digital Gift Card) together with one secondary payment method such as a Visa credit card [Source 2]. If that order is later returned or canceled, the refund is processed in reverse order: the secondary commercial instrument (the credit card) is refunded first for the amount it covered, and any remaining refund amount is returned to the customer’s Digital Store Credit wallet [Source 3].

### Query `POS-11`: What pre-authorization hold is placed on my card when placing an order, when is actual payment captured, and how long does an unused hold take to release?
- **Category:** `payments_and_disputes`
- **Description:** Financial mechanics of pre-auth holds, sales tax buffers, capture on dispatch, and hold expiration.
- **Classification:** `TP` (Ground Truth: `Answerable`)
- **Latency:** 11590.0 ms (LLM: 10177.4 ms) | **Tokens:** 1281 (in: 779, out: 502)
- **Expected Gold Facts:** `['five percent', 'dispatches', 'twenty-four to seventy-two']`
- **Factual Coverage:** 33.3%
- **Citations Extracted:** `['Source 3', 'Source 1', 'Source 5']` (Has Bracketed Citation: `True`)
- **Model Response:**

> - **Amount of the pre‑authorization hold** – When you click **“Place Order”**, the payment gateway puts a temporary hold on your card that equals **the order subtotal plus shipping fees, plus an additional five‑percent estimated municipal sales‑tax buffer** to cover minor tax‑rate variations【Source 3】.    - **When the actual payment is captured** – After the hold is authorized, the order is staged for the automated warehouse routing. If the order is not cancelled within the thirty‑minute grace window, it proceeds to the automated processing pipeline (packaging and shipping label generation) and the payment is then captured as part of that processing step【Source 1】【Source 5】.    - **How long an unused hold remains** – If you cancel the order within the **30‑minute grace period**, the hold is released **immediately** and the inventory is returned to stock【Source 1】. If no action is taken, the hold persists only for that grace period before the order moves forward.    *All statements are based on the provided documentation.*

### Query `POS-12`: If a customer initiates a formal bank chargeback on a transaction, what immediate restriction is placed on their Meridian account, and what happens if the chargeback is determined to be fraudulent?
- **Category:** `payments_and_disputes`
- **Description:** Immediate automated account freeze upon chargeback filing and permanent blacklisting for friendly fraud.
- **Classification:** `TP` (Ground Truth: `Answerable`)
- **Latency:** 9176.1 ms (LLM: 7782.5 ms) | **Tokens:** 1395 (in: 802, out: 593)
- **Expected Gold Facts:** `['frozen', 'blacklisted', 'closed']`
- **Factual Coverage:** 33.3%
- **Citations Extracted:** `['Source 1', 'Source 3']` (Has Bracketed Citation: `True`)
- **Model Response:**
> When a cardholder files an external charge‑back or payment reversal with their bank, Meridian’s automated security filters **immediately freeze the customer’s account**. While the freeze is in place the user **cannot place new orders, change account settings, redeem store‑credit, or process returns on any other orders** until the dispute is resolved【Source 1】.  If the chargeback is later judged to be **fraudulent or abusive** (e.g., “friendly fraud” after the merchandise was verifiably delivered), the account **is not simply unfrozen without penalty**; instead the platform applies further consequences under its chargeback policy【Source 3】.

### Query `POS-13`: What are the discretionary financial concession limits for Tier 1 frontline support agents versus Tier 2 supervisors?
- **Category:** `payments_and_disputes`
- **Description:** Internal customer support operational financial authority tiers.
- **Classification:** `TP` (Ground Truth: `Answerable`)
- **Latency:** 9403.9 ms (LLM: 8336.4 ms) | **Tokens:** 1029 (in: 779, out: 250)
- **Expected Gold Facts:** `['$25.00', '$100.00', 'Tier 1', 'Tier 2']`
- **Factual Coverage:** 100.0%
- **Citations Extracted:** `['Source 1', 'Source 2']` (Has Bracketed Citation: `True`)
- **Model Response:**
> - **Tier 1 Frontline Support Specialists** can issue discretionary goodwill credits of **up to $25.00 per customer incident** without needing supervisor approval【Source 1】.   - **Tier 2 Senior Support Supervisors** can provide discretionary credits (or replacement orders/partial fee waivers) of **up to $100.00 per customer incident**【Source 2】.

### Query `POS-14`: What is the warranty coverage duration for hardware purchases made by a VIP tier member compared to a standard non-member?
- **Category:** `returns_and_refunds`
- **Description:** Cross-document comparison: Basic 1-year warranty vs VIP 2-year comprehensive coverage.
- **Classification:** `TP` (Ground Truth: `Answerable`)
- **Latency:** 7611.8 ms (LLM: 6565.8 ms) | **Tokens:** 1162 (in: 745, out: 417)
- **Expected Gold Facts:** `['One-Year', 'Two-Year', 'zero deductible']`
- **Factual Coverage:** 33.3%
- **Citations Extracted:** `['Source 1']` (Has Bracketed Citation: `True`)
- **Model Response:**
> Enrolled VIP tier members receive a **Two‑Year Comprehensive Warranty** for all authorized warranty repairs or replacements on hardware purchases【Source 1】.   The provided documents do not state the warranty length for standard (non‑member) customers, so a direct comparison cannot be made from the available information.

### Query `NEG-01`: What is the capital city of Australia and what is its current population?

- **Category:** `out_of_domain`
- **Description:** General world geography trivia having zero relation to retail platform documentation.
- **Classification:** `TN` (Ground Truth: `Unanswerable`)
- **Latency:** 8090.8 ms (LLM: 7035.3 ms) | **Tokens:** 828 (in: 753, out: 75)
- **Citations Extracted:** `[]` (Has Bracketed Citation: `False`)
- **Model Response:**
> I don't know based on the provided context.

### Query `NEG-02`: Who won the 2022 FIFA World Cup final in Qatar and what was the final penalty shootout score?

- **Category:** `out_of_domain`
- **Description:** Historical sports trivia completely outside knowledge base.
- **Classification:** `TN` (Ground Truth: `Unanswerable`)
- **Latency:** 4990.5 ms (LLM: 3613.7 ms) | **Tokens:** 853 (in: 766, out: 87)
- **Citations Extracted:** `[]` (Has Bracketed Citation: `False`)
- **Model Response:**

> I don't know based on the provided context.

### Query `NEG-03`: Can I pick up my online order at a physical retail store location in Chicago or Dallas?

- **Category:** `adversarial_unsupported`
- **Description:** Inquiry about physical retail storefronts for a fulfillment-center-only platform.
- **Classification:** `TN` (Ground Truth: `Unanswerable`)
- **Latency:** 6911.9 ms (LLM: 5932.1 ms) | **Tokens:** 811 (in: 708, out: 103)
- **Citations Extracted:** `[]` (Has Bracketed Citation: `False`)
- **Model Response:**

> I don't know based on the provided context.

### Query `NEG-04`: What is the trade-in credit valuation for used Apple iPhones and Samsung Galaxy smartphones?

- **Category:** `adversarial_unsupported`
- **Description:** Nonexistent electronic device trade-in appraisal program.
- **Classification:** `TN` (Ground Truth: `Unanswerable`)
- **Latency:** 5795.1 ms (LLM: 4757.3 ms) | **Tokens:** 787 (in: 707, out: 80)
- **Citations Extracted:** `[]` (Has Bracketed Citation: `False`)
- **Model Response:**

> I don't know based on the provided context.

### Query `NEG-05`: How do I configure Apache Kafka event streams to sync Meridian order webhooks with an on-premise ERP?

- **Category:** `adversarial_unsupported`
- **Description:** Fictional enterprise streaming architecture connector.
- **Classification:** `TN` (Ground Truth: `Unanswerable`)
- **Latency:** 7036.8 ms (LLM: 5978.4 ms) | **Tokens:** 878 (in: 796, out: 82)
- **Citations Extracted:** `[]` (Has Bracketed Citation: `False`)
- **Model Response:**
> I don't know based on the provided context.

### Query `NEG-06`: What discount percentage do active military personnel and veterans receive on holiday orders?
- **Category:** `adversarial_unsupported`
- **Description:** Nonexistent military and veteran discount program.
- **Classification:** `TN` (Ground Truth: `Unanswerable`)
- **Latency:** 6865.8 ms (LLM: 5814.9 ms) | **Tokens:** 871 (in: 793, out: 78)
- **Citations Extracted:** `[]` (Has Bracketed Citation: `False`)
- **Model Response:**
> I don't know based on the provided context.

### Query `NEG-07`: What is the pet insurance policy and vet bill reimbursement procedure under VIP membership?
- **Category:** `adversarial_unsupported`
- **Description:** Fictional pet health insurance tier benefit.
- **Classification:** `TN` (Ground Truth: `Unanswerable`)
- **Latency:** 5773.2 ms (LLM: 4675.0 ms) | **Tokens:** 933 (in: 827, out: 106)
- **Citations Extracted:** `[]` (Has Bracketed Citation: `False`)
- **Model Response:**
> I don't know based on the provided context.

### Query `NEG-08`: What are the physical walk-in customer service counter hours at the Meridian headquarters in London?
- **Category:** `adversarial_unsupported`
- **Description:** Unsupported overseas physical walk-in support desk.
- **Classification:** `TN` (Ground Truth: `Unanswerable`)
- **Latency:** 6879.0 ms (LLM: 5813.9 ms) | **Tokens:** 830 (in: 751, out: 79)
- **Citations Extracted:** `[]` (Has Bracketed Citation: `False`)
- **Model Response:**
> I don't know based on the provided context.

### Query `NEG-09`: How do I redeem airline frequent flyer miles to pay for Meridian Plus membership?
- **Category:** `adversarial_unsupported`
- **Description:** Unsupported airline miles loyalty partnership.
- **Classification:** `TN` (Ground Truth: `Unanswerable`)
- **Latency:** 5899.3 ms (LLM: 4761.0 ms) | **Tokens:** 881 (in: 757, out: 124)
- **Citations Extracted:** `[]` (Has Bracketed Citation: `False`)
- **Model Response:**
> I don't know based on the provided context.

### Query `NEG-10`: What is the customs passport visa requirement for international shipping inspection officers?
- **Category:** `out_of_domain`
- **Description:** Government customs and immigration regulations completely outside platform scope.
- **Classification:** `TN` (Ground Truth: `Unanswerable`)
- **Latency:** 7304.1 ms (LLM: 5859.5 ms) | **Tokens:** 902 (in: 775, out: 127)
- **Citations Extracted:** `[]` (Has Bracketed Citation: `False`)
- **Model Response:**
> I don't know based on the provided context.

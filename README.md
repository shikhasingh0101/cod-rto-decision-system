# COD Guard — COD RTO Decision System

A small, runnable prototype for the Associate Software Engineer take-home assignment. It uses the supplied order sample, a supervised RTO-risk model, explainable business rules, an optional LLM address review, a WhatsApp flow simulator, and a founder dashboard.

## 1. What is included

- `app.py` — Streamlit interface with four tabs.
- `src/data.py` — status cleanup, duplicate handling, and feature engineering.
- `src/model.py` — scikit-learn preprocessing pipeline and Random Forest classifier.
- `src/decision.py` — four-action policy, explainable address checks, optional LLM review.
- `src/economics.py` — transparent unit-economics assumptions.
- `data/orders_sample.csv` — the dataset supplied with the assignment.
- `BUSINESS_MEMO.pdf` — founder-facing memo based on the supplied sample.
- `DECISION_LOG.md` and `AI_USAGE_LOG.md` — decision trail and AI-use disclosure.

## 2. Run locally (under 10 minutes)

Python 3.10–3.12 is recommended.

```bash
unzip cod_rto_decision_system.zip
cd cod_rto_decision_system
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\\Scripts\\activate     # Windows PowerShell
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

The app trains the model from `data/orders_sample.csv` on first launch and stores it under `artifacts/`. No API key is needed for the core application.

### Optional LLM review

Copy `.env.example` to `.env`, add an OpenAI API key, then export it before launching:

```bash
export OPENAI_API_KEY="your-key"
export OPENAI_MODEL="gpt-4o-mini"
streamlit run app.py
```

The optional LLM is used only to interpret ambiguous address wording (including Hinglish). It must not decide whether a customer is trustworthy. If the API is unavailable, the application falls back to deterministic checks.

## 3. Decision policy

Default risk bands are configurable in the sidebar:

| Adjusted RTO risk | Action |
|---|---|
| Below 18% | Ship as-is |
| 18% to below 32% | Ship after confirmation |
| 32% to below 48% | Nudge to prepaid |
| 48% and above | Confirmation / hold review; auto-hold only at the stricter 78% threshold with serious data-quality concerns |

First-time customers are protected from automatic cancellation by default. A high model score alone is not enough to cancel a potentially genuine first order.

## 4. Model and data treatment

The model is a Random Forest classifier in a scikit-learn Pipeline. Numeric fields are imputed and scaled; categorical fields are imputed and one-hot encoded. The split is stratified (75% train / 25% test) with a fixed seed. The target is `RTO` versus `DELIVERED`.

Data treatment is intentionally explicit:
- Normalize final-status text before filtering.
- Exclude `IN_TRANSIT` because it has no final outcome yet.
- Drop duplicate `order_id` values, keeping the first occurrence.
- Do not use `final_status`, `order_id`, or phone number as a learned identity feature.
- The file contains COD orders only; it cannot validate prepaid RTO performance.

The displayed metrics are a hold-out result on a small historical sample. They are not a guarantee of future performance. Before live use, run a time-based validation and monitor calibration, false positives, and drift.

## 5. WhatsApp flow

- Confirmation: YES moves to dispatch; NO cancels; unclear text gets one clarification; no response after 24 hours creates a hold/manual follow-up.
- Prepaid nudge: offer 2% off (capped at ₹30), and confirm payment only after a verified payment-gateway webhook.
- Send a reminder after 12 hours. Do not treat silence as consent.

## 6. Dashboard financial scenario

The dashboard defaults to yesterday in the Asia/Kolkata timezone. The supplied sample ends on 30 September 2026, so on 30 October 2026 the default reporting date is 29 October 2026 and the correct sample result is zero matching orders. The dashboard explains the coverage gap instead of presenting September orders as yesterday.

For historical replay, the dashboard exposes assumptions for confirmation effectiveness and prepaid-offer take-up. It estimates:
- Confirmation savings = predicted RTO risk × assumed confirmation effectiveness × ₹169.40.
- Prepaid savings = assumed take-up × max(predicted COD risk − 4%, 0) × ₹169.40.
- Hold/cancel savings = predicted RTO risk × ₹169.40.
- Potential genuine-order profit at risk = (1 − predicted RTO risk) × order value × 62%.
- Net impact = avoided RTO cost − potential delivered profit at risk − prepaid discount cost.

These are scenario estimates, not observed savings. The dashboard displays the assumptions used and excludes payment fees and other costs not provided in the assignment.

## 6. Economics

The assignment's RTO cost inputs imply:
- Forward freight ₹65 + return freight ₹65 + packaging/handling ₹25.
- Expected damaged-stock cost = 8% × ₹180 = ₹14.40.
- Estimated direct cost per RTO = **₹169.40**.

The suggested 2% prepaid discount on the ₹649 reference AOV is ₹12.98. Moving one order from COD (28% assumed RTO) to prepaid (4% assumed RTO) reduces expected RTO cost by 24% × ₹169.40 = ₹40.66. Indicative net benefit before payment fees and other effects = **₹27.68 per converted order**. A ₹30 cap limits the concession on larger baskets.

## 7. Important limitations

This is an assignment prototype, not a production integration. It does not connect to Shopify, a courier API, a live WhatsApp Business account, or a payment gateway. Dashboard savings are modeled estimates, not audited realized savings. The sample covers April–September 2026 and includes no prepaid rows. Small city/courier groups should not be blocked based on raw rates alone.

## 8. Suggested production next steps

1. Shopify webhook ingestion and idempotent order events.
2. WhatsApp Business API with consent, templates, opt-out, and delivery receipts.
3. Courier/pincode-level outcomes with minimum sample-size and confidence controls.
4. Time-based model validation, calibration, and a human override queue.
5. A/B test confirmation and discount policies against contribution margin, not just RTO rate.
# cod-rto-decision-system

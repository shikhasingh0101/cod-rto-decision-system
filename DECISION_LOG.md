# Decision log

1. **Treat the target as a binary final outcome.** `DELIVERED` is 0 and `RTO` is 1. `IN_TRANSIT` is not a completed outcome, so it is excluded rather than incorrectly labeled.
2. **Normalize status strings.** The file contains casing/whitespace variants (`Rto`, `RTO `, `delivered`); normalizing avoids splitting the same business outcome into multiple categories.
3. **Deduplicate by `order_id`.** The CSV has 30 repeated order IDs. Keeping the first record avoids counting the same order more than once. In production, conflicting duplicates should be reconciled against the order event history rather than silently accepted.
4. **Use a model for ranking, rules for actions.** A Random Forest estimates RTO probability; configurable thresholds map risk to the four requested actions. This separates prediction from business policy.
5. **Keep first-time customer protection.** Operations wants fewer avoidable RTOs, while Marketing points out acquisition cost and trust barriers. A high score alone does not automatically cancel a first purchase; ask for confirmation instead.
6. **Reserve the LLM for language ambiguity.** Regex/code handles phone length, missing text, and obvious placeholders cheaply and consistently. An optional LLM can interpret Hinglish landmarks and explain missing address details. It does not assess customer character or replace the numeric model.
7. **Offer 2% prepaid discount, capped at ₹30.** At ₹649 AOV, the typical discount is ₹12.98. The assignment's 28% COD versus 4% prepaid RTO assumptions imply ₹40.66 expected RTO-cost reduction per conversion, leaving ₹27.68 before fees.
8. **Do not interpret silence as confirmation.** Send a reminder, then hold for manual follow-up after 24 hours. This avoids dispatching an unconfirmed COD order or silently cancelling a potentially genuine customer.
9. **Show estimates as estimates.** A static historical CSV cannot prove causal savings from an intervention. The dashboard labels its counterfactual figures accordingly.
10. **Avoid pincode bans from small samples.** Geographic rates are useful for investigation, but should have minimum sample-size and confidence checks before any policy is applied.

import os, re, json
from dataclasses import dataclass
from src.data import make_features, FEATURES

DEFAULTS = {
    "ship_max_risk": 0.18,
    "confirm_max_risk": 0.32,
    "prepaid_max_risk": 0.48,
    "auto_hold_min_risk": 0.78,
    "prepaid_discount_pct": 2.0,
    "first_order_protection": True,
    "confirmation_timeout_hours": 24
}

def address_checks(name, address, phone):
    """Fast, explainable checks. These do not pretend to understand every local-language address."""
    name_s = str(name or "").strip()
    addr = str(address or "").strip()
    digits = re.sub(r"\D", "", str(phone or ""))
    bad_names = {"test", "asdf", "qwerty", "unknown", "na", "none"}
    name_flag = (not name_s) or name_s.lower() in bad_names or len(name_s) < 3
    landmark_words = ["near", "opp", "opposite", "mandir", "school", "bank", "chowk",
                      "temple", "landmark", "ke paas", "saamne", "station", "road", "nagar"]
    has_landmark = any(w in addr.lower() for w in landmark_words)
    has_number = bool(re.search(r"\d", addr))
    short_address = len(addr) < 12
    phone_ok = len(digits) == 10
    flags = []
    if name_flag: flags.append("The customer name looks incomplete or placeholder-like.")
    if short_address: flags.append("The address is very short.")
    if not has_landmark and not has_number: flags.append("The address has no obvious house number or landmark.")
    if not phone_ok: flags.append("The phone number is not a 10-digit Indian mobile number.")
    quality_risk = min(0.24, 0.08*name_flag + 0.07*short_address + 0.05*(not has_landmark and not has_number) + 0.08*(not phone_ok))
    return {"name_flag":bool(name_flag),"short_address":bool(short_address),"has_landmark":bool(has_landmark),
            "has_number":bool(has_number),"phone_ok":bool(phone_ok),"flags":flags,"risk_adjustment":quality_risk}

def llm_address_review(name, address):
    """Optional LLM review for ambiguous Hinglish/locality wording; deterministic checks remain the fallback."""
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key, timeout=12.0, max_retries=1)
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            response_format={"type":"json_object"},
            messages=[
                {"role":"system","content":"Review Indian delivery address text. Do not infer whether a person is trustworthy. Return JSON with address_usable (boolean), missing_detail (short string), and confidence (0 to 1). Understand Hinglish phrases such as 'mandir ke saamne' and 'ke paas'. Do not make up missing details."},
                {"role":"user","content":json.dumps({"customer_name":name,"address":address})}
            ]
        )
        return json.loads(response.choices[0].message.content)
    except Exception:
        return None

def decide_order(order, model, config=None, use_llm=False):
    cfg = {**DEFAULTS, **(config or {})}
    import pandas as pd
    row = pd.DataFrame([order])
    engineered = make_features(row)[FEATURES]
    ml_risk = float(model.predict_proba(engineered)[0,1])
    checks = address_checks(order.get("customer_name"), order.get("address"), order.get("phone"))
    llm = llm_address_review(order.get("customer_name"), order.get("address")) if use_llm else None
    risk = min(0.99, max(0.01, ml_risk + checks["risk_adjustment"]))
    if llm and llm.get("address_usable") is False and float(llm.get("confidence",0)) >= 0.75:
        risk = min(0.99, risk + 0.06)
    first_order = int(order.get("customer_prior_orders",0) or 0) == 0
    severe = checks["name_flag"] and checks["short_address"] and not checks["phone_ok"]
    if risk < cfg["ship_max_risk"]:
        action = "Ship as-is"
        reason = "The estimated RTO risk is low and the basic contact/address checks look acceptable."
    elif risk < cfg["confirm_max_risk"]:
        action = "Ship after confirmation"
        reason = "The order has some delivery uncertainty, so confirm the customer's intent before dispatch."
    elif risk < cfg["prepaid_max_risk"]:
        action = "Nudge to prepaid"
        reason = "The estimated RTO risk is elevated; offer a small prepaid discount instead of rejecting the order."
    elif risk >= cfg["auto_hold_min_risk"] and (not (first_order and cfg["first_order_protection"]) or severe):
        action = "Hold / cancel"
        reason = "The estimated risk is very high and the contact/address checks show serious issues."
    else:
        action = "Ship after confirmation"
        reason = "The risk is too high for automatic dispatch, but there is not enough evidence to cancel a potentially genuine sale."
    if first_order and cfg["first_order_protection"] and action == "Hold / cancel":
        action = "Ship after confirmation"
        reason = "This is a first order, so the system protects the potential new customer and asks for confirmation rather than cancelling."
    return {"action":action,"ml_rto_probability":ml_risk,"adjusted_risk":risk,
            "reason":reason,"address_checks":checks,"llm_review":llm}

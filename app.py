from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import streamlit as st
import pandas as pd
import numpy as np

from src.data import load_orders
from src.model import train_and_save
from src.decision import decide_order
from src.economics import RTO_COST, DELIVERED_GROSS_PROFIT

st.set_page_config(page_title="COD Guard | RTO Decision System", page_icon="📦", layout="wide")
st.title("📦 COD Guard")
st.caption("A COD order decision-support prototype: reduce avoidable RTO costs without automatically rejecting genuine customers.")

with st.sidebar:
    st.header("Decision settings")
    st.caption("Business rules can be adjusted here without changing Python code.")
    cfg = {
        "ship_max_risk": st.slider("Ship as-is below risk", 0.05, 0.40, 0.18, 0.01),
        "confirm_max_risk": st.slider("Confirmation below risk", 0.15, 0.60, 0.32, 0.01),
        "prepaid_max_risk": st.slider("Prepaid nudge below risk", 0.25, 0.75, 0.48, 0.01),
        "auto_hold_min_risk": st.slider("Strict hold threshold", 0.50, 0.99, 0.78, 0.01),
        "prepaid_discount_pct": st.slider("Prepaid discount (%)", 0.0, 5.0, 2.0, 0.25),
        "prepaid_discount_cap": st.number_input(
            "Maximum prepaid discount (₹)", min_value=0, max_value=100, value=30, step=5
        ),
        "first_order_protection": st.checkbox("Protect first-time customers", value=True),
        "confirmation_timeout_hours": 24,
    }
    use_llm = st.checkbox(
        "Use optional LLM address review",
        value=False,
        help="Requires OPENAI_API_KEY. The LLM reviews ambiguous address wording only; it does not judge customer trustworthiness."
    )
    st.caption("Keep thresholds in increasing order: ship < confirm < prepaid < hold.")
    thresholds_valid = (
        cfg["ship_max_risk"] < cfg["confirm_max_risk"]
        < cfg["prepaid_max_risk"] < cfg["auto_hold_min_risk"]
    )

@st.cache_resource
def load_model_and_metrics():
    return train_and_save()

model, metrics, clean = load_model_and_metrics()
clean_dates = pd.to_datetime(clean["order_datetime"], errors="coerce")
first_date = clean_dates.min().date()
last_date = clean_dates.max().date()

tab1, tab2, tab3, tab4 = st.tabs([
    "Morning dashboard", "Check an order", "WhatsApp simulator", "Model & data notes"
])

with tab1:
    st.subheader("Yesterday's operating picture")
    india_today = datetime.now(ZoneInfo("Asia/Kolkata")).date()
    yesterday = india_today - timedelta(days=1)
    st.caption(f"Today: {india_today:%d %b %Y}  |  Reporting date: {yesterday:%d %b %Y}")
    st.info(
        f"The uploaded CSV covers {first_date:%d %b %Y} to {last_date:%d %b %Y}. "
        "It is historical sample data, not a live Shopify feed. The dashboard defaults to yesterday; "
        "if the CSV has no orders for that date, it will show zero rather than inventing activity."
    )
    selected = st.date_input("Reporting date", value=yesterday)
    day_mask = clean_dates.dt.date == selected
    day = clean.loc[day_mask].copy()

    if not thresholds_valid:
        st.error("Please set the thresholds in increasing order: ship < confirmation < prepaid < hold.")
    else:
        replay = []
        for _, row in day.iterrows():
            result = decide_order(row.to_dict(), model, cfg, use_llm=False)
            replay.append({**row.to_dict(), **result})
        replay_df = pd.DataFrame(replay)

        action_labels = [
            "Ship as-is", "Ship after confirmation",
            "Nudge to prepaid", "Hold / cancel"
        ]
        bucket_counts = (
            replay_df["action"].value_counts()
            if not replay_df.empty else pd.Series(dtype=int)
        )
        cols = st.columns(4)
        for col, label in zip(cols, action_labels):
            col.metric(label, int(bucket_counts.get(label, 0)))

        if replay_df.empty:
            st.warning(
                f"No orders in the supplied dataset match {selected:%d %b %Y}. "
                "The correct count for this sample date is zero. Select a date within the dataset range "
                "to replay historical orders. Connect Shopify webhooks to populate real daily activity."
            )
            st.markdown("#### Financial impact")
            empty1, empty2, empty3, empty4 = st.columns(4)
            empty1.metric("Estimated RTO cost avoided", "—")
            empty2.metric("Potential delivered profit at risk", "—")
            empty3.metric("Prepaid discount cost", "—")
            empty4.metric("Indicative net impact", "—")
            st.caption(
                "Financial impact is unavailable because no orders exist for the selected date. "
                "No savings, discounts, or potential losses have been assumed."
            )
        else:
            st.markdown("#### Financial impact assumptions")
            st.caption(
                "These are scenario assumptions, not measured outcomes. Adjust them to test different "
                "operating scenarios; validate them in a pilot before using them for financial reporting."
            )
            s1, s2 = st.columns(2)
            confirm_effectiveness = s1.slider(
                "Confirmation effectiveness (%)", 0, 100, 60, 5,
                help="Share of risk-weighted RTO exposure assumed to be prevented after a successful confirmation."
            ) / 100
            prepaid_takeup = s2.slider(
                "Prepaid offer take-up (%)", 0, 100, 30, 5,
                help="Assumed share of customers who accept the prepaid offer."
            ) / 100
            discount_cap = cfg["prepaid_discount_cap"]

            rto_saving = 0.0
            discount_cost = 0.0
            confirmation_prevented = 0.0
            prepaid_prevented = 0.0

            for _, order in replay_df.iterrows():
                risk = float(order["adjusted_risk"])
                action = order["action"]
                order_value = float(order.get("order_value", 649) or 649)

                if action == "Ship after confirmation":
                    prevented = risk * confirm_effectiveness
                    confirmation_prevented += prevented
                    rto_saving += prevented * RTO_COST

                elif action == "Nudge to prepaid":
                    # Only converted customers receive the prepaid risk reduction.
                    prevented = prepaid_takeup * max(risk - 0.04, 0.0)
                    prepaid_prevented += prevented
                    rto_saving += prevented * RTO_COST
                    discount_cost += prepaid_takeup * min(
                        order_value * cfg["prepaid_discount_pct"] / 100,
                        discount_cap
                    )

                elif action == "Hold / cancel":
                    # Holding/cancelling avoids the expected RTO logistics cost,
                    # but may sacrifice the gross profit from a genuine order.
                    rto_saving += risk * RTO_COST

            cancelled = replay_df[replay_df["action"] == "Hold / cancel"]
            lost_profit = float(sum(
                (1 - float(row["adjusted_risk"]))
                * float(row.get("order_value", 649) or 649)
                * 0.62
                for _, row in cancelled.iterrows()
            ))
            total_benefit = rto_saving
            net_impact = total_benefit - lost_profit - discount_cost

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Estimated RTO cost avoided", f"₹{rto_saving:,.0f}")
            k2.metric("Potential delivered profit at risk", f"₹{lost_profit:,.0f}")
            k3.metric("Prepaid discount cost", f"₹{discount_cost:,.0f}")
            k4.metric("Indicative net impact", f"₹{net_impact:,.0f}")

            st.caption(
                f"Calculation: confirmation prevention = risk × {confirm_effectiveness:.0%}; "
                f"prepaid prevention = take-up × max(COD risk − 4%, 0); "
                f"hold/cancel saving = risk × ₹{RTO_COST:.2f}; "
                "lost profit = (1 − risk) × order value × 62%; "
                "net impact = avoided RTO cost − lost delivered profit − prepaid discount cost. "
                "The ₹169.40 RTO cost includes ₹65 forward freight, ₹65 return freight, ₹25 handling "
                "and ₹14.40 expected damaged-stock cost. These estimates assume the stated intervention "
                "effectiveness and do not include payment fees or other operational costs."
            )

            st.markdown("#### Order-level decision replay")
            st.dataframe(
                replay_df[[
                    "order_id", "city", "order_value", "action",
                    "ml_rto_probability", "adjusted_risk", "reason"
                ]],
                use_container_width=True, hide_index=True
            )

            st.markdown("#### Decision mix")
            chart_data = pd.DataFrame({
                "Decision": action_labels,
                "Orders": [int(bucket_counts.get(label, 0)) for label in action_labels]
            }).set_index("Decision")
            st.bar_chart(chart_data)

with tab2:
    st.subheader("Evaluate a new COD order")
    st.write("Enter the order details to receive a risk estimate, recommended action, and plain-English explanation.")
    with st.form("order_form"):
        a, b = st.columns(2)
        name = a.text_input("Customer name", "Aarav Sharma")
        phone = b.text_input("Phone", "9876543210")
        address = st.text_area("Delivery address", "Flat 12, Green Park, near Hanuman Mandir")
        city = a.text_input("City", "Mumbai")
        state = b.text_input("State", "Maharashtra")
        pincode = a.text_input("Pincode", "400001")
        value = b.number_input("Order value (₹)", min_value=0, value=649, step=1)
        items = a.text_input("Items", "Satin Bonnet")
        num_items = b.number_input("Number of items", min_value=1, max_value=20, value=1)
        prior = a.number_input("Previous orders", min_value=0, max_value=100, value=0)
        prior_rto = b.number_input("Previous RTOs", min_value=0, max_value=100, value=0)
        source = a.selectbox("Acquisition source", [
            "meta_ads", "google_ads", "direct", "instagram_organic",
            "influencer", "whatsapp_broadcast"
        ])
        courier = b.selectbox("Courier", [
            "Delhivery", "BlueDart", "Ecom Express", "Shadowfax", "Xpressbees"
        ])
        submitted = st.form_submit_button("Evaluate order")

    if submitted:
        if not thresholds_valid:
            st.error("Correct the threshold order in the sidebar first.")
        else:
            try:
                pin = int(pincode)
            except ValueError:
                pin = -1
            order = {
                "order_id": "NEW",
                "order_datetime": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
                "customer_name": name, "phone": phone, "address": address,
                "city": city, "state": state, "pincode": pin,
                "order_value": value, "items": items, "num_items": num_items,
                "customer_prior_orders": prior, "customer_prior_rto": prior_rto,
                "utm_source": source, "courier": courier, "payment_method": "COD"
            }
            result = decide_order(order, model, cfg, use_llm)
            st.success(result["action"])
            st.write(result["reason"])
            st.metric("Estimated RTO risk", f"{result['adjusted_risk']:.1%}")
            if result["address_checks"]["flags"]:
                st.warning("Address/contact checks: " + " ".join(result["address_checks"]["flags"]))
            if result["llm_review"]:
                st.write("Optional LLM address review:", result["llm_review"])
            st.caption(
                "This is a decision-support recommendation. Keep a human override for unusual, "
                "high-value, or disputed orders."
            )

with tab3:
    st.subheader("WhatsApp flow mock")
    st.write("This is a simulator, not a live WhatsApp connection. In production, send approved templates through WhatsApp Business API.")
    kind = st.radio("Message type", ["Confirmation", "Prepaid nudge"], horizontal=True)
    language = st.radio("Language", ["Hinglish", "English"], horizontal=True)
    demo_order_value = st.number_input("Example order value (₹)", min_value=1, value=649, step=1)
    demo_discount = min(
        demo_order_value * cfg["prepaid_discount_pct"] / 100,
        cfg["prepaid_discount_cap"]
    )
    if kind == "Confirmation":
        message = (
            f"Namaste! Aapka order ₹{demo_order_value} ka receive hua hai. Dispatch se pehle please YES reply karke confirm karein. Agar order place nahi kiya, NO reply karein. 24 hours mein reply na milne par order hold rahega."
            if language == "Hinglish" else
            f"Hello! We received your COD order worth ₹{demo_order_value}. Please reply YES to confirm before dispatch, or NO if you did not place it. If we do not hear from you within 24 hours, the order will be put on hold."
        )
    else:
        message = (
            f"Aapka order ready hai! Online prepaid payment par {cfg['prepaid_discount_pct']:.2f}% off (maximum ₹{cfg['prepaid_discount_cap']}, approx. ₹{demo_discount:.2f}) paayein. Secure payment link: [mock-link]. Prepaid optional hai; aap COD bhi continue kar sakte hain."
            if language == "Hinglish" else
            f"Your order is ready! Pay online and get {cfg['prepaid_discount_pct']:.2f}% off (maximum ₹{cfg['prepaid_discount_cap']}, approx. ₹{demo_discount:.2f}). Secure payment link: [mock-link]. Prepaid is optional; you can continue with COD."
        )
    st.code(message)
    reply = st.text_input("Simulate customer reply", "yes")
    if st.button("Process reply"):
        r = reply.strip().lower()
        if r in ["yes", "y", "haan", "ha", "confirm", "confirmed"]:
            st.success("Confirmed. Move the order to the dispatch queue.")
        elif r in ["no", "n", "nahi", "cancel"]:
            st.warning("Customer declined. Cancel the order and record the reason.")
        elif r in ["paid", "payment done", "prepaid"]:
            st.success("Do not mark as prepaid until the payment gateway confirms successful payment.")
        else:
            st.info("Reply not understood. Send one clarification message; do not dispatch until a valid confirmation is received.")
    st.caption(
        "Suggested timing: send immediately after order placement, one reminder after 12 hours, "
        "then hold and create a manual follow-up task at 24 hours. Silence is not consent."
    )

with tab4:
    st.subheader("Model performance on a held-out test set")
    c1, c2, c3 = st.columns(3)
    c1.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
    c2.metric("Accuracy", f"{metrics['accuracy']:.1%}")
    c3.metric("RTO recall", f"{metrics['recall_rto']:.1%}")
    st.write(
        f"Test rows: {metrics['test_rows']}. The split is stratified with a fixed random seed. "
        "These are sample hold-out metrics, not a guarantee of future performance."
    )
    st.subheader("Data quality notes")
    st.write(f"- {len(clean):,} unique final-outcome orders are used for training.")
    st.write("- Status text is normalized before training.")
    st.write("- Duplicate order IDs are removed; IN_TRANSIT rows are excluded because they have no final outcome.")
    st.write("- The supplied dataset contains COD orders only, so it cannot independently validate prepaid outcomes.")
    st.write("- The optional LLM only reviews ambiguous address wording. Numeric risk scoring and business thresholds remain separate.")
    st.write(f"- Historical date coverage in this file: {first_date:%d %b %Y} to {last_date:%d %b %Y}.")

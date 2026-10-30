RTO_COST = 65 + 65 + 25 + (0.08 * 180)  # forward + return + handling + expected damaged stock
AOV = 649
GROSS_MARGIN = 0.62
DELIVERED_GROSS_PROFIT = AOV * GROSS_MARGIN
PREPAID_RTO_RATE = 0.04
COD_RTO_RATE = 0.28
DISCOUNT_RATE = 0.02

def expected_prepaid_benefit(aov=AOV, discount_rate=DISCOUNT_RATE):
    avoided_rto_cost = (COD_RTO_RATE - PREPAID_RTO_RATE) * RTO_COST
    discount = aov * discount_rate
    return {"avoided_rto_cost":avoided_rto_cost, "discount":discount, "net_expected_benefit":avoided_rto_cost-discount}

def estimate_avoided_rto_savings(avoided_rtos):
    return float(avoided_rtos) * RTO_COST

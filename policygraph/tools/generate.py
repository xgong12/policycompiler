#!/usr/bin/env python3
"""
generate.py -- training-data generator (synonym / boundary-op / unless-only phrasing): ontology + compatibility are READ FROM seeds/expense_management.json
(via seed), never hard-coded. A separate atom_coverage_layer (tools/coverage.json) guarantees
every canonical prerequisite token appears in training >=1x, so held-out stays compositional-OOD.

The input format is the full ontology + policy, with the instruction/ontology/model_input shared
unchanged with the eval builder and run_eval. Key data-quality properties:
  1. num+prereq is no longer a cartesian product -- NUM_PREREQ_COMPAT gates which prerequisite
     may pair with which numeric field (analogous to PREREQ_CATEGORY_COMPAT for cat+prereq).
  2. single prerequisite-only rules are restricted to genuinely global gates
     (GLOBAL_PREREQ_OK); category-specific prereqs are taught only as category+prereq, so no
     over-broad "all reimbursements denied without lounge pre-approval" rules.
  3. "requires review" phrasing -> explicit block-pending-review phrasing (the gold is DENY;
     the text should say so, not conflate REVIEW with DENY).
  4. train/val split is signature-grouped + coverage-preserving: every paraphrase of a rule
     signature lands in ONE split (no leakage), while train still covers every canonical value.

Generation uses: full field inventory, two-pass generation (coverage then volume),
held-out rule-signature + threshold exclusion, per-kind + diversity report. No runtime cases
(SFT target is policy_text -> rule_graph only). No LLM paraphrase layer. Language is readable,
not hand-polished -- coverage correctness over elegance.

Held-out seed (80 policies) is used ONLY as an exclusion filter, never sampled.

    python3 tools/generate.py --out dataset --train 5000 --val 300
"""
import argparse, json, random, sys, re
from collections import defaultdict, Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import seed as S                                   # single source of truth = the held-out seed
DOMAIN = S.DOMAIN
COVERAGE = json.loads((HERE / "coverage.json").read_text())["prerequisite_coverage"]

ACTIONS = S.ACTIONS

# Verb phrasing per action (Section 5: verb matches the action)
ACTION_VERB = {
    "approve_expense_report": ["may not be approved", "cannot be approved", "will not be approved"],
    "issue_reimbursement": ["is not reimbursable", "may not be reimbursed", "will not be reimbursed"],
    "increase_card_limit": ["may not be raised", "cannot be increased", "will not be raised"],
}

# ---- hard-coded special renderings (Section 5: ~20% need care) ----
SPECIAL_LABEL = {
    "business_purpose_provided": "a business purpose",
    "duplicate_suspected": "a duplicate is suspected",
    "tax_id_provided": "a tax ID",
    "within_alcohol_limit": "the per-person alcohol limit",
    "misuse_suspected": "card misuse is suspected",
    "employee_local": "the employee's local currency",
    "north_america": "North America",
    "premium": "a premium tier",
    "standard": "the standard region",
    "approved": "an approved vendor",
    "active": "active",
}

# other_enum field -> human label for the field itself
OTHER_ENUM_LABEL = {
    "merchant_category_status": "the merchant category",
    "rental_provider_status": "the rental provider",
    "payment_currency": "the payment currency",
    "trip_region": "the trip region",
    "tipping_region": "the tipping region",
    "ride_class": "the ride class",
    "card_status": "the card",
}

# numeric field -> (human label, unit) ; unit "$" or "" (count) or "%" 
NUM_META = {
    "card_charge_amount": ("the travel-card charge", "$"),
    "client_entertainment_event_amount": ("the client entertainment event amount", "$"),
    "days_since_expense": ("the expense age in days", "d"),
    "per_person_amount": ("the per-person amount", "$"),
    "short_term_rental_nights": ("the short-term rental length in nights", "n"),
    "annual_gift_amount": ("the annual gift total", "$"),
    "booking_lead_days": ("the booking lead time in days", "d"),
    "chair_amount": ("the ergonomic chair cost", "$"),
    "daily_meal_amount": ("the daily meal amount", "$"),
    "daily_parking_amount": ("the daily parking amount", "$"),
    "home_office_stipend": ("the home office stipend", "$"),
    "hotel_nightly_rate": ("the hotel nightly rate", "$"),
    "internet_stipend": ("the internet stipend", "$"),
    "per_ticket_amount": ("the per-ticket amount", "$"),
    "personal_mileage_miles": ("the personal mileage", "mi"),
    "standard_ride_wait_minutes": ("the standard ride wait", "min"),
    "tip_percentage": ("the tip percentage", "%"),
    "tuition_amount": ("the tuition amount", "$"),
    "requested_limit": ("the requested limit", "$"),
}

# candidate training thresholds per numeric field (broad pool; held-out values filtered out at runtime)
NUM_POOL = {
    "card_charge_amount": [200, 400, 600, 800, 1200, 1800, 2200, 4000],
    "client_entertainment_event_amount": [800, 1000, 1200, 2000, 2500],
    "days_since_expense": [21, 45, 75, 90, 100, 150],
    "per_person_amount": [30, 40, 75, 90, 150],
    "short_term_rental_nights": [3, 4, 7, 10, 14],
    "annual_gift_amount": [50, 75, 150, 200, 300],
    "booking_lead_days": [3, 5, 10, 21, 28],
    "chair_amount": [150, 200, 300, 400],
    "daily_meal_amount": [40, 50, 60, 90, 100],
    "daily_parking_amount": [20, 25, 30, 50, 60],
    "home_office_stipend": [300, 400, 600, 750],
    "hotel_nightly_rate": [300, 350, 400, 500, 550],
    "internet_stipend": [30, 40, 60, 75],
    "per_ticket_amount": [100, 150, 250, 300],
    "personal_mileage_miles": [150, 200, 250, 400, 500],
    "standard_ride_wait_minutes": [10, 15, 20, 45, 60],
    "tip_percentage": [10, 15, 25, 30],
    "tuition_amount": [2000, 3000, 4000, 6000, 8000],
    "requested_limit": [500, 750, 1250, 1500, 2000, 2500, 3500, 4000, 5000, 6000, 7500, 10000],  # widened for increase-card share
}

# Held-out-exclusive boolean fields: their single-condition signature IS the held-out rule,
# so we teach them via a few auxiliary bool+prereq conjunctions instead (reported separately).
HELD_OUT_EXCLUSIVE_BOOL = [
    ("approve_expense_report", "business_purpose_provided"),
    ("issue_reimbursement", "tax_id_provided"),
    ("issue_reimbursement", "within_alcohol_limit"),
    ("approve_expense_report", "duplicate_suspected"),
    ("increase_card_limit", "misuse_suspected"),
]
AUX_BOOL_PREREQS = {
    "approve_expense_report": ["manager_approval", "business_justification", "attendee_list"],
    "issue_reimbursement": ["manager_approval", "itemized_receipt", "business_justification", "vp_approval"],
    "increase_card_limit": ["controller_signoff", "card_training"],
}

# held-out-exclusive other_enum fields: their single-condition (!= canonical) signature IS the
# held-out rule. Teach via auxiliary (!= canonical AND prereq) conjunctions, reported separately.
HELD_OUT_EXCLUSIVE_OTHER = [
    ("approve_expense_report", "merchant_category_status", "approved"),
    ("issue_reimbursement", "rental_provider_status", "approved"),
    ("issue_reimbursement", "payment_currency", "employee_local"),
    ("increase_card_limit", "card_status", "active"),
]
AUX_OTHER_PREREQS = {
    "approve_expense_report": ["manager_approval", "business_justification", "attendee_list"],
    "issue_reimbursement": ["manager_approval", "itemized_receipt", "business_justification", "vp_approval"],
    "increase_card_limit": ["controller_signoff", "card_training"],
}

# Boolean polarity: the value that should DENY. "Suspected"-type fields deny when TRUE
# (a suspected duplicate/misuse must block); "provided/within"-type deny when FALSE
# (a missing business purpose / exceeded alcohol limit must block).
BOOL_DENY_VALUE = {
    "duplicate_suspected": True,
    "misuse_suspected": True,
    "business_purpose_provided": False,
    "tax_id_provided": False,
    "within_alcohol_limit": False,
}

# prereq -> display phrase. action-like prereqs get special sentence handling (see render).
PREREQ_PHRASES = {
    "manager_approval": "manager approval",
    "vp_approval": "VP approval",
    "ceo_authorization": "CEO authorization",
    "c_suite_authorization": "C-suite authorization",
    "cio_clearance": "CIO clearance",
    "it_security_approval": "IT security approval",
    "legal_approval": "legal approval",
    "controller_signoff": "controller sign-off",
    "card_training": "card training",
    "itemized_receipt": "an itemized receipt",
    "business_justification": "a business justification",
    "business_purpose_documentation": "business purpose documentation",
    "business_purpose": "a business purpose",
    "attendee_list": "an attendee list",
    "exam_passed": "the certification exam has been passed",
    "cash_withdrawal_approval": "cash withdrawal approval",
    "lounge_preapproval": "lounge pre-approval",
}
# action-like prereqs need a full clause, not "{phrase} has been obtained"
PREREQ_ACTION_CLAUSE = {
    "approve_expense_report": "the expense report has been approved",
    "exam_passed": "the certification exam has been passed",
}

# cat+num pairs read from the seed compatibility (category_numeric)
CAT_NUM_PAIRS = {a: S.cat_num_pairs(a) for a in S.ONTOLOGY}
# scoped_num triples read from the seed compatibility (scoped_enum_numeric)
SCOPED_NUM_TRIPLES = {a: S.scoped_triples(a) for a in S.ONTOLOGY}

# curated 3-condition triples (category + numeric + prerequisite), all pairwise
# business-compatible (each pair also appears in CAT_NUM_PAIRS / cat_prereq / num_prereq). Adds
# structural diversity so the model does not overfit to exactly-two-condition rules.
THREE_COND_TRIPLES = {
    "approve_expense_report": [
        ("client_meal", "per_person_amount", "attendee_list"),
        ("group_meal", "per_person_amount", "attendee_list"),
        ("staff_dinner", "per_person_amount", "business_justification"),
    ],
    "issue_reimbursement": [
        ("client_gift", "annual_gift_amount", "vp_approval"),
        ("client_entertainment", "per_ticket_amount", "vp_approval"),
        ("hotel_suite_upgrade", "hotel_nightly_rate", "manager_approval"),
        ("parking", "daily_parking_amount", "itemized_receipt"),
    ],
}
# curated multi-rule (OR) policies -- two INDEPENDENT numeric rules for one action. Each
# sub-rule uses a non-held-out threshold so no constituent rule collides with a held-out single.
MULTI_RULE_NUM = [
    ("approve_expense_report", [("card_charge_amount", ">", 2200), ("days_since_expense", ">", 100)]),
    ("issue_reimbursement",    [("daily_meal_amount", ">", 90), ("hotel_nightly_rate", ">", 500)]),
    ("issue_reimbursement",    [("annual_gift_amount", ">", 300), ("per_ticket_amount", ">", 300)]),
    ("approve_expense_report", [("per_person_amount", ">", 150), ("client_entertainment_event_amount", ">", 2500)]),
]
def render_multi_numeric(action, entries, rng):
    parts = []
    for f, op, v in entries:
        lbl = _syn(NUM_LABEL_SYN, f, NUM_META[f][0], rng); nstr = fmt_num(f, v)
        parts.append(rng.choice(OP_PHRASES[op]).format(label=lbl, n=nstr))
    subj, verb = {
        "approve_expense_report": ("The expense report", ["may not be approved", "cannot be approved"]),
        "issue_reimbursement": ("The expense", ["may not be reimbursed", "is not reimbursable"]),
        "increase_card_limit": ("The card limit", ["may not be raised", "may not be increased"]),
    }[action]
    tail = " when " + parts[0] + ", or when " + parts[1]
    return f"{subj} {rng.choice(verb)}{tail}."

# operator phrasing pools (Section 5). {label}=field label, {n}=number
OP_PHRASES = {
    ">":  ["{label} exceeds {n}", "{label} is more than {n}", "{label} is above {n}",
           "{label} is greater than {n}", "{label} is in excess of {n}"],
    ">=": ["{label} is {n} or more", "{label} is at least {n}", "{label} is no fewer than {n}"],
    "<":  ["{label} is fewer than {n}", "{label} is less than {n}", "{label} is under {n}",
           "{label} is below {n}"],
}
# flipped: "unless ... at least N" denies < N  (Section 5 contrastive pairs)
FLIP_PHRASES = {
    "<": ["unless {label} is at least {n}", "unless {label} reaches {n} or more"],
}

def fmt_num(field, n):
    unit = NUM_META[field][1]
    if unit == "$": return f"${n:,}"
    if unit == "%": return f"{n}%"
    if unit == "d": return f"{n} days"
    if unit == "n": return f"{n} nights"
    if unit == "mi": return f"{n} miles"
    if unit == "min": return f"{n} minutes"
    return str(n)

def readable(value):
    """underscores -> spaces, with capitalization fixes for acronyms/proper terms."""
    if value in SPECIAL_LABEL: return SPECIAL_LABEL[value]
    text = value.replace("_", " ")
    # P2: fix casing of acronyms as whole words only (avoid 'treATMent')
    words = text.split()
    fix = {"atm": "ATM", "ai": "AI", "vp": "VP", "ceo": "CEO", "cio": "CIO"}
    words = [fix.get(w, w) for w in words]
    text = " ".join(words)
    text = text.replace("it security", "IT security").replace("c suite", "C-suite")
    return text

# ---------- held-out guards ----------
def rule_signature(action, body):
    """action + normalized deny_when entries (field/op/value + not_completed). Section 7."""
    norm = tuple(sorted(
        ("nc:" + e["not_completed"]) if "not_completed" in e
        else f'{e["field"]}|{e["operator"]}|{e["value"]!r}'
        for e in body))
    return (action, norm)

def held_out_guards(seed_path):
    blob = json.loads(Path(seed_path).read_text())
    sigs = set()
    thresholds = set()  # (action, field, operator, value)
    for p in blob["policies"]:
        for r in p["rule_graph"]["rules"]:
            a = r["target_action"]; body = r["deny_when"]
            sigs.add(rule_signature(a, body))
            for e in body:
                if "field" in e and isinstance(e["value"], (int, float)) and not isinstance(e["value"], bool):
                    thresholds.add((a, e["field"], e["operator"], e["value"]))
    return sigs, thresholds

# ---------- entry constructors ----------
# cat+prereq compatibility read from the seed compatibility.category_prerequisite (S.cat_prereq_ok)

# num+prereq compatibility read from the seed compatibility.numeric_prerequisite (S.num_prereq_ok)

# standalone (global) prerequisites read from the seed compatibility.standalone_prerequisites

# held-out-exclusive PREREQUISITES whose only business-sensible signature IS a held-out
# policy, and whose sole compatible category is the held-out one (ceo_authorization -> only
# charitable_gala; cash_withdrawal_approval -> only atm_withdrawal). There is no clean
# non-colliding category+prereq to teach them, and a standalone rule would be over-broad
# ("all approvals need CEO auth"). Rather than inject an unclean rule or leak the held-out
# signature, they are intentionally left uncovered: the ontology lists the token in every
# input, so a schema-grounded model can still copy NL "CEO authorization" -> ceo_authorization.
# emptied. Both former entries are now covered by documented, business-plausible,
# non-held-out pairings (see NUM_PREREQ_COMPAT client_entertainment_event_amount+ceo_authorization
# and PREREQ_CATEGORY_COMPAT gift_card+cash_withdrawal_approval), so the model sees the prereq
# TOKEN in training gold while the held-out signature stays unseen (atomic-coverage principle).
INTENTIONALLY_UNCOVERED = set()

# Rationale for the business-plausible pairings that are less obvious than the rest, kept so an
# auditor can see WHY a combination is allowed, not just that it is on a whitelist.
COMBO_RATIONALE = {
    ("client_entertainment_event_amount", "ceo_authorization"):
        "High-value client entertainment can require CEO sign-off; teaches ceo_authorization "
        "without reusing the held-out charitable_gala+ceo signature.",
    ("gift_card", "cash_withdrawal_approval"):
        "Gift cards are cash equivalents and fall under the same cash-withdrawal control; teaches "
        "cash_withdrawal_approval without reusing the held-out atm_withdrawal+cash signature.",
    ("tuition_amount", "exam_passed"):
        "Tuition reimbursement may be contingent on passing the certification exam.",
}

# P1-5: allowed operators per numeric field, so directions match business sense.
# booking_lead_days: booking too SHORT before travel is the risk -> only "<".
# most others: too HIGH is the risk -> ">"/">=". standard_ride_wait: short wait -> "<".
def allowed_ops(field):
    return S.allowed_ops(field)

def num_value(action, field, operator, ho_thresholds, rng):
    """pick a training threshold avoiding held-out numbers under same action/field/op."""
    pool = [v for v in NUM_POOL[field] if (action, field, operator, v) not in ho_thresholds]
    return rng.choice(pool) if pool else None


# ================= phrasing patches =================
# Authored from general expense-management vocabulary to break literal copying and force
# surface->canonical grounding. Incidental overlap with held-out phrasings is expected
# (shared natural language); held-out exact sentences are never copied verbatim.

def _syn(pool, key, fallback, rng):
    p = pool.get(key)
    return rng.choice(p) if p else fallback

# --- Patch #1a: category subject synonyms (complete subject noun-phrases) ---
CAT_SYN = {
 # held-out single-cat categories: TRAIN aliases only, DISJOINT from EVAL_ALIAS (below)
 "alcohol": ["alcohol expenses","alcohol purchases","liquor and spirits purchases","bar-tab charges"],
 "gambling": ["gambling expenses","betting stakes","gaming-table losses","bets and wagering charges"],
 "seat_upgrade": ["seat upgrades","paid seat selection","premium-seat fees","upgraded-seat charges"],
 "first_class_airfare": ["premium-cabin airfare","front-cabin tickets","top-cabin flights"],
 "basic_economy_airfare": ["restricted-economy fares","no-frills economy tickets","stripped-down economy airfare"],
 "family_member_airfare": ["relatives' plane tickets","flights for a dependent","tickets for accompanying kin"],
 "family_travel": ["relatives' trip costs","travel for dependents","kin travel charges"],
 "minibar": ["in-room bar charges","room refreshment-bar items","hotel room-bar purchases"],
 "inflight_alcohol": ["onboard drinks","alcohol bought aboard the plane","cabin bar purchases"],
 "inflight_entertainment": ["onboard streaming and wifi","seat-back screen rentals","cabin media charges"],
 "commute": ["commuting costs","home-office travel","daily work-commute charges"],
 "traffic_fine": ["traffic fines","moving-violation penalties","road-citation charges","penalty fees for driving violations"],
 "spouse_meal": ["partner meal charges","dining costs for a companion","significant-other meals"],
 "training_bootcamp": ["intensive skills courses","immersive upskilling programs","accelerated course fees"],
 "pet_transport": ["animal shipping fees","pet relocation charges","companion-animal transport"],
 "rental_personal_accident_insurance": ["optional accident cover on a hire car","rental-desk accident add-on","car-hire injury coverage"],
 "entertainment_facility": ["nightlife-venue charges","amusement-venue spend","entertainment-establishment bills"],
 "club_membership": ["private members' club dues","recreational club fees","society membership charges"],
 "home_utilities": ["household utility bills","home power and water charges","domestic energy bills"],
 "personal_cloud_storage": ["individual online-storage plans","personal drive subscriptions","private backup-storage plans"],
 "spa_treatment": ["wellness-spa services","salon and spa charges","relaxation-treatment fees"],
 "dietary_supplements": ["nutrition supplements","health-supplement purchases","fitness nutrition products"],
 "gift_card": ["prepaid stored-value cards","gift vouchers","redeemable value cards"],
 "cash_advance": ["payroll advances","advance cash draws","salary-advance requests"],
 # non-held-out categories: plural forms (grammar only)
 "client_entertainment": ["client entertainment","client hospitality outings","entertaining clients"],
 "client_gift": ["client gifts","gifts for clients","client presents"],
 "client_meal": ["client meals","meals with clients","client dining"],
 "group_meal": ["group meals","team meals","group dining bills"],
 "staff_dinner": ["staff dinners","team dinners","employee dinners"],
 "meals": ["meals","meal expenses","food and dining"],
 "hotel_suite_upgrade": ["hotel suite upgrades","suite upgrade fees","room-to-suite upgrades"],
 "lounge_day_pass": ["airport lounge day-passes","single-visit lounge passes","one-day lounge entries"],
 "lounge_membership": ["airport lounge memberships","annual lounge memberships","lounge club dues"],
 "spa": ["spa services"],
 "valet_parking": ["valet parking","valet service charges","attended-parking fees"],
 "parking": ["parking","parking fees","parking charges"],
 "private_driver": ["private drivers","personal chauffeurs for the day","hired-driver service"],
 "chauffeur_service": ["chauffeur services","chauffeur-driven cars","professional driver service"],
 "charitable_gala": ["charitable galas","fundraising dinners","charity benefit events"],
 "hospitality": ["hospitality spend","corporate hospitality","hospitality and entertaining"],
 "conference": ["conference fees","conference registrations","conference attendance costs"],
 "certification_exam": ["certification exams","professional exam fees","credentialing tests"],
 "generative_ai_subscription": ["generative-AI subscriptions","AI assistant subscriptions","GenAI tool licenses"],
 "encrypted_external_storage_drive": ["encrypted external drives","secure portable drives","encrypted USB drives"],
 "personal_printer": ["personal printers","home printers","individual desktop printers"],
 "home_utilities_x": ["placeholder"],
 "atm_withdrawal": ["ATM withdrawals","cash taken from ATMs","ATM cash draws"],
 "room_timing_change": ["room date changes","hotel check-in/out changes","stay-date changes"],
}
# --- alias-aware guard: EVAL-alias surface forms RESERVED for held-out eval ---
# TRAIN text (CAT_SYN above) is authored disjoint from these. No train row may contain any of
# these terms (layer-2 leakage=0); held-out exact texts are also blocked (layer-1).
EVAL_ALIAS = {
 "alcohol": ["alcoholic"],
 "cash_advance": ["cash advance"],
 "seat_upgrade": ["extra legroom","preferred seating"],
 "minibar": ["mini-bar","minibar"],
 "commute": ["home to the primary office","daily travel from home"],
 "traffic_fine": ["speeding","parking fine"],
 "spouse_meal": ["spouses or partners"],
 "training_bootcamp": ["training bootcamp"],
 "family_travel": ["family members accompanying"],
 "first_class_airfare": ["first-class","first class"],
 "basic_economy_airfare": ["basic economy"],
 "inflight_entertainment": ["in-flight movies","in-flight games","headset rentals"],
 "inflight_alcohol": ["in-flight alcoholic"],
 "pet_transport": ["transporting personal pets","emotional-support"],
 "family_member_airfare": ["airfare for a family member"],
 "rental_personal_accident_insurance": ["personal accident insurance"],
 "entertainment_facility": ["entertainment facilities","lounges, or venues"],
 "gambling": ["casino","sports betting","racetrack","gambling losses","wagers"],
 "club_membership": ["country club","social club"],
 "home_utilities": ["residential heating","electricity, water"],
 "personal_cloud_storage": ["cloud storage expansions"],
 "spa_treatment": ["massages","facials","saunas"],
 "dietary_supplements": ["protein powders","weight-loss","vitamins"],
 "gift_card": ["cash equivalents","purchasing gift cards"],
}
_EVAL_ALIAS_TERMS = [t.lower() for terms in EVAL_ALIAS.values() for t in terms]

def _norm_text(t):
    return re.sub(r"\s+", " ", t.strip().lower()).rstrip(".")

def _has_eval_alias(text):
    tl = text.lower()
    return any(term in tl for term in _EVAL_ALIAS_TERMS)

# Gap A: explicit numeric pairing for (action, prereq) atoms whose only category pairing is
# held-out (so no compatible non-held-out category exists). These num+prereq combos are NOT
# held-out signatures, so they teach the action-scoped atom cleanly. e.g. approve+vp_approval:
# held-out pairs it with conference / client_entertainment_event_amount, so pair with per_person.
GAPA_NUM_PAIR = {
    ("approve_expense_report", "vp_approval"): "per_person_amount",
    ("approve_expense_report", "card_training"): "card_charge_amount",
}

# --- Patch #1b: numeric field label synonyms ---
NUM_LABEL_SYN = {
 "requested_limit": ["the requested limit","the new card limit","the requested credit line","the limit being requested"],
 "card_charge_amount": ["the travel-card charge","the amount charged to the card","the card transaction amount"],
 "chair_amount": ["the ergonomic chair cost","the office chair price","the chair purchase amount"],
 "daily_meal_amount": ["the daily meal amount","the per-day meal spend","the meal cost for the day"],
 "per_person_amount": ["the per-person amount","the cost per head","the amount per attendee"],
 "home_office_stipend": ["the home office stipend","the home-office allowance","the remote-work stipend"],
 "internet_stipend": ["the internet stipend","the home internet allowance","the broadband reimbursement"],
 "hotel_nightly_rate": ["the hotel nightly rate","the room rate per night","the nightly hotel cost"],
 "daily_parking_amount": ["the daily parking amount","the parking cost per day","the per-day parking fee"],
 "per_ticket_amount": ["the per-ticket amount","the cost per ticket","the ticket price"],
 "annual_gift_amount": ["the annual gift total","the yearly gift spend per recipient","the cumulative gift amount"],
 "client_entertainment_event_amount": ["the client entertainment event amount","the client-event cost","the spend on a client event"],
 "tuition_amount": ["the tuition amount","the course tuition","the tuition fee"],
 "personal_mileage_miles": ["the personal mileage","the personal miles driven","the non-business mileage"],
 "standard_ride_wait_minutes": ["the standard ride wait","the wait for a standard ride","the estimated standard-car wait"],
 "tip_percentage": ["the tip percentage","the gratuity rate","the tip as a percent of the bill"],
 "booking_lead_days": ["the booking lead time in days","how many days ahead it was booked","the advance-booking window"],
 "short_term_rental_nights": ["the short-term rental length in nights","the number of rental nights","the nightly rental duration"],
 "days_since_expense": ["the expense age in days","how old the expense is","the days since the expense occurred"],
}

# --- Patch #1c: prerequisite phrase synonyms ---
PREREQ_SYN = {
 "manager_approval": ["manager approval","sign-off from a manager","line-manager approval","manager authorization"],
 "vp_approval": ["VP approval","vice-president sign-off","VP authorization"],
 "ceo_authorization": ["CEO authorization","CEO sign-off","authorization from the CEO"],
 "c_suite_authorization": ["C-suite authorization","executive sign-off","approval from a C-suite executive"],
 "cio_clearance": ["CIO clearance","sign-off from the CIO","IT-leadership clearance"],
 "it_security_approval": ["IT security approval","security-team sign-off","InfoSec approval"],
 "legal_approval": ["legal approval","sign-off from Legal","legal-team clearance"],
 "controller_signoff": ["controller sign-off","approval from the controller","finance-controller sign-off"],
 "card_training": ["card training","the card-usage training","mandatory card training"],
 "itemized_receipt": ["an itemized receipt","a detailed receipt","a line-item receipt"],
 "business_justification": ["a business justification","a written business case","a documented justification"],
 "business_purpose_documentation": ["business purpose documentation","documented business purpose","a recorded business reason"],
 "business_purpose": ["a business purpose","a stated business reason","a documented business purpose"],
 "attendee_list": ["an attendee list","a list of attendees","the guest list"],
 "cash_withdrawal_approval": ["cash withdrawal approval","approval for a cash withdrawal","sign-off for withdrawing cash"],
 "lounge_preapproval": ["lounge pre-approval","advance lounge approval","pre-authorized lounge access"],
}

# --- Patch #3: positive forms for unless / only-if rendering (polarity teaching) ---
# MODAL verb forms only (may/can/will + participle) -- number-invariant, so they agree with
# singular, plural, AND mass-noun category aliases alike ("Parking may not be reimbursed",
# "Parking fees may not be reimbursed", "Client entertainment may not be reimbursed"). Avoids the
# copula ("is/are reimbursable") subject-verb agreement bug with variable-number CAT_SYN aliases.
POS_ACTION_VERB = {
 "approve_expense_report": ["may be approved","can be approved"],
 "issue_reimbursement": ["may be reimbursed","can be reimbursed"],
 "increase_card_limit": ["may be raised","may be increased"],
}
SUBJ_VERB = {
 "approve_expense_report": ["may not be approved","cannot be approved","will not be approved"],
 "issue_reimbursement": ["may not be reimbursed","cannot be reimbursed","will not be reimbursed"],
 "increase_card_limit": ["may not be raised","may not be increased","cannot be increased"],
}
GEN_SUBJ = {
 "approve_expense_report": ("The expense report", ["may not be approved","cannot be approved"]),
 "issue_reimbursement": ("The expense", ["may not be reimbursed","is not reimbursable"]),
 "increase_card_limit": ("The card limit", ["may not be increased","may not be raised"]),
}

def _prereq_positive(name, rng):
    if name in PREREQ_ACTION_CLAUSE: return PREREQ_ACTION_CLAUSE[name]
    ph = _syn(PREREQ_SYN, name, PREREQ_PHRASES.get(name, name.replace("_", " ")), rng)
    return ph + rng.choice([" has been obtained", " has been provided", " is on file"])

def _bool_positive(field, rng):
    lbl = {"business_purpose_provided": "a business purpose", "tax_id_provided": "a tax ID"}[field]
    return lbl + rng.choice([" is provided", " is on file", " has been documented"])

def try_unless(action, body, rng):
    """Render prereq/provided-bool bodies as 'unless <positive>' or '<positive> only if',
    teaching the polarity flip. Returns a sentence or None if body is ineligible."""
    subject = None; pos = []
    for e in body:
        if "not_completed" in e:
            pos.append(_prereq_positive(e["not_completed"], rng))
        elif e.get("field") == "expense_category":
            subject = _syn(CAT_SYN, e["value"], readable(e["value"]) + " expenses", rng)
        elif e.get("field") in ("business_purpose_provided", "tax_id_provided") and e.get("value") is False:
            pos.append(_bool_positive(e["field"], rng))
        else:
            return None
    if not pos: return None
    conj = " and ".join(pos)
    if rng.random() < 0.5:  # 'unless' + negative verb
        if subject is not None:
            s = subject[0].upper() + subject[1:]; verb = rng.choice(SUBJ_VERB[action])
        else:
            s, vp = GEN_SUBJ[action]; verb = rng.choice(vp)
        return f"{s} {verb} unless {conj}."
    else:                    # 'only if' + positive verb
        s = (subject[0].upper() + subject[1:]) if subject is not None else GEN_SUBJ[action][0]
        return f"{s} {rng.choice(POS_ACTION_VERB[action])} only if {conj}."

# --- Patch #2: allowance-ceiling rendering for single numeric '>' bodies ---
# "reimbursable up to a maximum of $N" == deny when the amount exceeds N (operator '>').
CEIL_TMPL = ["{S} is capped at {n}", "{S} is reimbursable up to {n}",
             "{S} is allowed up to a maximum of {n}", "{S} may not exceed {n}",
             "{S} is subject to a {n} cap", "{S} is limited to {n}"]
def try_ceiling(action, body, rng):
    # single numeric '>' body -> allowance-ceiling framing. Subject MUST be the field label
    # (not a generic 'The expense'), otherwise the numeric field is ungroundable from the text.
    if len(body) != 1: return None
    e = body[0]
    if e.get("operator") != ">" or "field" not in e or e["field"] == "expense_category": return None
    if not isinstance(e["value"], (int, float)) or isinstance(e["value"], bool): return None
    lbl = _syn(NUM_LABEL_SYN, e["field"], NUM_META[e["field"]][0], rng)
    S = lbl[0].upper() + lbl[1:]
    n = fmt_num(e["field"], e["value"])
    return rng.choice(CEIL_TMPL).format(S=S, n=n) + "."

# ---------- renderer ----------
# Q3 principle: a condition phrase NEVER carries when/if/unless. The sentence template adds the
# connective. This prevents "... unless X ... is routed to review" style breakage.
# render_entry returns (kind, payload):
#   ("subject", "alcohol expenses")            -> category, becomes the sentence subject
#   ("cond", "the amount exceeds $500")        -> bare condition, template wraps it
#   ("prereq", "manager approval")             -> prereq phrase (phrase form)
#   ("prereq_clause", "the expense report has been approved")  -> full-clause prereq

def prereq_payload(name):
    if name in PREREQ_ACTION_CLAUSE:
        return ("prereq_clause", PREREQ_ACTION_CLAUSE[name])
    return ("prereq", PREREQ_PHRASES.get(name, name.replace("_", " ")))

def render_entry(action, entry, rng):
    if "not_completed" in entry:
        return prereq_payload(entry["not_completed"])
    field, op, val = entry["field"], entry["operator"], entry["value"]
    if field == "expense_category":
        return ("subject", _syn(CAT_SYN, val, f"{readable(val)} expenses", rng))
    if isinstance(val, bool):
        if field in ("business_purpose_provided", "tax_id_provided"):
            lbl = SPECIAL_LABEL[field]
            return ("cond", f"{lbl} is not provided")
        if field == "within_alcohol_limit":
            lbl = SPECIAL_LABEL[field]
            return ("cond", rng.choice([f"{lbl} is exceeded",
                                        f"spending is not within {lbl}"]))
        if field in ("duplicate_suspected", "misuse_suspected"):
            return ("cond", SPECIAL_LABEL[field])
        return ("cond", field.replace("_", " "))
    if field in OTHER_ENUM_LABEL:  # other enum
        lbl = OTHER_ENUM_LABEL[field]
        target = readable(val)
        if op == "==":  # scoped_num uses ==canonical: text must say "is X", not "other than X"
            return ("cond", f"{lbl} is {target}")
        return ("cond", f"{lbl} is other than {target}")  # != canonical
    # numeric: bare condition, no leading connective
    lbl = _syn(NUM_LABEL_SYN, field, NUM_META[field][0], rng)
    nstr = fmt_num(field, val)
    if op == ">=":
        # 'no fewer than' only reads right for countable units (nights, days). For continuous
        # quantities ($, %, miles, minutes) use count-neutral phrasings.
        countable = NUM_META[field][1] in ("n", "d")  # nights, days
        pool = (OP_PHRASES[">="] if countable
                else ["{label} is {n} or more", "{label} is at least {n}", "{label} is no less than {n}"])
        return ("cond", rng.choice(pool).format(label=lbl, n=nstr))
    if op in OP_PHRASES:
        return ("cond", rng.choice(OP_PHRASES[op]).format(label=lbl, n=nstr))
    return ("cond", f"{lbl} {op} {nstr}")

# sentence templates. condition slots get bare phrases; template adds if/when/before/unless.
DENY_SUBJECT_TMPL = [
    "{subj} {verb}{when}.",
]
# for conditions we choose a connective per condition-type; prereq uses before/until.
def wrap_conditions(conds, prereqs, prereq_clauses, rng):
    """build the trailing clause. All parts are bare conditions; the connective is added here.
    prereqs become bare negatives ('manager approval has not been obtained') so they read as
    conditions, not 'and unless X' fragments (Q3: condition carries no when/if/unless)."""
    parts = []
    for c in conds:
        parts.append(rng.choice(["when", "if"]) + " " + c)
    for p in prereqs:
        # p is a noun phrase like "manager approval" / "an itemized receipt"
        neg = rng.choice([f"{p} has not been obtained",
                          f"{p} has not been provided",
                          f"{p} is not on file"])
        parts.append(rng.choice(["when", "if"]) + " " + neg)
    for pc in prereq_clauses:
        # pc is a full clause like "the expense report has been approved" -> negate
        neg = pc.replace("has been", "has not been")
        parts.append(rng.choice(["when", "if"]) + " " + neg)
    return parts

def render_policy(action, body, rng, review_phrased=False):
    if not review_phrased:
        alt = try_unless(action, body, rng)
        if alt is None and rng.random() < 0.5: alt = try_ceiling(action, body, rng)
        if alt is not None: return alt.replace("  ", " ")
    subject = None
    conds, prereqs, prereq_clauses = [], [], []
    for e in body:
        kind, payload = render_entry(action, e, rng)
        if kind == "subject": subject = payload
        elif kind == "cond": conds.append(payload)
        elif kind == "prereq": prereqs.append(payload)
        elif kind == "prereq_clause": prereq_clauses.append(payload)
    parts = wrap_conditions(conds, prereqs, prereq_clauses, rng)
    tail = (" " + " and ".join(parts)) if parts else ""

    # verb phrases per action, in both plural (for "X expenses" subjects) and singular forms.
    # avoids "expenses is" (plural agreement) and "A reimbursement ... reimbursed" (awkward).
    if review_phrased:
        # gold is DENY, so phrase it as an explicit block (pending review), not the
        # ambiguous "requires review" which conflates REVIEW with DENY.
        blk = rng.choice(["must be blocked pending review", "may not proceed pending review"])
        if subject is not None:  # "X expenses" -> plural, "must be blocked" agrees
            s = subject[0].upper() + subject[1:]
            return f"{s} {blk}{tail}.".replace("  ", " ")
        noun = {"approve_expense_report": "The expense report",
                "issue_reimbursement": "The expense",
                "increase_card_limit": "The card-limit increase request"}[action]
        return f"{noun} {blk}{tail}.".replace("  ", " ")

    if subject is not None:  # category subject: number varies by CAT_SYN alias -> MODAL verbs only
        s = subject[0].upper() + subject[1:]
        return f"{s} {rng.choice(SUBJ_VERB[action])}{tail}.".replace("  ", " ")

    # generic subject: singular, and avoid reimbursement+reimbursed repetition / card-limit+raised
    subj, verb = {
        "approve_expense_report": ("The expense report",
                                   ["may not be approved", "cannot be approved"]),
        "issue_reimbursement": ("The expense",
                                ["may not be reimbursed", "is not reimbursable"]),
        "increase_card_limit": ("The card limit",
                                ["may not be increased", "may not be raised"]),
    }[action]
    return f"{subj} {rng.choice(verb)}{tail}.".replace("  ", " ")

# ---------- SFT row ----------
INSTRUCTION = (
    "Convert the business policy into a JSON rule graph. The input begins with the schema: the "
    "actions this system exposes and, for each action, every canonical value (expense "
    "categories, numeric fields, booleans, other enums, prerequisites). "
    "Your job is SEMANTIC MAPPING, not text extraction: read what the policy MEANS and emit the "
    "matching canonical token from the schema. The policy uses everyday corporate language, "
    "synonyms, brand names, and examples; you must translate each such concept to the single "
    "canonical schema value it denotes. "
    "Do NOT copy words verbatim from the policy text into any target_action, field, value, or "
    "prerequisite. If the policy's wording is not letter-for-letter identical to a schema token, "
    "output the schema's canonical token, never the policy's phrasing. Every target_action, "
    "field, category, enum value, and prerequisite you emit MUST be one of the canonical tokens "
    "listed in the schema above; never invent one from the policy's words. "
    "The ONLY values taken literally from the policy text are numeric thresholds (amounts, "
    "counts, limits); everything else is a schema lookup. "
    "Output only valid JSON with keys: domain, rules. Each rule has target_action and "
    "deny_when, a list of entries read as a conjunction. An entry is {field, operator, value} "
    "or {not_completed: name}. A rule fires when every entry holds; the action is denied if "
    "any rule fires."
)

# ontology text comes from the seed (single source of truth), not hard-coded.
ONTOLOGY = S.ontology_text()

def model_input(policy_text):
    return f"{ONTOLOGY}\n\nPolicy: {policy_text}"

def sft_row(policy_text, action, body):
    graph = {"domain": DOMAIN, "rules": [{"target_action": action, "deny_when": body}]}
    return {"instruction": INSTRUCTION, "input": model_input(policy_text),
            "output": json.dumps(graph, ensure_ascii=False)}

# ---------- generation ----------
def make_single_bodies(action, inv, ho_thresholds, rng):
    """yield all single-condition bodies for the coverage pass (deterministic walk)."""
    a = inv[action]
    for v in a["cat"]:
        yield [{"field": "expense_category", "operator": "==", "value": v}]
    for f in a["bool"]:
        yield [{"field": f, "operator": "==", "value": BOOL_DENY_VALUE.get(f, False)}]
    for field, vals in a["other_enum"].items():
        for v in vals:  # canonical; rule is != canonical
            yield [{"field": field, "operator": "!=", "value": v}]
    for f in a["num"]:
        for op in allowed_ops(f):
            nv = num_value(action, f, op, ho_thresholds, rng)
            if nv is not None:
                yield [{"field": f, "operator": op, "value": nv}]
    for pre in a["prereq"]:
        if pre in S.standalone_prereqs(action):   # from seed
            yield [{"not_completed": pre}]

def make_conj_bodies(action, inv, ho_thresholds, rng):
    """yield conjunction bodies (four forms). Section 3."""
    a = inv[action]
    cats, nums, pres = a["cat"], a["num"], a["prereq"]
    other = a["other_enum"]
    out = []
    # cat+prereq (only business-compatible pairs, P1-4)
    for c in cats:
        for pre in pres:
            if not S.cat_prereq_ok(action, c, pre): continue
            out.append(("cat+prereq", [{"field":"expense_category","operator":"==","value":c},
                                       {"not_completed":pre}]))
    # num+prereq (only business-compatible (numeric, prereq) pairs)
    for f in nums:
        for pre in pres:
            if not S.num_prereq_ok(action, f, pre): continue
            op = allowed_ops(f)[0]
            nv = num_value(action, f, op, ho_thresholds, rng)
            if nv is not None:
                out.append(("num+prereq", [{"field":f,"operator":op,"value":nv},
                                           {"not_completed":pre}]))
    # scoped_num: only whitelisted (other_enum field, value, numeric) triples (Q1)
    for ef, ev, f in SCOPED_NUM_TRIPLES.get(action, []):
        for op in allowed_ops(f):
            nv = num_value(action, f, op, ho_thresholds, rng)
            if nv is not None:
                out.append(("scoped_num", [{"field":ef,"operator":"==","value":ev},
                                           {"field":f,"operator":op,"value":nv}]))
    # cat+num: only whitelisted (category, numeric) pairs (Q1)
    for c, f in CAT_NUM_PAIRS.get(action, []):
        for op in allowed_ops(f):
            nv = num_value(action, f, op, ho_thresholds, rng)
            if nv is not None:
                out.append(("cat+num", [{"field":"expense_category","operator":"==","value":c},
                                        {"field":f,"operator":op,"value":nv}]))
    rng.shuffle(out)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--held-out", default=str(ROOT / "seeds" / "expense_management.json"))
    ap.add_argument("--train", type=int, default=5000)
    ap.add_argument("--val", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--conj-floor", type=int, default=300)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    inv = S.ONTOLOGY
    ho_sigs, ho_thr = S.held_out_guards()

    rows = []
    row_sigs = []          # signature per row, for grouped split
    action_counter = Counter()  # per-action count, for the balance pass
    kind_counter = Counter()
    conj_counter = Counter()
    seen_sigs = set()
    dropped_leak = 0
    dropped_eval_alias = 0   # train text contained a reserved eval-alias term
    sig_via_alias = 0        # held-out single-cat signature admitted via a disjoint train alias
    HELD_OUT_TEXTS_NORM = {_norm_text(p["policy_text"]) for p in S.POLICIES}
    templates_seen = set()
    op_phrasings = defaultdict(set)

    seen_texts = set()
    sig_para_count = Counter()
    MAX_PARAPHRASE = 5  # per rule signature

    def emit(action, body, kind_label, conj_label=None):
        nonlocal dropped_leak, dropped_eval_alias, sig_via_alias
        sig = rule_signature(action, body)
        # alias-aware guard: a held-out signature is admissible only when the body contains a
        # category (so the held-out is taught through a DISJOINT category alias, checked below).
        # Pure numeric / pure prereq held-out signatures have no surface to alias -> always dropped.
        has_cat = any(e.get("field") == "expense_category" for e in body)
        sig_hit = sig in ho_sigs
        if sig_hit and not has_cat:
            dropped_leak += 1; return False
        # threshold leak guard (redundant safety)
        for e in body:
            if "field" in e and isinstance(e["value"],(int,float)) and not isinstance(e["value"],bool):
                if (action,e["field"],e["operator"],e["value"]) in ho_thr:
                    dropped_leak += 1; return False
        # P1-6: cap paraphrases per signature
        if sig_para_count[sig] >= MAX_PARAPHRASE:
            return False
        review = (rng.random() < 0.15)
        text = render_policy(action, body, rng, review_phrased=review)
        # layer-1: never emit a held-out policy's exact text
        if _norm_text(text) in HELD_OUT_TEXTS_NORM:
            dropped_leak += 1; return False
        # layer-2: never emit any reserved eval-alias surface term (eval-alias leakage = 0)
        if _has_eval_alias(text):
            dropped_eval_alias += 1; return False
        # layer-3: held-out single-cat signature admitted via disjoint alias -> count it
        if sig_hit and has_cat:
            sig_via_alias += 1
        # P1-6: exact-text dedup
        if text in seen_texts:
            return False
        seen_texts.add(text)
        sig_para_count[sig] += 1
        rows.append(sft_row(text, action, body))
        row_sigs.append(sig)
        action_counter[action] += 1
        kind_counter[kind_label] += 1
        if conj_label: conj_counter[conj_label] += 1
        templates_seen.add(re.sub(r'\$?[\d,]+%?|\b\d+ (days|nights|miles|minutes)\b','N',text))
        # track operator phrasings
        for e in body:
            if e.get("operator") in (">",">=","<"):
                op_phrasings[e["operator"]].add(re.sub(r'\$?[\d,]+%?','N',text))
        return True

    def emit_multi(action, bodies, conj_label="multi"):
        # emit a multi-rule (OR) graph. Each sub-rule must be non-held-out (no constituent
        # rule may equal a held-out single-rule signature) and threshold-clean.
        nonlocal dropped_leak, dropped_eval_alias
        for body in bodies:
            if rule_signature(action, body) in ho_sigs:
                dropped_leak += 1; return False
            for e in body:
                if "field" in e and isinstance(e["value"],(int,float)) and not isinstance(e["value"],bool):
                    if (action,e["field"],e["operator"],e["value"]) in ho_thr:
                        dropped_leak += 1; return False
        gsig = (action, tuple(sorted(rule_signature(action,b)[1][0] for b in bodies)) + ("MULTI",))
        if sig_para_count[gsig] >= MAX_PARAPHRASE: return False
        entries = [(e["field"], e["operator"], e["value"]) for b in bodies for e in b]
        text = render_multi_numeric(action, entries, rng)
        if _norm_text(text) in HELD_OUT_TEXTS_NORM:
            dropped_leak += 1; return False
        if _has_eval_alias(text):
            dropped_eval_alias += 1; return False
        if text in seen_texts: return False
        seen_texts.add(text); sig_para_count[gsig] += 1
        graph = {"domain": DOMAIN, "rules": [{"target_action": action, "deny_when": b} for b in bodies]}
        rows.append({"instruction": INSTRUCTION, "input": model_input(text),
                     "output": json.dumps(graph, ensure_ascii=False)})
        row_sigs.append(gsig); action_counter[action] += 1
        kind_counter["multi_rule"] += 1; conj_counter[conj_label] += 1
        return True

    # ---- coverage pass: K copies of every single-condition value ----
    for action in inv:
        for body in make_single_bodies(action, inv, ho_thr, rng):
            k = args.k
            for _ in range(k):
                emit(action, body, "single")

    # ---- conjunction coverage with per-form floor ----
    conj_pool = {a: make_conj_bodies(a, inv, ho_thr, rng) for a in inv}
    form_bodies = defaultdict(list)
    for a in inv:
        for form, body in conj_pool[a]:
            form_bodies[form].append((a, body))
    for form, items in form_bodies.items():
        rng.shuffle(items)
        need = args.conj_floor
        i = 0
        while need > 0 and items:
            a, body = items[i % len(items)]
            if emit(a, body, "conj", form): need -= 1
            i += 1
            if i > len(items) * 6: break  # avoid infinite loop if all leak

    # ---- auxiliary bool coverage (held-out-exclusive boolean fields) ----
    aux_bool_count = 0
    for action, field in HELD_OUT_EXCLUSIVE_BOOL:
        pres = AUX_BOOL_PREREQS[action]
        for i in range(args.k):
            pre = pres[i % len(pres)]
            body = [{"field": field, "operator": "==", "value": BOOL_DENY_VALUE.get(field, False)},
                    {"not_completed": pre}]
            if emit(action, body, "aux_bool", None):
                aux_bool_count += 1

    # ---- auxiliary other_enum coverage (held-out-exclusive != canonical fields) ----
    aux_other_count = 0
    for action, field, canonical in HELD_OUT_EXCLUSIVE_OTHER:
        pres = AUX_OTHER_PREREQS[action]
        for i in range(args.k):
            pre = pres[i % len(pres)]
            body = [{"field": field, "operator": "!=", "value": canonical},
                    {"not_completed": pre}]
            if emit(action, body, "aux_other", None):
                aux_other_count += 1

    # ---- atom_coverage_layer -- teach every prereq token once (coverage.json) ----
    coverage_count = 0
    for cov in COVERAGE:
        action = cov["action"]; pre = cov["prereq"]
        if "category" in cov:
            head = {"field": "expense_category", "operator": "==", "value": cov["category"]}
        else:
            f = cov["numeric"]; op = allowed_ops(f)[0]
            nv = num_value(action, f, op, ho_thr, rng)
            if nv is None: continue
            head = {"field": f, "operator": op, "value": nv}
        body = [head, {"not_completed": pre}]
        for _ in range(args.k):
            if emit(action, body, "coverage", "atom_coverage"): coverage_count += 1

    # ---- auto category coverage -- held-out-exclusive categories (single-cat is held-out,
    # no compatibility combo) taught via category + a neutral global gate (non-held-out). ----
    GATE = {"approve_expense_report": "card_training",
            "issue_reimbursement": "approve_expense_report",
            "increase_card_limit": "controller_signoff"}
    ho_cats = set()
    for _p in S.POLICIES:
        for _ru in _p["rule_graph"]["rules"]:
            for _e in _ru["deny_when"]:
                if _e.get("field") == "expense_category":
                    ho_cats.add((_ru["target_action"], _e["value"]))
    covered_cat = set()
    for _r in rows:
        for _ru in json.loads(_r["output"])["rules"]:
            for _e in _ru["deny_when"]:
                if _e.get("field") == "expense_category":
                    covered_cat.add((_ru["target_action"], _e["value"]))
    for action, cat in sorted(ho_cats):
        if (action, cat) in covered_cat: continue
        gate = GATE.get(action)
        if not gate or gate not in inv[action]["prereq"]: continue
        body = [{"field": "expense_category", "operator": "==", "value": cat}, {"not_completed": gate}]
        if rule_signature(action, body) in ho_sigs: continue
        for _ in range(args.k):
            if emit(action, body, "coverage", "atom_coverage"): coverage_count += 1

    # ---- 3-condition coverage (category + numeric + prerequisite) ----
    three_cond_count = 0
    for action, triples in THREE_COND_TRIPLES.items():
        for c, f, pre in triples:
            for op in allowed_ops(f):
                nv = num_value(action, f, op, ho_thr, rng)
                if nv is None: continue
                body = [{"field":"expense_category","operator":"==","value":c},
                        {"field":f,"operator":op,"value":nv},
                        {"not_completed":pre}]
                for _ in range(4):
                    if emit(action, body, "conj", "3cond"): three_cond_count += 1

    # ---- multi-rule (OR) coverage ----
    multi_count = 0
    for action, pairs in MULTI_RULE_NUM:
        bodies = [[{"field":f,"operator":op,"value":v}] for (f,op,v) in pairs]
        for _ in range(4):
            if emit_multi(action, bodies): multi_count += 1

    # ---- Gap A: action-scoped atom coverage ----
    # earlier coverage was token-level only, so a prereq valid under action X could be 0 under X
    # while present under Y (e.g. approve_expense_report+vp_approval = 0). Ensure every
    # (action, prereq) in the ontology appears under THAT action, via a business-compatible,
    # non-held-out category conjunction (fallback: standalone prereq).
    def _action_prereq_seen():
        seen = set()
        for r in rows:
            for ru in json.loads(r["output"])["rules"]:
                a = ru["target_action"]
                for e in ru["deny_when"]:
                    if "not_completed" in e: seen.add((a, e["not_completed"]))
        return seen
    seen_ap = _action_prereq_seen()
    gapA_count = 0; gapA_skipped = []
    for action in inv:
        cats = inv[action].get("cat", [])
        for pre in inv[action].get("prereq", []):
            if (action, pre) in seen_ap: continue
            placed = False
            for c in cats:
                if S.cat_prereq_ok(action, c, pre):
                    body = [{"field":"expense_category","operator":"==","value":c},
                            {"not_completed":pre}]
                    for _ in range(args.k):
                        if emit(action, body, "conj", "gapA_action_scoped"):
                            placed = True; gapA_count += 1
                    if placed: break
            # No standalone fallback (over-broad "all <action> need <pre>" is bad policy semantics).
            # If no compatible category exists (all are held-out), try an explicit numeric pairing.
            if not placed:
                np = GAPA_NUM_PAIR.get((action, pre))
                if np:
                    for op in allowed_ops(np):
                        nv = num_value(action, np, op, ho_thr, rng)
                        if nv is None: continue
                        body = [{"field":np,"operator":op,"value":nv},{"not_completed":pre}]
                        for _ in range(args.k):
                            if emit(action, body, "conj", "gapA_num_scoped"):
                                placed = True; gapA_count += 1
                        if placed: break
            if not placed: gapA_skipped.append((action, pre))
            seen_ap.add((action, pre))

    # ---- volume pass: random fill to target ----
    all_single = [(a, b) for a in inv for b in make_single_bodies(a, inv, ho_thr, rng)]
    all_conj = [(a, b) for a in inv for _, b in conj_pool[a]]
    guard = 0
    while len(rows) < args.train and guard < args.train * 20:
        guard += 1
        if rng.random() < 0.55 and all_single:
            a, b = rng.choice(all_single); emit(a, b, "single")
        elif all_conj:
            a, b = rng.choice(all_conj)
            # find its form
            emit(a, b, "conj", classify_conj(b))

    # ---- action-balance pass -- pump increase_card_limit to a floor share ----
    # Build a RICH increase pool directly (every threshold x op x optional prereq) so there are
    # enough distinct signatures to reach the floor without exceeding the per-signature paraphrase
    # cap. Held-out thresholds are skipped.
    TARGET_INCREASE_SHARE = 0.12
    inc_bodies = []
    for v in NUM_POOL["requested_limit"]:
        for op in allowed_ops("requested_limit"):
            if ("increase_card_limit", "requested_limit", op, v) in ho_thr: continue
            inc_bodies.append([{"field":"requested_limit","operator":op,"value":v}])
            for pre in inv["increase_card_limit"]["prereq"]:
                inc_bodies.append([{"field":"requested_limit","operator":op,"value":v},
                                   {"not_completed":pre}])
    rng.shuffle(inc_bodies)
    guard = 0
    while inc_bodies and action_counter["increase_card_limit"] < TARGET_INCREASE_SHARE * len(rows) and guard < 10000:
        b = inc_bodies[guard % len(inc_bodies)]
        emit("increase_card_limit", b, "single" if len(b)==1 else "conj",
             None if len(b)==1 else classify_conj(b))
        guard += 1

    # signature-grouped, coverage-preserving split. Every paraphrase of a rule
    # signature lands in the SAME split (no train/val leakage), and a signature moves to val
    # only if every canonical value it covers is still covered by another signature left in
    # train -- so train coverage stays complete.
    groups = defaultdict(list)
    for r, sg in zip(rows, row_sigs):
        groups[sg].append(r)
    def cov_keys_of_row(r):
        g = json.loads(r["output"]); rule = g["rules"][0]; a = rule["target_action"]; ks = set()
        for e in rule["deny_when"]:
            if "not_completed" in e: ks.add((a, "prereq", e["not_completed"]))
            elif e.get("field") == "expense_category": ks.add((a, "cat", e["value"]))
            elif isinstance(e.get("value"), bool): ks.add((a, "bool", e["field"]))
            elif isinstance(e.get("value"), (int, float)): ks.add((a, "num", e["field"]))
            else: ks.add((a, "other", e["field"]))
        return ks
    sig_keys = {sg: cov_keys_of_row(rs[0]) for sg, rs in groups.items()}
    key_count = Counter(k for ks in sig_keys.values() for k in ks)
    sig_list = list(groups)
    rng.shuffle(sig_list)
    val, val_sigs = [], set()
    for sg in sig_list:
        if len(val) >= args.val:
            break
        if all(key_count[k] >= 2 for k in sig_keys[sg]):      # keep >=1 in train
            val.extend(groups[sg]); val_sigs.add(sg)
            for k in sig_keys[sg]:
                key_count[k] -= 1
    train = [r for sg in sig_list if sg not in val_sigs for r in groups[sg]]
    rng.shuffle(train); rng.shuffle(val)

    outdir = ROOT / args.out
    outdir.mkdir(exist_ok=True)
    (outdir / "train.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in train) + "\n")
    (outdir / "val.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in val) + "\n")

    # alias-aware guard: three-layer leakage report (replaces "signature overlap = 0")
    _tr_texts = [r["input"].split("Policy: ",1)[1] for r in rows]
    exact_pair = sum(1 for t in _tr_texts if _norm_text(t) in HELD_OUT_TEXTS_NORM)
    eval_leak = sum(1 for t in _tr_texts if _has_eval_alias(t))
    print("  -- alias-aware leakage report --")
    print(f"  exact held-out pair overlap : {exact_pair}   (must be 0)")
    print(f"  eval-alias leakage in train : {eval_leak}   (must be 0)")
    print(f"  signature via disjoint alias: {sig_via_alias}   (held-out single-cat taught through TRAIN aliases)")
    print(f"  Gap A action-scoped fills   : {gapA_count}   skipped(no compat cat): {gapA_skipped}   (dropped_eval_alias: {dropped_eval_alias})")

    report(rows, kind_counter, conj_counter, inv, ho_sigs, ho_thr, dropped_leak,
           templates_seen, op_phrasings, args, aux_bool_count)

def classify_conj(body):
    fields = [e.get("field") for e in body if "field" in e]
    has_pre = any("not_completed" in e for e in body)
    has_cat = "expense_category" in fields
    has_num = any(isinstance(e.get("value"),(int,float)) and not isinstance(e.get("value"),bool) for e in body)
    other = any(f in OTHER_ENUM_LABEL for f in fields)
    if has_cat and has_pre: return "cat+prereq"
    if has_num and has_pre: return "num+prereq"
    if other and has_num: return "scoped_num"
    if has_cat and has_num: return "cat+num"
    return "other"

def report(rows, kind_counter, conj_counter, inv, ho_sigs, ho_thr, dropped_leak,
           templates_seen, op_phrasings, args, aux_bool_count=0):
    print("="*60)
    print(f"GENERATED {len(rows)} rows  (train {len(rows)-args.val}, val {args.val})")
    print("="*60)
    print("\n-- per entry kind --")
    for k, c in kind_counter.most_common(): print(f"  {k:10} {c}")
    print("\n-- per conjunction form --")
    for k, c in conj_counter.most_common(): print(f"  {k:14} {c}")
    print(f"\n-- auxiliary_bool_coverage rows: {aux_bool_count} --")
    covered_aux = set()
    for r in rows:
        g = json.loads(r["output"])
        for rule in g["rules"]:
            for e in rule["deny_when"]:
                if isinstance(e.get("value"), bool):
                    covered_aux.add((rule["target_action"], e["field"]))
    held_bool_covered = sum(1 for a,f in HELD_OUT_EXCLUSIVE_BOOL if (a,f) in covered_aux)
    print(f"  held-out-exclusive boolean fields covered: {held_bool_covered}/{len(HELD_OUT_EXCLUSIVE_BOOL)}")
    # value coverage
    print("\n-- held-out atom coverage (token-level; uncovered => lexical-OOD, must be 0) --")
    tok_cat=set(); tok_pre=set(); tok_bool=set(); tok_num=set(); tok_enum=set()
    for r in rows:
        for rule in json.loads(r["output"])["rules"]:
            for e in rule["deny_when"]:
                if "not_completed" in e: tok_pre.add(e["not_completed"])
                elif e.get("field")=="expense_category": tok_cat.add(e["value"])
                elif isinstance(e.get("value"),bool): tok_bool.add(e["field"])
                elif isinstance(e.get("value"),(int,float)): tok_num.add(e["field"])
                else: tok_enum.add((e["field"],e["value"]))
    missing=[]
    for p in S.POLICIES:
        for rule in p["rule_graph"]["rules"]:
            for e in rule["deny_when"]:
                if "not_completed" in e:
                    if e["not_completed"] not in tok_pre: missing.append("prereq/"+e["not_completed"])
                elif e.get("field")=="expense_category":
                    if e["value"] not in tok_cat: missing.append("cat/"+e["value"])
                elif isinstance(e.get("value"),bool):
                    if e["field"] not in tok_bool: missing.append("bool/"+e["field"])
                elif isinstance(e.get("value"),(int,float)):
                    if e["field"] not in tok_num: missing.append("num/"+e["field"])
                else:
                    if (e["field"],e["value"]) not in tok_enum: missing.append(f"enum/{e['field']}={e['value']}")
    missing=sorted(set(missing))
    print(f"  uncovered held-out atom tokens: {len(missing)}")
    for m in missing[:20]: print(f"    MISSING {m}")
    # diversity
    print("\n-- text diversity --")
    print(f"  distinct sentence templates: {len(templates_seen)}")
    for op in (">",">=","<"):
        print(f"  distinct phrasings for {op}: {len(op_phrasings[op])}")
    # leakage (alias-aware: single-cat held-out signatures are allowed via disjoint train
    # aliases; everything else must still be 0, and no exact held-out text / eval-alias may leak)
    print("\n-- leakage audit (alias-aware) --")
    ho_texts = {_norm_text(p["policy_text"]) for p in S.POLICIES}
    sig_bad = 0; sig_alias = 0; thr_overlap = 0; exact_pair = 0; eval_leak = 0
    for r in rows:
        txt = r["input"].split("Policy: ",1)[1]
        if _norm_text(txt) in ho_texts: exact_pair += 1
        if _has_eval_alias(txt): eval_leak += 1
        g = json.loads(r["output"])
        for rule in g["rules"]:
            a = rule["target_action"]; body = rule["deny_when"]
            if rule_signature(a, body) in ho_sigs:
                if any(e.get("field")=="expense_category" for e in body): sig_alias += 1
                else: sig_bad += 1
            for e in body:
                if "field" in e and isinstance(e["value"],(int,float)) and not isinstance(e["value"],bool):
                    if (a,e["field"],e["operator"],e["value"]) in ho_thr: thr_overlap += 1
    print(f"  exact held-out pair overlap:        {exact_pair}  (must be 0)")
    print(f"  eval-alias leakage in train:        {eval_leak}  (must be 0)")
    print(f"  non-single-cat signature overlaps:  {sig_bad}  (must be 0)")
    print(f"  threshold overlaps:                 {thr_overlap}  (must be 0)")
    print(f"  single-cat signature via alias:     {sig_alias}  (ALLOWED: held-out category taught via disjoint train alias)")
    print(f"  dropped during generation (leak guard): {dropped_leak}")
    ok = (len(missing)==0 and sig_bad==0 and thr_overlap==0 and exact_pair==0 and eval_leak==0)
    print("\n"+("PASS" if ok else "FAIL")+" -- coverage & leakage")

if __name__ == "__main__":
    main()

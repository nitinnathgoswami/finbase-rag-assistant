"""Generates eval/eval_set.json (kept as code so every gold label is reviewable).

Fields
  q            question (what a customer would type; many are paraphrases, not copies of FAQ text)
  type         lookup | paraphrase | calc | conflict | negative | unanswerable | followup
  answerable   False => KB has no answer; the assistant must say so
  gold         any ONE of these substrings must appear in a retrieved/cited chunk (retrieval + citation label)
  must_include list of groups; every group needs at least one of its alternatives in the answer (correctness)
  history      optional prior turns for follow-up questions
"""
import json
from pathlib import Path

Q = []
def add(q, type, gold, must, history=None, answerable=True):
    d = {"q": q, "type": type, "answerable": answerable, "gold": gold, "must_include": must}
    if history: d["history"] = history
    Q.append(d)

# ---- Savings -----------------------------------------------------------------
add("Is there a minimum balance I need to keep in the savings account?", "lookup",
    ["100% zero-balance structure", "zero-balance account"], [["no min", "zero", "not required", "no minimum"]])
add("How much interest will I earn in a year if I keep 5 lakh in my savings account?", "calc",
    ["Above ₹1,00,000 and up to ₹10,00,000", "the incremental ₹4,00,000 earns 6.00%"], [["3.5"], ["6"]])
add("How many free withdrawals do I get at non-partner ATMs if I live outside the metros?", "paraphrase",
    ["5 in non-metro locations"], [["5"]])
add("What will a physical debit card cost me?", "conflict",
    ["Physical Debit Card Annual Fee", "one-time issuance fee of ₹199"], [["199"]])
add("Will I be charged anything for closing my savings account?", "paraphrase",
    ["Account Closure Fee", "does not levy any account closure fee"], [["no", "free", "zero", "₹0"]])
add("What's my daily UPI limit and how many payments can I make in 24 hours?", "lookup",
    ["Daily cumulative limit of ₹1,00,000", "daily UPI limit is ₹1,00,000"], [["1,00,000"], ["20"]])
add("When does the bank add interest to my savings account?", "paraphrase",
    ["credited quarterly", "March 31, June 30"], [["quarter"], ["march", "31"]])

# ---- Fixed deposits & wealth -----------------------------------------------------
add("What rate does a senior citizen get on a 1 year FD?", "lookup",
    ["1 year to less than 2 years", "effective rate of 8.00% p.a."], [["8.00", "8%"]])
add("If I break my FD within the first week, how much interest do I get?", "paraphrase",
    ["zero interest", "within 7 days"], [["zero", "no interest", "0"]])
add("How can I stop TDS being cut from my FD interest?", "paraphrase",
    ["Form 15G"], [["15g", "15h"]])
add("Does FinBase let me buy and sell Bitcoin?", "negative",
    ["does NOT provide cryptocurrency", "does NOT offer cryptocurrency"], [["not", "no", "n't"]])
add("Can I take a loan against my farmland?", "negative",
    ["Agricultural Property Loans", "does NOT provide agricultural property"], [["not", "no", "n't"]])
add("Is money in my FinBase fixed deposit insured?", "lookup",
    ["DICGC insurance scheme up to"], [["5,00,000", "5 lakh"]])
add("What's the smallest SIP I can start?", "paraphrase",
    ["₹100 per month"], [["100"]])
add("I want to buy digital gold - what is the minimum purchase?", "lookup",
    ["starting from as low as ₹10", "as low as ₹10"], [["10"]])
add("Can you give me intraday stock tips for today?", "negative",
    ["Speculative Intraday Tips", "does NOT provide speculative intraday"], [["not", "no", "n't"]])

# ---- Payments / UPI ----------------------------------------------------------------
add("My UPI payment failed but the money left my account. When will I get it back?", "paraphrase",
    ["T+2 business days"], [["t+2", "2 business days", "two business days"]])
add("What do I get if the bank is late returning my failed transaction money?", "paraphrase",
    ["₹100 per calendar day", "₹100 per day"], [["100"]])
add("How much can I send by UPI in the 24 hours after resetting my PIN?", "lookup",
    ["₹5,000"], [["5,000"]])
add("How long do I have to report an unauthorised debit in the app?", "lookup",
    ["within 3 calendar days"], [["3"]])
add("A merchant debit failed on an online purchase. When should the refund arrive and when does compensation start?", "conflict",
    ["P2M Merchant Online Debit Failed"], [["t+2", "t + 2", "2 business"], ["t+5", "t + 5", "5"]])
add("What does the UPI error U69 mean?", "lookup",
    ["temporary network timeout", "U69 NPCI Timeout"], [["timeout"]])

# ---- Credit cards ----------------------------------------------------------------
add("How much is the Luxe card annual fee and how do I get it waived?", "lookup",
    ["₹1,20,000"], [["999"], ["1,20,000", "1.2"]])
add("Which of your credit cards has the cheapest foreign transaction markup?", "calc",
    ["Industry-leading 1.00% markup", "1.00% + GST"], [["metal"], ["1"]])
add("For how many days is the credit card interest free?", "lookup",
    ["45 to 50 days"], [["45"], ["50"]])
add("How much interest is charged if I pay only the minimum due?", "paraphrase",
    ["3.49% per month"], [["3.49"]])
add("What late fee applies if my card balance is ₹7,000?", "calc",
    ["₹5,001 - ₹10,000"], [["750"]])
add("Does the Neo card include airport lounge access?", "negative",
    ["No complimentary airport lounge access"], [["no", "not"]])

# ---- Personal loans ----------------------------------------------------------------
add("What is the foreclosure charge if I close my personal loan after 18 months?", "lookup",
    ["3% of the outstanding principal", "3.0% of outstanding principal"], [["3%", "3.0"]])
add("What CIBIL score do I need to get a personal loan?", "lookup",
    ["not less than 720"], [["720"]])
add("What is the most I can borrow as a personal loan?", "lookup",
    ["₹15,00,000"], [["15,00,000", "15 lakh"]])
add("On which day of the month is my EMI taken?", "paraphrase",
    ["5th calendar day"], [["5th", "fifth"]])
add("Can I pay back part of my loan in the first three months?", "paraphrase",
    ["minimum of 6 consecutive monthly EMI", "at least 6 consecutive EMIs"], [["not", "no", "6"]])
add("Who can I escalate to if support does not resolve my loan complaint?", "lookup",
    ["grievance.loans@finbase.com"], [["grievance"]])

# ---- KYC & security ----------------------------------------------------------------
add("What hours is video KYC available?", "lookup",
    ["Monday through Saturday from 9:00 AM to 8:00 PM"], [["monday"], ["9"]])
add("If I report a fraudulent transaction after 5 days, how much could I lose?", "lookup",
    ["4 to 7 calendar days"], [["5,000"]])
add("How many different ID documents does FinBase accept for KYC?", "lookup",
    ["5 Officially Valid Documents"], [["5", "five"]])
add("What number do I call to report card fraud?", "lookup",
    ["1800-FIN-BASE"], [["1800"]])
add("After how long without transactions does my account turn dormant?", "lookup",
    ["24 continuous months"], [["24"]])

# ---- Follow-ups (use chat history) ---------------------------------------------------
add("And what if I close it after 30 months?", "followup",
    ["1.5% of the outstanding principal", "reduced to 1.5%"], [["1.5"]],
    history=[{"role": "user", "content": "What is the foreclosure charge on my personal loan?"},
             {"role": "assistant", "content": "It is 3% of the outstanding principal plus GST if you close before 24 months."}])
add("What about the Metal one?", "followup",
    ["₹4,999 + 18% GST", "Waived upon crossing ₹5,00,000"], [["4,999"]],
    history=[{"role": "user", "content": "What is the annual fee on the Luxe credit card?"},
             {"role": "assistant", "content": "The Luxe card has a ₹999 + GST annual fee, waived if you spend over ₹1,20,000 a year."}])

# ---- Not in the knowledge base (must abstain) --------------------------------------------
for q in ["What is FinBase's home loan interest rate?",
          "What is the current RBI repo rate?",
          "Who is the CEO of FinBase?",
          "What are the NEFT cut-off timings?",
          "How do I redeem my credit card reward points?",
          "How long does a chargeback arbitration take to conclude?",
          "Can I take a joint personal loan with my spouse?"]:
    add(q, "unanswerable", [], [], answerable=False)

Path(__file__).with_name("eval_set.json").write_text(json.dumps(Q, ensure_ascii=False, indent=1))
print(len(Q), "questions;", sum(1 for x in Q if not x["answerable"]), "unanswerable")

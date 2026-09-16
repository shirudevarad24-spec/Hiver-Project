"""
Intent taxonomy for AmazonHelp customer messages.

Derived from TF-IDF + KMeans clustering (k=12) over ~9.3k customer messages
in data/processed/exchanges.jsonl, followed by manual reading of cluster
samples to merge/relabel into human-meaningful categories. See
report/decision_log.md for why these 9 and not Banking77's 77: Twitter
support complaints are coarser-grained than banking queries, and the
escalation decision cares more about "which of these 9 shapes is this"
than fine-grained sub-intent.

Each intent carries a short definition and a few representative example
utterances pulled from the data, used for the LLM few-shot classifier
prompt in src/classify.py.
"""

INTENTS = {
    "delivery_delay": {
        "description": "Order is still in transit / hasn't arrived by the expected date. No claim that tracking is wrong.",
        "examples": [
            "placed order under prime fr delivery 11am Apr 28 delivery not come web shows on the way",
            "your garbage service can't even deliver a package when I got one day shipping",
            "why is this my second prime purchase in a row that I'm getting later than my two day guaranteed delivery?",
        ],
    },
    "delivery_not_received_marked_delivered": {
        "description": "Tracking/website says delivered, but the customer says they never received the package. Higher-stakes than a plain delay - often needs investigation or a claim.",
        "examples": [
            "Product was not delivered to me but marked as delivered on the website. Amazon isn't refunding",
            "delivery not come web shows on the way but tracking says delivered",
            "I need an immediate response. I haven't received my order but it shows delivered.",
        ],
    },
    "damaged_or_wrong_item": {
        "description": "Item arrived damaged, tampered with, wrong, or incomplete.",
        "examples": [
            "Purchased through Amazon. The packages were opened tampered with, additional items inserted and then shipped to me.",
            "this is what amazon delivered me with false commitment regarding replacement.",
            "disappointed in bought the regrip and this is how I received it. As you can see this is useless.",
        ],
    },
    "refund_or_return": {
        "description": "Customer is asking for a refund, return, or order cancellation.",
        "examples": [
            "Why i can't get refund of my headphone which is worth 8999. they are is no return only replace.",
            "I have shared the details. Hope to return the mobile soon.",
            "Well not that I know near me. Why do you need that? I just wanted to order and now my account is locked!",
        ],
    },
    "billing_or_payment": {
        "description": "Incorrect/unexpected charge, payment method problems, or gift card / balance issues.",
        "examples": [
            "We rec'd fraudulent charges to our VISA from imdb that imdbpro never had, but Amazon did. Is this a hack",
            "do I need 100 bank accounts to shop on your site. I needed to shop urgently.",
            "what in the world is a balance withheld? #customerservice doesnt seem to know!",
        ],
    },
    "account_access_or_security": {
        "description": "Locked/hacked account, suspicious emails or calls claiming to be Amazon, login/session problems.",
        "examples": [
            "My grandfather received a call claiming to be from Amazon. Is this actually from Amazon or a scam?",
            "Are you guys leaking email ids of users who applied for the Amazon BTS offer? My bro got a spam email.",
            "I'm having problems signing up for Amazon Music. When I enter my details, it says 'session expired'.",
        ],
    },
    "product_or_service_inquiry": {
        "description": "A question that isn't tied to a specific broken order - product info, invoices/receipts, review verification, membership questions.",
        "examples": [
            "a friend of mine bought my book and wrote a review but it doesn't say VERIFIED PURCHASE. Will you please help me fix this?",
            "If I send someone a gift book. Did bill receipt goes with gifted book?",
            "Was just told that my Prime order will take a 3rd day to be delivered to the Locker.",
        ],
    },
    "service_complaint_generic": {
        "description": "General venting about service quality with no single actionable ask - no specific order/account referenced.",
        "examples": [
            "these guys never help on time even for prime customer.",
            "Done so many times. did again just now. Amazon is worst service provider on earth now.",
            "tell me what should i do and whom to contact. You guys are pathetic. only diplomatic replies. No help",
        ],
    },
    "positive_or_resolved": {
        "description": "Thanks / confirmation that an issue is resolved. Candidate for auto-close, no drafting needed beyond acknowledgment.",
        "examples": [
            "Thanks for information",
            "- Thanks Amazon 4 delivering product b4 tym. Amazing Support and Fastest Delivery caring customer's emotions.",
        ],
    },
}

INTENT_NAMES = list(INTENTS.keys())

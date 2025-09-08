# -*- coding: utf-8 -*-
"""Llama3.2.ipynb

Original file is located at
    https://colab.research.google.com/drive/1dOovgM4w3vTUGZmNRkrcbHlRUfxD_hiK
"""


import os, io, hashlib, pickle, faiss, numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
from PyPDF2 import PdfReader
import subprocess

# -----------------------------
# CONFIG
# -----------------------------
PDF_DIR = "docs"
PAGE_FILE = "page.file"
CHUNK_WORDS = 180
TOPK = 5
MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)
CURRENT_FAISS_VERSION = faiss.__version__
CONVERSATION_FILE = "conversation.pkl"

# Load conversation if file exists
if os.path.exists(CONVERSATION_FILE):
    with open(CONVERSATION_FILE, "rb") as f:
        conversation = pickle.load(f)
else:
    conversation = []
# -----------------------------
# HELPERS
# -----------------------------
def file_sig(path):
    p = Path(path)
    h = hashlib.md5()
    h.update(str(p.stat().st_mtime_ns).encode())
    h.update(str(p.stat().st_size).encode())
    return h.hexdigest()

def load_texts_with_meta(pdf_path):
    out = []
    p = Path(pdf_path)
    try:
        reader = PdfReader(str(p))
        for i, pg in enumerate(reader.pages, start=1):
            t = pg.extract_text() or ""
            if t.strip():
                out.append((t, {"doc": p.name, "page": i}))
    except Exception as e:
        print(f"warn: failed to read {p}: {e}")
    return out

def chunk_text(text, n=CHUNK_WORDS):
    w = text.split()
    return [" ".join(w[i:i+n]) for i in range(0, len(w), n)]

def build_chunks(pdf_dir):
    chunks, metas = [], []
    for pdf in Path(pdf_dir).glob("*.pdf"):
        for t, meta in load_texts_with_meta(pdf):
            cs = chunk_text(t)
            chunks.extend(cs)
            metas.extend([{**meta, "chunk": j+1} for j in range(len(cs))])
    return chunks, metas

def encode(arr):
    if not arr:
        return np.zeros((0, model.get_sentence_embedding_dimension()), dtype='float32')
    return np.asarray(model.encode(arr, convert_to_numpy=True), dtype="float32")

def build_index(chunks):
    X = encode(chunks)
    if X.shape[0] == 0:
        return None, X
    ix = faiss.IndexFlatL2(X.shape[1])
    ix.add(X)
    return ix, X

def save_pagefile(ix, X, chunks, metas, manifest, path=PAGE_FILE):
    with open(path, "wb") as f:
        pickle.dump({
            "ix": ix,
            "X": X,
            "chunks": chunks,
            "metas": metas,
            "manifest": manifest,
            "faiss_version": CURRENT_FAISS_VERSION
        }, f)

def load_pagefile(path=PAGE_FILE):
    with open(path, "rb") as f:
        return pickle.load(f)

# -----------------------------
# BUILD OR UPDATE PAGEFILE
# -----------------------------
def build_pagefile(pdf_dir=PDF_DIR, path=PAGE_FILE):
    chunks, metas = build_chunks(pdf_dir)
    ix, X = build_index(chunks)
    manifest = {str(p): file_sig(p) for p in Path(pdf_dir).glob("*.pdf")}
    save_pagefile(ix, X, chunks, metas, manifest, path)
    return ix, X, chunks, metas, manifest

def update_pagefile(pdf_dir=PDF_DIR, path=PAGE_FILE):
    if not os.path.exists(path):
        return build_pagefile(pdf_dir, path)

    pf = load_pagefile(path)
    # Rebuild if Faiss version changed
    if pf.get("faiss_version") != CURRENT_FAISS_VERSION:
        print("Faiss version mismatch: rebuilding index")
        return build_pagefile(pdf_dir, path)

    old_manifest = pf["manifest"]
    current = {str(p): file_sig(p) for p in Path(pdf_dir).glob("*.pdf")}

    deleted = set(old_manifest) - set(current)
    added_or_changed = [p for p,s in current.items() if old_manifest.get(p) != s]

    if deleted:
        print("Some PDFs deleted: rebuilding index")
        return build_pagefile(pdf_dir, path)

    if not added_or_changed:
        return pf["ix"], pf["X"], pf["chunks"], pf["metas"], current

    # append new/changed
    new_chunks, new_metas = [], []
    for p in added_or_changed:
        for t, meta in load_texts_with_meta(p):
            cs = chunk_text(t)
            new_chunks.extend(cs)
            new_metas.extend([{**meta, "chunk": j+1} for j in range(len(cs))])

    if new_chunks:
        X_new = encode(new_chunks)
        pf["ix"].add(X_new)
        pf["X"] = np.vstack([pf["X"], X_new])
        pf["chunks"].extend(new_chunks)
        pf["metas"].extend(new_metas)

    pf["manifest"] = current
    save_pagefile(pf["ix"], pf["X"], pf["chunks"], pf["metas"], pf["manifest"], path)
    return pf["ix"], pf["X"], pf["chunks"], pf["metas"], pf["manifest"]

def ensure_pagefile():
    ix, X, chunks, metas, manifest = update_pagefile(PDF_DIR, PAGE_FILE)
    if ix is None:
        raise ValueError("No PDFs found or index could not be built.")
    return ix, X, chunks, metas, manifest

# -----------------------------
# QUERY FUNCTIONS
# -----------------------------
# def query_rag_ollama(query, index, chunks, k=TOPK):
#     if index is None or len(chunks) == 0:
#         return "No chunks available for query.", []

#     # Retrieve relevant chunks
#     qvec = encode([query]).astype('float32')
#     D, I = index.search(qvec, k)
#     retrieved = [chunks[i] for i in I[0]]
#     context = "\n\n".join(retrieved)

#     prompt = f"Answer based on context:\n{context}\n\nQuestion: {query}\nAnswer:"
#     print("Prompt sent to Ollama:\n", prompt)

#     # Run Ollama
#     result = subprocess.run(
#         ["ollama", "run", "llama3.2:latest", prompt],
#         capture_output=True,
#         text=True
#     )
#     return result.stdout, list(zip(D[0].tolist(), I[0].tolist()))

# -----------------------------
# Maintain conversation history
# -----------------------------
def run_prompt1(query, ix, chunks, conversation=None):
    """
    First type of prompt: zero-shot query.
    """
    if conversation is None:
        conversation = []
    return query_rag_ollama_with_history(query, ix, chunks, conversation)

def run_prompt2(query, ix, chunks, conversation):
    """
    Second type of prompt: includes previous conversation.
    """
    return query_rag_ollama_with_history(query, ix, chunks, conversation)

def query_rag_ollama_with_history(query, index, chunks, conversation, k=TOPK, max_history=5):
    """
    Sends a query to Ollama, including previous questions and answers in the prompt.
    """
    if index is None or len(chunks) == 0:
        return "No chunks available for query.", []

    # Retrieve relevant chunks from Faiss
    qvec = encode([query]).astype('float32')
    D, I = index.search(qvec, k)
    retrieved = [chunks[i] for i in I[0]]
    context = "\n\n".join(retrieved)

    # Include conversation history
    history_text = ""
    for q, a in conversation[-max_history:]:
        history_text += f"Previous Q: {q}\nPrevious A: {a}\n\n"


    prompt = f"{history_text}Context:\n{context}\n\nCurrent Q: {query}\nAnswer:"
    print("Prompt sent to Ollama:\n", prompt)

    # Run Ollama
    result = subprocess.run(
        ["ollama", "run", "llama3.2:latest", prompt],
        capture_output=True,
        text=True
    )

    answer = result.stdout.strip()
    # Append current Q&A to conversation
    conversation.append((query, answer))
    with open(CONVERSATION_FILE, "wb") as f:
        pickle.dump(conversation, f)
    return answer, list(zip(D[0].tolist(), I[0].tolist()))

def show_page_table(chunks, metas, scores):
    if not scores:
        print("No results to show.")
        return
    print("# Semantic Page Table (top-k)")
    for r,(d, idx) in enumerate(scores, 1):
        m = metas[idx]
        snip = chunks[idx][:80].replace('\n',' ')
        print(f"{r:>2}. idx={idx:>6}  L2^2={d:.4f}  {m['doc']}#p{m['page']}  '{snip}'")

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    ix, X, chunks, metas, manifest = ensure_pagefile()

    # ans1, scores1 = query_rag_ollama("Give me 40 use cases of food delivery application?", ix, chunks)
    # show_page_table(chunks, metas, scores1)
    # print("\n---\n", ans1)
    # # Follow-up query
    # ans, scores2 = query_rag_ollama_with_history(
    #     "Which of these use cases are most profitable?", ix, chunks, conversation
    # )
    # query = "Generate use cases for a food delivery application. Make sure to produce atleast 30 detailed use cases including the preconditions, main flows, subflows and alternative flows"
    # ans, scores = query_rag_ollama_with_history(query, ix, chunks, conversation)
    # print(ans)
    # show_page_table(chunks, metas, scores)



    quert="""You are an expert requirements engineer.
    Generate 30 new use cases for a food delivery application.

    Each use case must follow this exact structure:

    Use Case Name
    Primary Actor:
    Supporting Actors:
    Preconditions:
    Postconditions:
    - Success:
    - Failure:
    Main Flow:
    (list the numbered steps clearly)
    Subflows:
    (list subflows as S1, S2, …)
    Alternative Flows:
    (list alternatives as A1, A2, …)

    Be thorough and consistent. Ensure the use cases are distinct from each other and cover a wide range of scenarios in the food delivery lifecycle (customer, restaurant, delivery agent, admin, payment system, etc.).
    """

    ans, scores = run_prompt1(quert, ix, chunks, conversation)
    print(ans)
    show_page_table(chunks, metas, scores)
    
    with open(CONVERSATION_FILE, "wb") as f:
        pickle.dump(conversation, f)

    if os.path.exists(CONVERSATION_FILE):
        with open(CONVERSATION_FILE, "rb") as f:
            conversation = pickle.load(f)
        else:
        conversation = []

    query2= """You are an expert requirements engineer.
    I will provide 10 sample use cases for a food delivery application.
    Study their structure and formatting carefully. Then generate 30 new and distinct use cases that follow the same style, tone, and structure.

    Here are the 10 sample use cases:

    1. User Registration & Login, Account & Address Management
    Primary Actor: Customer
    Supporting Actors: Authentication Service, Social Login Provider, Notification Service
    Preconditions: Customer has the app or website open with internet access.
    Postconditions:
    Success: Customer authenticated; profile created/updated; addresses saved.
    Failure: No account created; customer not logged in.
    Main Flow:
    Customer selects “Register / Sign In.”
    System prompts for credentials (email/phone/social login).
    Customer enters credentials.
    System sends OTP/email verification.
    Customer verifies → Auth Service validates → account created.
    Customer updates profile (name, contact info).
    Customer saves one or more delivery addresses.
    Subflows:
    S1: Social Login (Google/Apple).
    S2: Auto-complete address via GPS lookup.
    Alternative Flows:
    A1: Invalid OTP → system prompts retry.
    A2: Forgot Password → reset flow.
    A3: Duplicate Account → system offers account merge.


    2. Browse Restaurants & Menus
    Primary Actor: Customer
    Supporting Actors: Location Service, Catalog Service, Search/Filter Service
    Preconditions: Customer logged in; catalog available.
    Postconditions: Customer views restaurant and menu items; can add to cart.
    Main Flow:
    Customer sets or confirms delivery address.
    System retrieves list of available restaurants with ETA and fees.
    Customer browses menus, applies filters/sorts.
    Customer views item details and adds to cart.
    Subflows:
    S1: Dietary filters (vegan, gluten-free, etc.).
    S2: Price sorting.
    Alternative Flows:
    A1: No restaurants available → fallback message.
    A2: Item unavailable → suggest substitutions.


    3. Restaurant Recommendation (Past History & Location)
    Primary Actor: Customer
    Supporting Actors: Recommendation Engine, Analytics Service
    Preconditions: Customer logged in.
    Postconditions: Personalized restaurant suggestions displayed.
    Main Flow:
    System fetches customer’s past orders, ratings, and location.
    Recommendation Engine ranks restaurants by relevance.
    System highlights trending/popular ones if no history.
    Customer can filter, sort, or view special offers.
    Subflows:
    S1: Personalized campaign promotions.
    Alternative Flows:
    A1: No history and no local data → show top global trending restaurants.


    4. Place Order & Payment
    Primary Actor: Customer
    Supporting Actors: Payment Gateway, Promo Engine, Address Service
    Preconditions: Customer logged in; cart contains items; valid delivery address.
    Postconditions:
    Success: Order created; payment authorized; order ID assigned.
    Failure: No order created.
    Main Flow:
    Customer reviews cart, charges, and promos.
    Customer selects address and payment method.
    System applies promo and computes taxes/fees.
    Customer confirms → Payment Gateway authorizes.
    System generates order ID and sends confirmation.
    Subflows:
    S1: Save card for future.
    S2: Split bill between methods.
    Alternative Flows:
    A1: Payment failure → retry flow.
    A2: Invalid promo → remove and recalc.
    A3: Address invalid → prompt correction.


    5. Restaurant Order Handling
    Primary Actor: Restaurant Operator
    Supporting Actors: POS/KDS System, Order Management Service
    Preconditions: Restaurant open; new order received.
    Postconditions: Order accepted, updated to “ready for pickup” OR rejected.
    Main Flow:
    Restaurant receives order notification.
    Restaurant reviews items and instructions.
    Restaurant accepts order.
    Marks status as “preparing.”
    Marks “ready for pickup” once finished.
    Subflows:
    S1: POS sync for automated order ingestion.
    Alternative Flows:
    A1: Reject order (stock issues) → system cancels/refunds.
    A2: Modify order (remove unavailable items).
    A3: Update prep ETA.


    6. Driver Assignment & Pickup
    Primary Actor: Rider
    Supporting Actors: Dispatching Service, Navigation, Restaurant
    Preconditions: Order ready; driver pool available.
    Postconditions: Rider picks up order.
    Main Flow:
    Dispatching Service notifies nearby riders.
    Rider accepts.
    Rider navigates to restaurant.
    Rider verifies pickup (QR/photo).
    Status updated to “picked up.”
    Subflows:
    S1: Auto-assign driver if no manual acceptance.
    S2: Batch pickup for multiple orders.
    Alternative Flows:
    A1: No driver accepts → escalate/reassign.
    A2: Restaurant delay → ETA recalculated.
    A3: Wrong items → correction or cancel.


    7. Delivery and Tracking
    Primary Actors: Customer and Rider
    Supporting Actors: GPS Service, Map/Navigation Service, Notification Service
    Preconditions: Order picked up by rider (or en route);  GPS/location services enabled.
    Postconditions: Order delivered to customer OR returned.; Customer sees live tracking and ETA throughout delivery.
    Main Flow:
    Rider picks up the order from the restaurant.
    Rider navigates to delivery address using Navigation Service.
    System updates order status and live location for customer (preparing → picked up → arriving).
    Customer views live ETA and map; may share tracking link.
    Rider delivers order to customer or leaves at designated drop-off location.
    Rider captures delivery proof (photo/signature/code).
    System marks order as “delivered” and notifies customer.
    Subflows:
    S1: Low-precision tracking if GPS weak or unavailable.
    S2: Alternate drop-off instructions (neighbor, security desk, safe spot).
    Alternative Flows:
    A1: Rider offline → system provides status updates via text or notifications.
    A2: Delay detected → ETA recalculated and customer notified.
    A3: Customer unreachable → rider contacts support.
    A4: Address invalid → system prompts rider/customer for correction.
    A5: Order cannot be delivered or must be returned/disposed → system updates status and notifies customer.


    8. Ratings & Feedback
    Primary Actor: Customer
    Supporting Actors: Restaurant, Rider, Moderation Service
    Preconditions: Order delivered.
    Postconditions: Ratings stored, aggregated, visible to stakeholders.
    Main Flow:
    The system prompts customers after delivery.
    Customer provides star rating, text, or private feedback.
    System stores and updates averages.
    Subflows:
    S1: Restaurant responds to review.
    Alternative Flows:
    A1: Offensive content flagged.
    A2: Issue instead of review → escalate to support.


    9. Customer Support & Dispute Resolution
    Primary Actor: Customer / Support Agent
    Supporting Actors: Restaurant, Rider, Payment Provider
    Preconditions: Order completed or in-progress with reported issue.
    Postconditions: Ticket resolved/closed or escalated.
    Main Flow:
    Customer raises issue (chat/call/ticket).
    Support agent reviews logs and order details.
    Contacts restaurant/rider/payment if needed.
    Issues resolution (refund, redelivery, credit).
    Ticket closed; customer notified.
    Subflows:
    S1: Auto-triage resolves common cases like refund, replacement and order cancellation.
    S2: Escalation path to fraud/legal/security.
    Alternative Flows:
    A1: Customer disputes resolution → reopen case.
    A2: Payment dispute filed externally.


    10. Restaurant Menu & Availability Management
    Primary Actor: Restaurant Operator
    Supporting Actors: Catalog Service, Validation Service
    Preconditions: Restaurant onboarded.
    Postconditions: Menu updated and visible.
    Main Flow:
    Restaurant logs in to portal.
    Updates menu items, prices, tags.
    Adjusts hours/prep times.
    Publishes changes; system validates and updates catalog.
    Subflows:
    S1: Bulk upload.
    Alternative Flows:
    A1: Invalid entry → validation error.
    A2: Out-of-stock toggle.

    Now, based on these examples, generate 30 NEW use cases for the food delivery application.

    Each new use case must follow this exact structure:

    Use Case Name
    Primary Actor:
    Supporting Actors:
    Preconditions:
    Postconditions:
    - Success:
    - Failure:
    Main Flow:
    (list the numbered steps clearly)
    Subflows:
    (list subflows as S1, S2, …)
    Alternative Flows:
    (list alternatives as A1, A2, …)

    Do not repeat the original 10. Cover new scenarios involving customers, restaurants, delivery agents, payment services, admins, and external systems. Ensure the outputs are consistent, complete, and structured like the examples.
    """

    ans, scores = run_prompt2(query2, ix, chunks, conversation)
    print(ans)
    show_page_table(chunks, metas, scores)

import sqlite3
import faiss
import numpy as np
import pickle
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sklearn.metrics.pairwise import cosine_similarity
from langdetect import detect

# === Models ===
emmb_model = SentenceTransformer("all-MiniLM-L6-v2")  # For embeddings

# Phi-3 Mini for answer generation
phi_model_name = "microsoft/Phi-3-mini-128k-instruct"
phi_tokenizer = AutoTokenizer.from_pretrained(phi_model_name)
phi_model = AutoModelForCausalLM.from_pretrained(phi_model_name, device_map="auto")
qa_pipe = pipeline("text-generation", model=phi_model, tokenizer=phi_tokenizer)

# === System FAQ Knowledge Base ===
faq_data = {
    "How do I book a facility?": "To book a facility, search for one and click 'Book Now' on its detail page.",
    "How can I book a facility?": "To book a facility, search for one and click 'Book Now' on its detail page.",
    "How to make a reservation?": "To book a facility, search for one and click 'Book Now' on its detail page.",
    "How can I reserve a facility?": "To book a facility, search for one and click 'Book Now' on its detail page.",
    "I want to book a place": "To book a facility, search for one and click 'Book Now' on its detail page.",
    "Can I make a booking?": "Yes, you can search for a facility and click 'Book Now' to make a reservation.",
    "How do I make a booking?": "Search for a facility, then click 'Book Now' on its detail page.",
    "Can I book through the application?": "Yes, you can book facilities through the application.",
    "Can I book from the app?": "Yes, you can book facilities directly from the mobile app.",
    "Can I book via app?": "Yes, the mobile app allows direct booking.",
    "Can I use the app to reserve?": "Yes, you can reserve through the app.",
    "How can I book please?": "To book a facility, search for one and click 'Book Now' on its detail page.",
    "I would like to book": "To book a facility, search for one and click 'Book Now' on its detail page.",
    "Where to book a facility?": "Go to the facility page and click 'Book Now'.",
    "Can I make a reservation?": "Yes, click 'Book Now' on the facility page.",
    "Can I book now?": "Yes, go to the facility and click 'Book Now'.",
    "Booking process?": "Search for a facility and click 'Book Now' to reserve.",
    "Steps to reserve?": "Find your facility, then click 'Book Now' to make a reservation.",
    "How do I cancel my booking?": "To cancel, go to your profile, find the reservation, and click 'Cancel'.",
    "Can I cancel a booking?": "Yes, go to your profile and click 'Cancel' next to your reservation.",
    "How can I cancel?": "Go to your profile, find your reservation, and click 'Cancel'.",
    "Cancel reservation steps?": "Visit your profile, find the reservation, and cancel it.",
    "I want to cancel my booking": "To cancel a booking, go to your profile and click 'Cancel'.",
    "Can I cancel my reservation?": "Yes, go to your profile and click 'Cancel'.",
    "Cancel my reservation": "Visit your profile and use the 'Cancel' option for your reservation.",
    "How to undo a booking?": "Use the 'Cancel' option in your profile.",
    "I need to cancel my booking": "You can cancel it from your profile by clicking 'Cancel'.",
    "What payment methods are accepted?": "We accept credit cards, debit cards, and cash.",
    "How can I pay?": "You can pay using credit/debit cards or cash.",
    "Can I pay online?": "Yes, online payment is available via credit or debit card.",
    "Do you accept cash?": "Yes, we accept cash along with card payments.",
    "What are your payment options?": "You can pay by credit card, debit card, or cash.",
    "Can I get a refund?": "Refunds are allowed if you cancel at least 24 hours in advance.",
    "How do I get a refund?": "Cancel at least 24 hours before the reservation to receive a refund.",
    "Is refund possible?": "Yes, if canceled at least 24 hours before the reservation.",
    "Do I get my money back if I cancel?": "Yes, as long as you cancel at least 24 hours in advance.",
    "Refund policy?": "Refunds are given if cancellation happens at least 24 hours ahead.",
}
faq_questions = list(faq_data.keys())
faq_embeddings = emmb_model.encode(faq_questions)

# === Language detection and translation (if needed) ===
from transformers import pipeline as hf_pipeline
translator_to_en = hf_pipeline("translation", model="Helsinki-NLP/opus-mt-ar-en")
translator_to_ar = hf_pipeline("translation", model="Helsinki-NLP/opus-mt-en-ar")

def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"

# === FastAPI ===
app = FastAPI(debug=True)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === SQLite ===
conn = sqlite3.connect("facilities.db", check_same_thread=False)

class Question(BaseModel):
    question: str

def create_example_db():
    cur = conn.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS facilities (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT, sport TEXT, city TEXT, area TEXT, description TEXT)''')
    cur.execute("INSERT INTO facilities (name, sport, city, area, description) VALUES (?, ?, ?, ?, ?)",
                ("Cairo Football Club", "Football", "Cairo", "Nasr City", "A great place for 11-a-side football matches."))
    cur.execute("INSERT INTO facilities (name, sport, city, area, description) VALUES (?, ?, ?, ?, ?)",
                ("Zamalek Sports Center", "Tennis", "Cairo", "Zamalek", "Tennis courts available for booking"))
    conn.commit()

def get_data_from_db():
    cur = conn.cursor()
    cur.execute("SELECT * FROM facilities")
    rows = cur.fetchall()
    texts, metadata = [], []
    for row in rows:
        text = f"{row[1]} – {row[2]} – {row[3]}, {row[4]} – {row[5]}"
        texts.append(text)
        metadata.append({"id": row[0], "name": row[1], "sport": row[2], "city": row[3], "area": row[4]})
    return texts, metadata

def create_faiss_index():
    texts, metadata = get_data_from_db()
    embeddings = emmb_model.encode(texts, show_progress_bar=True)
    dim = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    faiss.write_index(index, "faiss_index.fac")
    with open("metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

def match_faq(user_question, threshold=0.5):
    user_embedding = emmb_model.encode([user_question])
    similarities = cosine_similarity(user_embedding, faq_embeddings)[0]
    best_idx = np.argmax(similarities)
    best_score = similarities[best_idx]
    if best_score >= threshold:
        matched_question = faq_questions[best_idx]
        return faq_data[matched_question]
    return None

def query_phi3(prompt):
    generated = qa_pipe(prompt, max_new_tokens=100, do_sample=False, temperature=0.3)[0]["generated_text"]
    if "### Answer:" in generated:
        return generated.split("### Answer:")[-1].strip()
    return generated.strip()

def query_facilities(user_query, top_k=3):
    query_embedding = emmb_model.encode([user_query], convert_to_numpy=True).astype("float32")
    index = faiss.read_index("faiss_index.fac")
    texts, metadata = get_data_from_db()

    # Filtering based on keywords (optional)
    filtered_texts, filtered_meta = [], []
    for i, meta in enumerate(metadata):
        if "football" in user_query.lower() and meta["sport"].lower() != "football":
            continue
        if "cairo" in user_query.lower() and meta["city"].lower() != "cairo":
            continue
        filtered_texts.append(texts[i])
        filtered_meta.append(meta)

    if filtered_texts:
        texts, metadata = filtered_texts, filtered_meta

    if not texts:
        return {"answer": "Sorry, no matching facilities found.", "references": []}

    embeddings = emmb_model.encode(texts, convert_to_numpy=True).astype("float32")
    local_index = faiss.IndexFlatL2(embeddings.shape[1])
    local_index.add(embeddings)

    D, I = local_index.search(query_embedding, top_k)

    results = []
    seen_texts = set()
    for idx in I[0]:
        if idx == -1 or idx >= len(texts):
            continue
        text = texts[idx]
        if text not in seen_texts:
            seen_texts.add(text)
            results.append((text, metadata[idx]["id"]))

    if not results:
        return {"answer": "Sorry, no matching facilities found.", "references": []}

    context = "\n".join([text for text, _ in results])
    prompt = f"""### Question: {user_query}\n\n### Facility Information:\n{context}\n\n### Answer:"""
    answer = query_phi3(prompt)
    references = [{"facility_id": fid, "text": text} for text, fid in results]
    return {"answer": answer, "references": references}

@app.post("/ask")
def ask(query: Question):
    try:
        user_q = query.question
        lang = detect_language(user_q)
        if lang != "en":
            user_q_en = translator_to_en(user_q, max_length=512)[0]['translation_text']
        else:
            user_q_en = user_q

        faq_answer = match_faq(user_q_en)
        if faq_answer:
            answer_en = faq_answer
        else:
            matches = query_facilities(user_q_en)
            if not matches:
                answer_en = "No facilities found."
            else:
                response = matches["answer"] + "\n\n"
                for ref in matches["references"]:
                    text = ref["text"]
                    facility_id = ref["facility_id"]
                    response += f"{text}\n[View Details](http://localhost:8000/facility/{facility_id})\n\n"
                answer_en = response.strip()

        if lang != "en":
            answer = translator_to_ar(answer_en, max_length=512)[0]['translation_text']
        else:
            answer = answer_en
        return {"answer": answer}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

@app.get("/api/facilities/{facility_id}")
def get_facility(facility_id: int):
    cur = conn.cursor()
    cur.execute("SELECT * FROM facilities WHERE id = ?", (facility_id,))
    row = cur.fetchone()
    if not row:
        return {"error": "Facility not found"}
    return {
        "id": row[0], "name": row[1], "sport": row[2],
        "city": row[3], "area": row[4], "description": row[5]
    }

# === INIT: create db + FAISS ===
create_example_db()
create_faiss_index()

# === Start App ===
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

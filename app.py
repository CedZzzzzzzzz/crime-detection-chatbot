from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv 
import google.generativeai as genai
from rag_engine import RAGEngine

load_dotenv()

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": ["https://crime-detection-system-steel.vercel.app"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Initialize Gemini AI
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    print("❌ Error: GOOGLE_API_KEY not found in .env file")
else: 
    genai.configure(api_key=GOOGLE_API_KEY)
    chat_model = genai.GenerativeModel('gemini-3-flash')
    print("✅ Gemini AI initialized successfully")

# Initialize RAG Engine
print("\n🔧 Initializing RAG Engine...")
rag_engine = RAGEngine(documents_folder="rules_documents")

@app.route('/')
def home():
    return jsonify({
        "status": "Detective AI Chatbot Service Active (RAG Enabled)",
        "endpoints": {
            "/chat": "POST - Investigation assistant with RAG"
        },
        "rag_status": "Active" if rag_engine.vectorstore else "No documents loaded"
    })

@app.route('/chat', methods=['POST', 'OPTIONS'])
def chat_with_ai():
    if request.method == 'OPTIONS':
        return '', 200
        
    """Detective AI investigation assistant with RAG"""
    try:
        data = request.get_json()
        user_question = data.get('question', '')
        detections = data.get('detections', [])
        
        log_entries = []
        highest_conf = 0
        primary_obj = "None"
        
        if detections:
            for d in detections:
                name = d.get('class_name', 'Unknown Object').upper()
                conf = d.get('confidence', 0)
                log_entries.append(f"- {name}: {conf}% confidence")
                
                # Track the most certain detection for the summary
                if conf > highest_conf:
                    highest_conf = conf
                    primary_obj = name
        
        evidence_str = "\n".join(log_entries) if log_entries else "No specific objects identified yet"
        
        # --- RAG SEARCH ---
        rag_context = ""
        source_citation = ""
        if rag_engine.vectorstore:
            results = rag_engine.search(user_question, k=2)  
            if results:
                sources = set()
                context_parts = []
                for result in results:
                    source = os.path.basename(result['source'])
                    sources.add(source.replace('.pdf', '').replace('-', ' ').title())
                    context_parts.append(result['content'])
                rag_context = "\n\n".join(context_parts)
                source_citation = ", ".join(sources)

        # Revised System Prompt for better Persona and Conversational Flow
        prompt = f"""You are Detective Co-AI-Nan, a sharp, observant, and professional Forensic AI Assistant. 
Your goal is to assist the user in their investigation by analyzing evidence and interpreting regulations.

### FORENSIC EVIDENCE LOG:
{evidence_str}
- PRIMARY THREAT: {primary_obj}
- ANALYSIS TIMESTAMP: 2026-01-19 13:36:08

### LEGAL REFERENCE DATA:
{rag_context if rag_context else "No direct legal matches found in current database."}
- SOURCE CITATIONS: {source_citation if source_citation else "N/A"}

### OPERATIONAL DIRECTIVES:
1. Persona: Maintain an analytical, professional, and observant detective persona. You can be slightly formal but conversational—like a seasoned investigator briefing a partner.
2. Conversation: If the user is just talking or asking non-investigative questions, respond naturally but stay in character.
3. Using Law/Regulations: If the user's question relates to the provided "Relevant Legal Documents," cite the source as: "According to [Source Name]..." Never mention page numbers.
4. General Analysis: If the user asks a question not covered by the legal documents, use your "Detective Expertise" (general knowledge) to provide a logical analysis.
5. Handling Uncertainty: If you lack evidence or documents to support a specific claim, state: "The current evidence is insufficient to reach a definitive forensic conclusion on this matter."
6. Brevity: Keep responses professional and to-the-point. Only elaborate if the user asks for a "Deep Dive" or "Case Analysis."

User's Question: "{user_question}" """

        response = chat_model.generate_content(prompt)
        
        return jsonify({
            "reply": response.text,
            "rag_used": bool(rag_context)
        })

    except Exception as e:
        print(f"❌ Gemini error: {e}")
        return jsonify({"reply": "⚠️ Investigation AI temporarily unavailable. Please try again."}), 500
    
if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 Starting Detective AI Chatbot Service")
    print("="*50)
    app.run(host='0.0.0.0', port=5001, debug=True)
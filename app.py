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
    chat_model = genai.GenerativeModel('gemini-2.5-flash')
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
    """Detective AI investigation assistant with RAG"""
    try:
        data = request.get_json()
        user_question = data.get('question', '')
        detections = data.get('detections', [])
        
        # Get detection context
        det_list = [d.get('class_name', 'object') for d in detections]
        context_str = ", ".join(det_list) if det_list else "no objects"
        
        # Get relevant rules/regulations from RAG
        rag_context = ""
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
            else:
                rag_context = None
                source_citation = None
        
        # Build enhanced prompt
        if rag_context:
            prompt = f"""Act as a professional detective and analytical assistant specialized in evidence-based scenarios.

Evidence Detected: {context_str}

Relevant Regulations:
{rag_context}
[Source: {source_citation}]

Question: "{user_question}"

Instructions:
- Provide direct, concise answers for simple questions
- For regulation-related queries, cite sources as "According to [source name], ..."
- Never mention page numbers
- Base answers on the regulations provided above
- Keep responses professional and to-the-point
- Only elaborate when the question requires detailed analysis
- Still answer some questions if they are not related to in the regulations provided and if they are nout provided, state that you don't have enough evidence to support your claims"""
        else:
            prompt = f"""Act as a professional detective analyzing this evidence.

**Evidence Detected:** {context_str}

**Question:** "{user_question}"

Provide a direct, professional answer based on your analytical expertise. Keep it concise unless the question requires detailed investigation."""

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
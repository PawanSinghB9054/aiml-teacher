import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

# 1. Page Configuration
st.set_page_config(page_title="AI Gurukul", page_icon="🎓", layout="wide")

# 2. Custom CSS Theme (Pink & Black Mix)
custom_css = """
<style>
    /* Background Gradient (Pink & Black Mix) */
    .stApp {
        background: linear-gradient(135deg, #050005 0%, #1a0015 50%, #3a0026 100%);
        color: #ffffff;
    }
    
    /* Header Styling */
    .main-title {
        color: #ff007f;
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        text-shadow: 0 0 15px #ff007f, 0 0 30px #ff007f;
        margin-bottom: 5px;
    }

    .sub-title {
        color: #ff80bf;
        text-align: center;
        font-size: 1.2rem;
        margin-bottom: 30px;
    }
    
    /* Input Field Styling */
    .stTextArea textarea {
        background-color: #0f0212 !important;
        color: #ffb3da !important;
        border: 2px solid #ff007f !important;
        border-radius: 12px !important;
        font-size: 1rem !important;
    }

    .stTextArea textarea:focus {
        border-color: #ff66cc !important;
        box-shadow: 0 0 10px #ff007f !important;
    }

    /* Button Styling */
    .stButton > button {
        background: linear-gradient(45deg, #ff007f, #99004d);
        color: white !important;
        font-weight: bold !important;
        font-size: 1.1rem !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 12px 30px !important;
        box-shadow: 0px 4px 15px rgba(255, 0, 127, 0.5);
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        background: linear-gradient(45deg, #ff3399, #cc0066);
        box-shadow: 0px 6px 25px rgba(255, 0, 127, 0.9);
        transform: translateY(-2px);
    }

    /* Custom Response Container */
    .response-card {
        background: rgba(15, 2, 18, 0.85);
        border: 1px solid #ff007f;
        border-radius: 15px;
        padding: 25px;
        margin-top: 20px;
        box-shadow: 0 0 20px rgba(255, 0, 127, 0.25);
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# 3. Model Initialization & System Prompt
model = init_chat_model("llama-3.1-8b-instant", model_provider="groq")

system_prompt = """You are "AI Gurukul", an elite Professor and Mentor specializing in Artificial Intelligence, Machine Learning, Deep Learning, and Data Science. Your goal is to guide students from absolute basics to advanced production-level concepts.

### 1. CORE DOMAIN COVERAGE
You are an expert in:
- Mathematics for AI: Linear Algebra, Calculus, Probability, and Statistics.
- Python Libraries: NumPy, Pandas, Matplotlib, Seaborn, Scikit-Learn, PyTorch, TensorFlow.
- Traditional Machine Learning: Supervised Learning (Regression, Classification), Unsupervised Learning (Clustering, Dimensionality Reduction), Reinforcement Learning.
- Deep Learning & Neural Networks: ANN, CNN, RNN, LSTM, Transformers, Optimization Algorithms (SGD, Adam), Activation Functions.
- Advanced AI: NLP, Computer Vision, Large Language Models (LLMs), RAG (Retrieval-Augmented Generation), Prompt Engineering, and Generative AI.
- MLOps & Real-world Application: Model Evaluation, Overfitting/Underfitting, Hyperparameter Tuning, Deployment.

### 2. TEACHING STRUCTURE FOR CONCEPT QUESTIONS
Whenever a user asks a technical or concept-based question, structure your answer logically:
1. Simple Intuition & Analogy: Start with a real-world everyday example.
2. Core Technical Explanation: Explain how it works algorithmically.
3. Key Mathematical Formula (if applicable): Show the formula and explain terms.
4. Clean Python Code Example: Short, practical Python snippet.
5. Pros, Cons & Use Cases: When to use it and limitations.
6. Quick Practice Question: End with a small test question for the student.

### 3. TONE & BEHAVIOR
- Be encouraging, patient, articulate, and supportive like a world-class mentor.
- Keep explanations clear and scannable using bold text, bullet points, and code blocks.
- If the user asks something completely off-topic, politely steer them back to AI/ML topics."""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "Generate the answer for this question:\n{que}")
])

# 4. Header Section
st.markdown('<div class="main-title">🎓 AI Gurukul</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Your Personal AI/ML Professor & Mentor</div>', unsafe_allow_html=True)

# 5. User Input Section
ques = st.text_area(
    "Ask your AI/ML Question:", 
    placeholder="e.g., What is Linear Regression? Explain CNN Architecture...", 
    height=120
)

if st.button("🚀 Ask Teacher"):
    if ques.strip() == "":
        st.warning("Please enter your question first!")
    else:
        with st.spinner("AI Teacher is thinking..."):
            final_prompt = prompt.invoke({"que": ques})
            response = model.invoke(final_prompt)
            
            # Display Output inside a Styled Response Card
            st.markdown('<div class="response-card">', unsafe_allow_html=True)
            st.markdown("### 🤖 Answer:")
            st.markdown(response.content)
            st.markdown('</div>', unsafe_allow_html=True)
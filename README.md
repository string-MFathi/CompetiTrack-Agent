# 🕵️‍♂️ CompetiTrack-Agent: Autonomous Competitive Intelligence

Welcome to **CompetiTrack-Agent**, an AI-powered autonomous agent built to conduct deep-dive competitive research, analyze market positioning, and extract actionable insights for any company, SaaS, or platform across any industry.

Developed as a foundational project in my journey as an **AI Engineer**, this agent demonstrates the practical implementation of agentic workflows, LLM orchestration, state management, and safety guardrails.

## 🚀 Project Overview
Traditional market research is manual and time-consuming. CompetiTrack-Agent automates this process by dynamically generating search queries, surfing the web, reading multiple sources, and synthesizing a structured, data-driven report. 

Provide the agent with a competitor's name and URL, and it will autonomously extract 4 core pillars:
1. **Pricing Models & Subscription Tiers**
2. **Core Competitive Advantages & Key Features**
3. **Target Audience & Product Weaknesses (via Customer Reviews)**
4. **Market Gaps & Opportunities**

## ✨ Key Features & Engineering Highlights
- **🤖 Agentic Reasoning:** Uses **Google Gemini 3.8 Flash** to read instructions, decide when to search, and when to synthesize data without human intervention.
- **🌐 Autonomous Web Search:** Integrates the **Tavily Search API** to fetch real-time, accurate data from the internet.
- **🧠 Stateful Memory:** Built on **LangGraph** to maintain a continuous memory of gathered facts across multiple search iterations.
- **🛡️ Infinite-Loop Guardrails:** Implements strict recursion limits (max 5 iterations) to prevent API quota exhaustion if public data is scarce.
- **🔄 Graceful Fallback Strategy:** Automatically detects when information is unavailable (e.g., for early-stage startups), halts the search gracefully, and drafts the best possible partial report based on memory without hallucinating facts.
- **📊 Observability & Tracing:** Fully integrated with **LangSmith** for tracing the agent's thought process, debugging, and monitoring step-by-step executions.

## 🛠️ Tech Stack
- **Framework:** [LangGraph](https://python.langchain.com/v0.1/docs/langgraph/) & LangChain
- **LLM:** Google GenAI (gemini-3.8-flash)
- **Tools:** Tavily Search API
- **Language:** Python 3.11+
- **Tracing:** LangSmith

## ⚙️ How to Run Locally

### 1. Clone the repository
```bash
git clone [https://github.com/YOUR_USERNAME/CompetiTrack-Agent.git](https://github.com/YOUR_USERNAME/CompetiTrack-Agent.git)
cd CompetiTrack-Agent

2. Set up a virtual environment
Bash
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
3. Install dependencies
Bash
pip install -r requirements.txt
4. Environment Variables
Create a .env file in the root directory and add your API keys:

Code snippet
GOOGLE_API_KEY="your_gemini_api_key_here"
TAVILY_API_KEY="your_tavily_api_key_here"
LANGCHAIN_TRACING_V2="true"
LANGCHAIN_API_KEY="your_langsmith_api_key_here"
LANGCHAIN_PROJECT="CompetiTrack-Agent"
5. Execute the Agent
Bash
python competitrack.py
The agent will prompt you for the competitor's details, print its reasoning process step-by-step, and generate the final structured report.

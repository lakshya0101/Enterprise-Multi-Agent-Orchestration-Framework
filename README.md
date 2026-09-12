# Enterprise Multi-Agent Orchestration Framework

A modular, production-ready framework designed to orchestrate specialized AI agents with strict state isolation, runtime verification, structured tool execution, and enterprise observability.

---

## 📌 Project Overview

The **Enterprise Multi-Agent Orchestration Framework** addresses the challenge of coordinating multiple specialized LLM agents in enterprise environments. By decoupling task planning, domain-specific execution, tool validation, and state reconciliation into deterministic graph workflows, the framework ensures reliability, explainability, and safety across automated business processes.

### High-Level Vision
* **Decoupled Architecture**: Separation of concerns between supervisory routing, specialized worker agents, and validator/critic mechanisms.
* **Deterministic State Flow**: Centralized state schemas with immutable audit logs and checkpointing.
* **Resilient Multi-Provider LLM Layer**: Provider-agnostic model routing with automatic fallback (Google Gemini ➔ Groq) and strict Pydantic structured output validation.
* **Extensible Tool Registry**: Sandboxed, schema-validated tool execution interfaces with timeout and permission enforcement.
* **Enterprise Observability**: Native support for LangSmith / LangChain tracing, telemetry, and structured execution logs.

---

## 🏛️ Conceptual Architecture

```
                              ┌─────────────────────────┐
                              │     API / Client UI     │
                              └────────────┬────────────┘
                                           │
                                           ▼
                              ┌─────────────────────────┐
                              │  Supervisor / Router    │
                              │  (Planning & Dispatch)  │
                              └────────────┬────────────┘
                                           │
             ┌─────────────────────────────┼─────────────────────────────┐
             ▼                             ▼                             ▼
   ┌───────────────────┐         ┌───────────────────┐         ┌───────────────────┐
   │ Research / Search │         │  Data Processing  │         │ Custom Tool Agent │
   │       Agent       │         │       Agent       │         │      (Domain)     │
   └─────────┬─────────┘         └─────────┬─────────┘         └─────────┬─────────┘
             │                             │                             │
             └─────────────────────────────┼─────────────────────────────┘
                                           │
                                           ▼
                              ┌─────────────────────────┐
                              │   Validator / Critic    │
                              │   (Output Verification) │
                              └────────────┬────────────┘
                                           │
                                           ▼
                              ┌─────────────────────────┐
                              │ State Checkpoint & Store│
                              │ (Memory / Database / RAG│
                              └─────────────────────────┘
```

---

## 🛠️ Technology Direction

* **Core Language**: Python 3.10+
* **Data Validation & Schemas**: Pydantic v2
* **Model Providers**: Google Gemini (`google-genai`), Groq (`groq`)
* **Graph Orchestration**: LangGraph / LangChain *(planned)*
* **Storage & Retrieval**: Vector stores (FAISS / Chroma / Qdrant) & SQL databases *(planned)*
* **API Layer**: FastAPI / Uvicorn *(planned)*
* **Testing & Quality**: Pytest, Ruff

---

## 🚦 Project Status

> **Current Phase**: `Phase 7: Observability, Auditability & Evaluation (Complete)`
>
> Phase 1 contracts, Phase 2 multi-provider LLM routing (Gemini + Groq), Phase 3 specialized agents (`PlannerAgent`, `RetrievalAgent`, `ToolExecutionAgent`, `ValidatorAgent`), Phase 4 centralized LangGraph supervisor runtime, Phase 5 RAG & persistence (local embeddings, FAISS, SQLite), Phase 6 FastAPI service layer (REST API endpoints, Server-Sent Events streaming, and Human-in-the-Loop review API), and Phase 7 enterprise observability, audit logging, real-time metrics, evaluation engine, and deterministic benchmark suite are fully implemented and verified with 118 offline deterministic tests.

---

## 🗺️ Implementation Roadmap

- [x] **Phase 1 — Contracts & Foundation** ✅
- [x] **Phase 2 — LLM Providers & Router** ✅
- [x] **Phase 3 — Specialized Worker & Critic Agents** ✅
- [x] **Phase 4 — LangGraph Orchestration Runtime** ✅
- [x] **Phase 5 — RAG & Persistent Retrieval** ✅
- [x] **Phase 6 — API / Service Layer** ✅
- [x] **Phase 7 — Observability, Auditability & Evaluation** ✅

---

## 💻 Local Setup & Development

### 1. Prerequisites
* Python 3.10 or higher
* Git
* Free Google AI Studio API Key ([Get Gemini API Key](https://aistudio.google.com/))
* Free Groq Console API Key ([Get Groq API Key](https://console.groq.com/keys))

### 2. Clone and Setup Environment

```bash
# Clone the repository
git clone https://github.com/lakshya0101/Enterprise-Multi-Agent-Orchestration-Framework.git
cd Enterprise-Multi-Agent-Orchestration-Framework

# Create a virtual environment
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux/macOS:
source .venv/bin/activate

# Install the package in editable mode with development dependencies
pip install -e ".[dev]"
```

### 3. Environment Configuration

Copy the sample environment file and configure your API keys:

```bash
cp .env.example .env
```

Edit `.env` to supply the necessary credentials:
```env
# Free LLM Provider Keys
GEMINI_API_KEY=your_gemini_api_key_here
GROQ_API_KEY=your_groq_api_key_here

# Provider Routing Options
LLM_PROVIDER=gemini
GEMINI_MODEL=gemini-2.5-flash
GROQ_MODEL=llama-3.3-70b-versatile
LLM_TIMEOUT_SECONDS=60.0
LLM_MAX_RETRIES=2
LLM_ENABLE_FALLBACK=true

# Tracing & Storage
LANGCHAIN_TRACING_V2=false
LANGCHAIN_API_KEY=
LANGCHAIN_PROJECT=enterprise-multi-agent-framework
VECTOR_STORE=faiss
DATABASE_URL=sqlite:///./data/app.db
```

> ⚠️ **Security Warning**: Never commit or share your `.env` file. The repository `.gitignore` automatically prevents accidental staging of `.env`.

### 4. Running Tests

Run the complete test suite (zero external API calls required):

```bash
python -m pytest tests/ -v
```

### 5. Optional Live Provider Smoke Test

To verify real network connectivity against your free Gemini/Groq accounts:

```bash
python scripts/smoke_test_providers.py
```

---

## 🗺️ Planned Roadmap

- [x] **Phase 0: Foundation & Security Baseline**
- [x] **Phase 1: Core Architecture & Contracts**
- [x] **Phase 2: LLM Provider Layer & Router (Gemini + Groq Fallback)**
- [x] **Phase 3: Specialized Worker & Critic Agents (Planner, Retriever, ToolExecutor, Validator)**
- [x] **Phase 4: Centralized LangGraph Supervisor Orchestrator**
- [x] **Phase 5: Persistent Checkpointing & Vector Store RAG**
- [x] **Phase 6: FastAPI Service Layer, Streaming, & HITL UI**
- [x] **Phase 7: End-to-End Evaluation & Enterprise Observability**

---

## 📄 License

This project is licensed under the terms defined in the repository license.
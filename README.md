# AI Bootcamp – Building Intelligent AI Agents

A hands-on AI engineering project exploring **LLM-powered agents, tool integration, conversational memory, multi-agent architectures, and structured data validation**, with a practical application in HVAC VRF/VRV engineering.

## 🚀 Project Overview

This repository documents the development of AI agents using Python, Groq LLMs, and LangChain. It progresses from building a basic AI agent to developing a validated multi-agent system capable of understanding HVAC-related queries, extracting engineering parameters, identifying missing information, and generating structured responses.

The primary objective is to explore practical AI agent development and apply these concepts to HVAC engineering workflows.

## 🎯 Key Features

- **LLM Integration:** Connect language models using Groq and LangChain.
- **AI Agent Development:** Build agents capable of understanding queries and executing tools.
- **Tool Calling:** Integrate Python functions for task-specific calculations.
- **Conversational Memory:** Maintain context across multiple interactions.
- **Persistent Memory:** Retain relevant information across sessions.
- **Multi-Agent Architecture:** Coordinate specialized agents for HVAC expertise, engineering analysis, and technical review.
- **Structured Outputs:** Generate structured JSON responses.
- **Schema Validation:** Validate HVAC engineering parameters using Pydantic.
- **Missing Data Detection:** Identify incomplete engineering inputs before performing calculations.

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| LLM | Groq API – GPT-OSS-120B |
| Frameworks | LangChain, LangGraph |
| Data Validation | Pydantic |
| Environment Management | Python-dotenv |
| Search Integration | Tavily |
| Development Tools | VS Code, Git, GitHub |

## 📂 Project Structure

```text
AI-Bootcamp-1/
│
├── 01_test_groq.py
├── 02_first_agent.py
├── 03_agent_multiple_tools.py
├── 04_agent_conversation_memory.py
├── 05_agent_presistent_memory.py
├── 06_multi_agent_system.py
├── 07_structured_multi_agent_system.py
├── 08_hvac_schema_validation.py
├── 09_validated_hvac_multi_agent.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

## 📚 Learning Modules

| Module | Description |
|---|---|
| 01 | Groq API connectivity and LLM setup |
| 02 | Building the first AI agent |
| 03 | Integrating multiple tools into an agent |
| 04 | Implementing conversational memory |
| 05 | Implementing persistent memory |
| 06 | Developing a multi-agent system |
| 07 | Generating structured multi-agent outputs |
| 08 | Defining and validating HVAC data schemas |
| 09 | Building a validated HVAC multi-agent system |

## 🏗️ HVAC VRF/VRV AI Agent

The final implementation explores an AI-powered HVAC learning assistant designed to support VRF/VRV engineering questions.

### Agent Responsibilities

1. **HVAC Expert:** Explains VRF/VRV concepts and engineering principles.
2. **Engineering Analyst:** Identifies required parameters, assumptions, and calculation inputs.
3. **Technical Reviewer:** Reviews outputs, flags unsupported assumptions, and identifies missing information.

### Engineering Parameters

The system uses structured HVAC room data, including:

- Room dimensions and floor area
- Ceiling height
- Occupancy
- Lighting and equipment loads
- Indoor and outdoor design temperatures
- Ventilation rate
- Wall and roof U-values
- Window area and SHGC
- Infiltration and solar heat gains
- Humidity control parameters

Pydantic validation helps ensure that engineering inputs follow the expected data structure and acceptable value constraints.

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/pr9je/AI-Bootcamp-1.git

cd AI-Bootcamp-1
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment:

**Windows PowerShell**

```powershell
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS**

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root directory:

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-120b
```

Replace the placeholder with your actual Groq API key.

Never commit your `.env` file or expose API credentials publicly.

### 5. Run the Projects

Execute individual modules in sequence:

```bash
python 01_test_groq.py
python 02_first_agent.py
python 03_agent_multiple_tools.py
python 04_agent_conversation_memory.py
python 05_agent_presistent_memory.py
python 06_multi_agent_system.py
python 07_structured_multi_agent_system.py
python 08_hvac_schema_validation.py
python 09_validated_hvac_multi_agent.py
```

## 💡 Learning Outcomes

Through this project, I explored:

- Building LLM-powered applications using Python.
- Designing AI agents with tool-calling capabilities.
- Managing conversational and persistent memory.
- Implementing multi-agent collaboration.
- Generating structured outputs using LLMs.
- Applying Pydantic for data validation.
- Developing domain-specific AI applications for HVAC engineering.
- Identifying missing engineering parameters and handling uncertainty in AI-generated responses.

## ⚠️ Disclaimer

This project is intended for educational and experimental purposes. HVAC cooling-load estimates and AI-generated engineering responses should not be treated as certified engineering designs.

Actual VRF/VRV equipment selection requires complete engineering inputs, manufacturer-specific performance data, applicable standards, and verification by qualified HVAC professionals.

## 👨‍💻 Author

**Mitesh Prajapati**

MSc Artificial Intelligence & Machine Learning

GitHub: [@pr9je](https://github.com/pr9je)

---

⭐ If you find this project useful, feel free to explore the repository and share your feedback.

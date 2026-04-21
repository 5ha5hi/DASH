# 📐 AI Analytics Platform

A high-performance, data-first analytics dashboard built with **Streamlit** and powered by **OpenRouter AI**. This platform provides deterministic data profiling, KPI computation, and AI-driven business insights without hallucinating numbers.

## 🚀 Quick Setup

### 1. Prerequisites
- **Python 3.9+** installed on your system.
- An **OpenRouter API Key** (for AI narration features).

### 2. Install Dependencies
Clone the repository and install the required Python packages:

```bash
# Recommended: Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Create a file named `.env` in the root directory (one already exists, but ensure it has your key):

```env
OPENROUTER_API_KEY=your_api_key_here
OPENROUTER_MODEL=openai/gpt-4o-mini
```

### 4. Run the Application
Launch the Streamlit dashboard:

```bash
streamlit run app.py
```

## 🛠 Features
- **Deterministic Data Profiling**: Statistical analysis of columns (metrics, dimensions, time series).
- **Business KPIs**: Automatic detection of Revenue, Profit, Growth, and Efficiency.
- **Smart Chart Selection**: Automatic visualization based on data distribution.
- **AI Business Narrative**: LLM-powered explanation of computed facts.
- **Safe Query Engine**: Natural language-like querying without `exec()` risks.
- **Database Support**: Connect directly to MySQL or upload CSVs.

## 📁 Project Structure
- `app.py`: Main Streamlit UI.
- `ai_layer.py`: AI explanation logic (OpenAI/OpenRouter).
- `analytics_engine.py`: Core deterministic computation logic.
- `data_ingestion.py`: CSV and MySQL loading.
- `chart_selector.py`: Logic for recommending charts.
- `safe_query.py`: Constrained query executor.

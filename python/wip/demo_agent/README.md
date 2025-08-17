# Getting started
1. Set up a `.env` file with your OpenAI, Galileo and Tavily API keys. See `.env.example` as reference.
2. Set up the project using `uv`:
   ```bash
    pip install --upgrade uv  # Install uv
    uv venv  # set up virtual environment
    source .venv/bin/activate  # activate environment
    uv sync --dev  # install dependencies
    ```
3. Start the StreamLit app using `uv run streamlit run app.py`

# Getting started
1. Set up a `.env` file with your OpenAI, Galileo and Tavily API keys.
2. The app will log to prod by default and set the console URL appropriately to log to other instances. See `.env.example` as reference.
3. Set up the project using `uv`:
   ```bash
    pip install --upgrade uv  # Install uv
    uv venv  # set up virtual environment
    source .venv/bin/activate  # activate environment
    uv sync --dev  # install dependencies
    ```
4. Start the StreamLit app using `streamlit run app.py`
5. Try out the app with the following prompts:
    - Check the compliance status for supplier SUP001
    - What's the supply chain risk in Southeast Asia right now?
    - What is the latest news in Southeast Asia?
    - What is the weather like in Singapore?

# Getting started
1. Set up a `.env` based on `.env.example`.
   1. The OpenAI api key is available in 1Password
   2. Sign up for Tavily and Pinecone to get their API keys (they are free).
2. The app will log to prod by default and set the console URL appropriately to log to other environments.
3. Set up the project using `uv`:
   ```bash
    pip install --upgrade uv  # Install uv
    uv venv  # set up virtual environment
    source .venv/bin/activate  # activate environment
    uv sync --dev  # install dependencies
    ```
4. Set up the Pinecone database using `python ./scripts/setup_pinecone.py`
5. Start the StreamLit app using `streamlit run app.py`

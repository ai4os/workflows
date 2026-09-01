#  AI4OS Workflows

This repo include workflows that run as GitHub Actions and that do not belong to any particular repo.
This includes:

* running a check to make sure that important workflows from other organization repos are not disabled due to repo inactivity
* running manual model discovery on LiteLLM configured sub-providers (local LiteLLMs)

This repo might be eventually migrated when the projects moves to a centralized workflow manager.

#### Usage

Install requirements:

```bash
pip install -r requirements.txt
```

Add environment variables:

* for keep alive workflow:

    ```bash
    GH_PAT_ACTIONS="*********************"
    ```

* for model discovery workflow:

    ```bash
    LITELLM_API_KEY="sk-*********************"
    LITELLM_BASE_URL="https://vllm.cloud.ai4eosc.eu"

    IFCA01_API_KEY="sk-*********************"  # API key of sub-providers (local LiteLLMs)
    ```

#  AI4OS Workflows

This repo include workflows that run as GitHub Actions and that do not belong to any particular ai4os repo.
If you want to run a workflow relating to a particular ai4os repo, please consider including the workflow directly in that repo.

Included workflows:

* Meta workflows (ie. workflows to monitor workflows):

    - check that important workflows from other ai4os repos are not disabled due to repo inactivity
    - monitor health of important workflows and notify workflow maintainers

* LiteLLM workflows:

    - running manual model discovery on LiteLLM configured sub-providers (local LiteLLMs)
    - automate model capability assessment for new models
    - automate access group cleaning of old assets
    - check aliases still work fine

All these workflows are stateless, not needing persistent storage.
Stateful workflows will need to be ran on the future ai4os workflow manager.

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

from modelscope.hub.api import HubApi
api = HubApi()

candidates = [
    ("XiaomiMiMo/MiMo-7B-RL", "MiMo-7B-RL"),
    ("XiaomiMiMo/MiMo-7B-Reasoning", "MiMo-7B-Reasoning"),
    ("XiaomiMiMo/MiMo-7B-RL-v2.5", "MiMo-7B-RL-v2.5"),
    ("HuggingFaceTB/SmolVLM2-256M-Instruct", "SmolVLM2-256M"),
    ("HuggingFaceTB/SmolVLM2-500M-Instruct", "SmolVLM2-500M"),
    ("HuggingFaceTB/SmolVLM2-2.2B-Instruct", "SmolVLM2-2.2B"),
    ("google/gemma-3n-E2B-it", "Gemma-3n-E2B"),
    ("google/gemma-3n-E4B-it", "Gemma-3n-E4B"),
    ("deepseek-ai/DeepSeek-V4", "DeepSeek-V4"),
    ("deepseek-ai/DeepSeek-V4-0324", "DeepSeek-V4-0324"),
    ("MiniMaxAI/MiniMax-M3", "MiniMax-M3"),
    ("moonshotai/Kimi-K2-Instruct", "Kimi-K2"),
    ("cohere/command-a", "Command-A"),
    ("CohereForAI/c4ai-command-a-03-2025", "Command-A-03-2025"),
]

for mid, name in candidates:
    try:
        info = api.get_model(mid)
        print(f"OK   {name:30s}  {mid}")
    except Exception as e:
        err = str(e)[:60]
        print(f"MISS {name:30s}  {err}")

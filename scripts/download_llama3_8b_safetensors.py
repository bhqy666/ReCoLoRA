from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id='meta-llama/Meta-Llama-3.1-8B-Instruct',
    resume_download=True,
    local_files_only=False,
    max_workers=1,
    allow_patterns=[
        'config.json',
        'generation_config.json',
        'model.safetensors.index.json',
        'model-*.safetensors',
        'tokenizer.json',
        'tokenizer.model',
        'tokenizer_config.json',
        'special_tokens_map.json',
        'README.md',
        'LICENSE',
        'USE_POLICY.md',
    ],
)
print(path)

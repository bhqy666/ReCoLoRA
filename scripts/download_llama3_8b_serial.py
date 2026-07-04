from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id='meta-llama/Meta-Llama-3.1-8B-Instruct',
    resume_download=True,
    local_files_only=False,
    max_workers=1,
)
print(path)

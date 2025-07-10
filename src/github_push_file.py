## Push file to new github branch

import os
import subprocess
import pandas as pd

def push_parquet_to_github(df, filename, commit_msg, github_user, github_email, github_token, target_subdir=".", branch="main"):
    """
    Save DataFrame as Parquet and push to GitHub repo from Kaggle notebook.
    If the branch doesn't exist, it will be created locally and pushed.

    Args:
        df (pd.DataFrame): DataFrame to save
        filename (str): Name of Parquet file to save (e.g., 'data/output.parquet')
        commit_msg (str): Commit message to use
        github_user (str): GitHub username
        github_email (str): GitHub email
        github_token (str): Personal Access Token (PAT) with repo access
        target_subdir (str): Subfolder inside repo to store the file
        branch (str): Branch to push to (default = 'main')
    """

    # Repo info
    repo_name = "crypto_market_prediction"
    repo_url = f"https://{github_user}:{github_token}@github.com/CapstoneTeam23UMICH/{repo_name}.git"

    # Save DataFrame to parquet in /kaggle/working
    parquet_path = f"/kaggle/working/{os.path.basename(filename)}"
    df.to_parquet(parquet_path, index=False)
    print(f"Saved {parquet_path}")

    # Clone the repo
    repo_local_path = f"/kaggle/working/{repo_name}"
    if os.path.exists(repo_local_path):
        subprocess.run(["rm", "-rf", repo_local_path])
    subprocess.run(["git", "clone", repo_url], cwd="/kaggle/working")

    # Change to cloned repo
    subprocess.run(["git", "checkout", "-b", branch], cwd=repo_local_path)

    # Move the parquet file into the target folder
    target_path = os.path.join(repo_local_path, target_subdir)
    os.makedirs(target_path, exist_ok=True)
    subprocess.run(["mv", parquet_path, os.path.join(target_path, os.path.basename(filename))])

    # Git config and push
    subprocess.run(["git", "config", "--global", "user.email", github_email])
    subprocess.run(["git", "config", "--global", "user.name", github_user])
    subprocess.run(["git", "add", "."], cwd=repo_local_path)
    subprocess.run(["git", "commit", "-m", commit_msg], cwd=repo_local_path)

    # Push the new branch and set upstream
    subprocess.run(["git", "push", "--set-upstream", "origin", branch], cwd=repo_local_path)

    print(f"Pushed `{filename}` to `{repo_name}/{branch}`")
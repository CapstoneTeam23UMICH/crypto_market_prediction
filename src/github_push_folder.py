## Push folder to new github branch

import subprocess

def push_folder_to_github(
    repo_path,
    folder_to_commit,
    commit_msg,
    new_branch,
    github_token,
    github_user="nikolozjaghiashvili",
    github_email="nikoloz.jaghiashvili@gmail.com",
    repo_name="crypto_market_prediction",
    base_branch="main"
):
    """
    Create a branch, add a folder, commit, and push to GitHub with a token.
    """
    remote_url = f"https://{github_user}:{github_token}@github.com/CapstoneTeam23UMICH/{repo_name}.git"
    subprocess.run(["git", "-C", repo_path, "remote", "set-url", "origin", remote_url], check=True)
    subprocess.run(["git", "-C", repo_path, "config", "user.email", github_email], check=True)
    subprocess.run(["git", "-C", repo_path, "config", "user.name", github_user], check=True)
    subprocess.run(["git", "-C", repo_path, "checkout", base_branch], check=True)
    subprocess.run(["git", "-C", repo_path, "pull", "origin", base_branch], check=True)
    subprocess.run(["git", "-C", repo_path, "checkout", "-b", new_branch], check=True)
    subprocess.run(["git", "-C", repo_path, "add", folder_to_commit], check=True)
    subprocess.run(["git", "-C", repo_path, "commit", "-m", commit_msg], check=False)
    subprocess.run(["git", "-C", repo_path, "push", "--set-upstream", "origin", new_branch], check=True)

    print(f"✅ Pushed {folder_to_commit} from {repo_name} to branch {new_branch} (base: {base_branch}).")

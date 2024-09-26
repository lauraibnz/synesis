import os

import requests


def download_github_dir(owner, repo, path, save_dir):
    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    response = requests.get(api_url)
    if response.status_code == 200:
        contents = response.json()

        # Create the save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        for item in contents:
            if item["type"] == "file":
                download_url = item["download_url"]
                file_name = item["name"]
                file_path = os.path.join(save_dir, file_name)
                file_content = requests.get(download_url).content
                with open(file_path, "wb") as file:
                    file.write(file_content)
                print(f"Downloaded: {file_path}")
    else:
        print(
            f"Failed to fetch directory contents. Status code: {response.status_code}"
        )


def download_github_file(owner, repo, file_path, save_dir):
    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{file_path}"
    response = requests.get(api_url)

    if response.status_code == 200:
        file_info = response.json()
        if file_info["type"] == "file":
            download_url = file_info["download_url"]
            file_name = os.path.basename(file_path)
            save_path = os.path.join(save_dir, file_name)

            os.makedirs(save_dir, exist_ok=True)

            file_content = requests.get(download_url).content
            with open(save_path, "wb") as file:
                file.write(file_content)
            print(f"Downloaded: {save_path}")
        else:
            print(f"The specified path is not a file: {file_path}")
    else:
        print(f"Failed to fetch file. Status code: {response.status_code}")

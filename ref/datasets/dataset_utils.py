import os
from typing import Dict, Type

import requests
from torch.utils.data import Dataset

from ref.datasets.magnatagatune import MagnaTagATune
from ref.datasets.mtgjamendo import MTGJamendo


class DatasetFactory:
    _datasets: Dict[str, Type[Dataset]] = {
        "magnatagatune": MagnaTagATune,
        "mtgjamendo": MTGJamendo,
    }

    @classmethod
    def get_dataset(cls, name: str, **kwargs) -> Dataset:
        """
        Retrieve a dataset instance based on its name and optional parameters.

        Args:
            name: The name of the dataset to retrieve.
            **kwargs: Optional parameters to pass to the dataset constructor.

        Returns:
            An instance of the requested dataset.

        Raises:
            ValueError: If the dataset name is not recognized.
        """
        dataset_class = cls._datasets.get(name.lower())
        if dataset_class is None:
            raise ValueError(f"Unknown dataset: {name}")

        # Get the default parameter values from the dataset class
        default_params = {
            k: v.default
            for k, v in dataset_class.__init__.__annotations__.items()
            if k != 'return' and hasattr(v, 'default')
        }

        # Update default parameters with provided kwargs
        params = {**default_params, **kwargs}

        return dataset_class(**params)


def get_dataset(name: str, **kwargs) -> Dataset:
    """
    A convenience function to get a dataset instance.

    This function is a wrapper around DatasetFactory.get_dataset for
    easier importing and usage.

    Args:
        name: The name of the dataset to retrieve.
        **kwargs: Optional parameters to pass to the dataset constructor.

    Returns:
        An instance of the requested dataset.
    """
    return DatasetFactory.get_dataset(name, **kwargs)


def download_github_dir(owner, repo, path, save_dir):
    """
    Download a directory from a GitHub repository.

    Args:
        owner (str): The owner of the repository.
        repo (str): The name of the repository.
        path (str): The path to the directory in the repository.
        save_dir (str): The directory to save the downloaded files.
    """
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
    """
    Download a file from a GitHub repository.

    Args:
        owner (str): The owner of the repository.
        repo (str): The name of the repository.
        file_path (str): The path to the file in the repository.
        save_dir (str): The directory to save the downloaded file.
    """
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

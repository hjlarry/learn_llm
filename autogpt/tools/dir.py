import os


def list_files_in_directory(path: str) -> str:
    """List all file names in the directory"""
    file_names = os.listdir(path)
    return "\n".join(file_names)

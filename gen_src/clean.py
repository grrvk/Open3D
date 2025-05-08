import os
import pathlib
from pathlib import Path


def is_folder_empty(folder_path):
    for item in Path(folder_path).iterdir():
        if item.is_file():
            return False
        if item.is_dir() and not is_folder_empty(item):
            return False
    return True


def delete_empty_dataset_folders(root_dir):
    root_path = Path(root_dir)
    deleted_count = 0

    for current_dir in sorted(root_path.rglob('*'), key=lambda p: len(p.parts), reverse=True):
        if current_dir.is_dir() and is_folder_empty(current_dir):
            try:
                os.rmdir(current_dir)
                print(f"Deleted empty folder: {current_dir}")
                deleted_count += 1
            except OSError as e:
                print(f"Error deleting {current_dir}: {e}")

    print(f"\nTotal deleted empty folders: {deleted_count}")


if __name__ == "__main__":
    base_path = pathlib.Path(__file__).resolve().parent.parent
    delete_empty_dataset_folders(os.path.join(base_path, 'results'))
    delete_empty_dataset_folders(os.path.join(base_path, 'formatted_results'))
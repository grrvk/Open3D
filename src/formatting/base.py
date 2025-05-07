class BaseDatasetFormatter:
    def __init__(self, split: list, prev_dataset_path: str):
        self.split = split
        self.prev_dataset_path = prev_dataset_path
        self.assign_base_folder()

    def assign_base_folder(self):
        pass

    def generate_folder_structure(self):
        assert 2 <= len(self.split) <= 3, 'Split must be into 2 or 3 folders (train, val, test(optional))'
        assert sum(self.split) == 1, 'Split rates must summ to 1'

        if len(self.split) == 2:
            folder_types = ['train', 'val']
        else:
            folder_types = ['train', 'val', 'test']

        return folder_types

    def generate_folders(self):
        pass

    def generate(self, **kwargs):
        pass

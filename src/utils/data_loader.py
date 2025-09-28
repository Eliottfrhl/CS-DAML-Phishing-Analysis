from ucimlrepo import fetch_ucirepo

class DataLoader:
    def __init__(self, dataset_id):
        self.dataset_id = dataset_id
        self.dataset = None

    def load_data(self):
        self.dataset = fetch_ucirepo(id=self.dataset_id)
        return self.dataset.data.features, self.dataset.data.targets, self.dataset.metadata, self.dataset.variables

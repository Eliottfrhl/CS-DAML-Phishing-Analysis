from utils.data_loader import DataLoader

def main():
    dataset_id = 967
    data_loader = DataLoader(dataset_id)
    features, targets, metadata, variables = data_loader.load_data()
    
    print("Features:", features)
    print("Targets:", targets)
    print("Metadata:", metadata)
    print("Variables:", variables)

main()
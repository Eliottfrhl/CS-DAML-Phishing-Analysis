from utils.data_loader import DataLoader
from utils.url_parser import URLParser

def main():
    dataset_id = 967
    data_loader = DataLoader(dataset_id)
    features, targets, metadata, variables = data_loader.load_data()

    url = "http://examplephishingsite.com"
    parsed_url = URLParser.parse(url)

    print("Parsed URL Domain Parts:", parsed_url)
    print("Features:", features)
    print("Targets:", targets)
    print("Metadata:", metadata)
    print("Variables:", variables)

if __name__ == "__main__":
    main()
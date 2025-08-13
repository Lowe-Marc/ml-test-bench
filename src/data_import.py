import kagglehub

DATA_DIR = "data"

def import_kaggle_fraud_data():
    # Download latest version
    path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
    print("Path to dataset files:", path)


if __name__ == "__main__":
    import_kaggle_fraud_data()

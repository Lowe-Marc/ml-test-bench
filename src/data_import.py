import kagglehub
import os
import shutil


# TODO: Abstract this? It's duplicated in data_generator.py
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "raw_data")


def import_kaggle_fraud_data(target_path: str = None) -> str:
    download_path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
    print(f"Downloaded to kagglehub cache: {download_path}")
    
    if target_path:
        os.makedirs(target_path, exist_ok=True)
        
        for file in os.listdir(download_path):
            src_file = os.path.join(download_path, file)
            dst_file = os.path.join(target_path, file)
            if os.path.isfile(src_file):
                shutil.copy2(src_file, dst_file)
                print(f"Copied {file} to {target_path}")
    else:
        target_path = download_path

    print(f"Dataset available in: {target_path}")
    return target_path


if __name__ == "__main__":
    fraud_data_dir = os.path.join(RAW_DATA_DIR, "kaggle", "fraud")
    import_kaggle_fraud_data(fraud_data_dir)

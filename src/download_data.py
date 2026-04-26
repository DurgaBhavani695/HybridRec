import urllib.request
import zipfile
import os

def download_movielens():
    url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    raw_dir = os.path.join("data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    zip_path = os.path.join(raw_dir, "ml-latest-small.zip")

    if not os.path.exists(os.path.join(raw_dir, "movies.csv")):
        print("Downloading MovieLens Small dataset...")
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(raw_dir)
        
        extracted_folder = os.path.join(raw_dir, "ml-latest-small")
        for filename in ["movies.csv", "ratings.csv"]:
            os.rename(os.path.join(extracted_folder, filename), os.path.join(raw_dir, filename))
        print("Dataset ready!")
    else:
        print("Dataset already exists.")

if __name__ == "__main__":
    download_movielens()

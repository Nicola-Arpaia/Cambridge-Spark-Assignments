import urllib.request
import zipfile
from io import BytesIO

DATASET_URL = "https://s3-eu-west-1.amazonaws.com/kate-datasets/hackernews/HNStories.zip"
DATA_DIR = "data"

if __name__ == "__main__":

    req = urllib.request.urlopen(DATASET_URL)
    data = BytesIO(req.read())

    with zipfile.ZipFile(data, "r") as zipref:
        zipref.extractall(DATA_DIR)

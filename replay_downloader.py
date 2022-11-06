import io
import os.path

import requests, zipfile
from tqdm import tqdm

if __name__ == '__main__':
    for dir in ["1v1", "2v2", "3v3"]:
        for i in tqdm(range(10)):
            r = requests.get("http://nevillewalo.ch/assets/{dir}_{i}.zip".format(dir=dir, i=i), stream=True)
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall("./Replays")

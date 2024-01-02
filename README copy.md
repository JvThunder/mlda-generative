# mlda-generative

## Getting the dataset:
```
Terminal:
mkdir data_faces
wget https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/celeba.zip

Python:
import zipfile
with zipfile.ZipFile("celeba.zip","r") as zip_ref:
   zip_ref.extractall("data_faces/")
```
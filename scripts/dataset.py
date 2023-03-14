import boto3
import os
import tarfile

# fetch credentials from env variables
aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY')

# setup a AWS S3 client/resource
s3 = boto3.resource(
    's3', 
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    )

# point the resource at the existing bucket
bucket = s3.Bucket('anyoneai-datasets')

# download the training dataset
with open('../data/training_image_set.tgz', 'wb') as data:
    bucket.download_fileobj('cars196/car_ims.tgz', data)
# download the dataset labels
with open('../data/car_dataset_labels.csv', 'wb') as data:
    bucket.download_fileobj('cars196/car_dataset_labels.csv', data)

# extract dataset
tar = tarfile.open("../data/training_image_set.tgz", "r")
for item in tar:
    tar.extract(item, "../data")

#!/bin/bash
# ==========================================================
# Script downloads dataset from GCS using gsutil 

# To run the script, use the following command:
# chmod +x download_dataset.sh
# ./download_dataset.sh <bucket_name> <project_name> <data_type>

# ==================================================================

# default values (Change these as needed)
BUCKET_NAME=${1:-"dataset_bucket"}
PROJECT_NAME=${2:-"Project"}
DATA_TYPE=${3:-"images"}

# remove existing dataset directory
rm -rf ~/$PROJECT_NAME/
# create new dataset directory
mkdir -p ~/$PROJECT_NAME

# download dataset from GCS bucket
mkdir -p ~/$PROJECT_NAME/dataset/train/$DATA_TYPE
mkdir -p ~/$PROJECT_NAME/dataset/test/$DATA_TYPE

# install Google Cloud SDK if not already installed
if ! command -v gcloud &> /dev/null
then
    echo "Google Cloud SDK not found. Installing..."
    curl https://sdk.cloud.google.com | bash
    exec -l $SHELL
fi
# authenticate with Google Cloud
gcloud auth login
gcloud config set project $BUCKET_NAME

gsutil -m cp -r "gs://$BUCKET_NAME/test/$DATA_TYPE/*" ~/$PROJECT_NAME/dataset/test/$DATA_TYPE/
gsutil -m cp -r "gs://$BUCKET_NAME/train/$DATA_TYPE/*" ~/$PROJECT_NAME/dataset/train/$DATA_TYPE/

# ==============================================================

# Note: The above script assumes that the dataset is organized in a specific structure in the GCS bucket.


# Run the following commands to verify the download
# TEST_COUNT=$(find ~/$PROJECT_NAME/dataset/test/$DATA_TYPE/ -type f | wc -l | tr -d ' ')
# TRAIN_COUNT=$(find ~/$PROJECT_NAME/dataset/train/$DATA_TYPE/ -type f | wc -l | tr -d ' ')
# ==============================================================


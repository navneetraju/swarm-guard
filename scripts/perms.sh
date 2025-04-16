PROJECT_ID=swarm-456603
BUCKET=swarm_guard_dataset
SA=47781043798-compute@developer.gserviceaccount.com

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:$SA" \
  --role="roles/storage.objectViewer"

gsutil iam ch \
  serviceAccount:$SA:objectViewer \
  gs://$BUCKET

BUCKET=swarm_guard_models
SA=47781043798-compute@developer.gserviceaccount.com

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:$SA" \
  --role="roles/storage.objectViewer"

gsutil iam ch \
  serviceAccount:$SA:objectViewer \
  gs://$BUCKET

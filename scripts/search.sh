#!/usr/bin/env bash

# usage: run_mmm_search.sh OUTPUT_DIR
usage() {
  echo "Usage: $0 OUTPUT_DIR"
  echo
  echo "  OUTPUT_DIR                 where to write your search results"
  echo
  echo "Example:"
  echo "  $0 ~/models/multi_modal_search/focal_loss"
  exit 1
}

if [ $# -ne 1 ]; then
  usage
fi

OUTPUT_DIR=$1

python3 -m src.mmm_search \
  --dataset-root-dir ~/data \
  --search-results-output-file-path "${OUTPUT_DIR}" \
  --graph-encoder-model-path ~/models/graph/graph_encoder.pth

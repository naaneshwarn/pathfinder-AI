"""
train.py — Run this once to train and save the model to disk.

Usage:
    python train.py --data ../data/postings.csv --output ../models/job_recommender.pkl
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from recommender import JobRecommender


def main():
    parser = argparse.ArgumentParser(description="Train Job Recommender model")
    parser.add_argument("--data", default="../data/postings.csv", help="Path to postings.csv")
    parser.add_argument("--output", default="../models/job_recommender.pkl", help="Output model path")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    recommender = JobRecommender()
    recommender.train(args.data)
    recommender.save(args.output)
    print("[DONE] Model ready. Start the Flask API with: python app.py")


if __name__ == "__main__":
    main()

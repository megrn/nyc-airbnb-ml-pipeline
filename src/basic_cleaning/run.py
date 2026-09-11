#!/usr/bin/env python
"""Clean an Airbnb CSV artifact and publish the result to W&B."""

import argparse
import logging

import pandas as pd
import wandb

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
LOGGER = logging.getLogger(__name__)


def go(args: argparse.Namespace) -> None:
    """Download, clean, save, and log the dataset described by ``args``."""
    run = wandb.init(job_type="basic_cleaning")
    run.config.update(vars(args))

    input_path = run.use_artifact(args.input_artifact).file()
    dataframe = pd.read_csv(input_path)
    LOGGER.info("Downloaded data has %s rows and %s columns", *dataframe.shape)

    dataframe = dataframe.drop_duplicates()
    dataframe = dataframe.dropna(subset=["price"])
    dataframe = dataframe[
        dataframe["price"].between(args.min_price, args.max_price)
    ].copy()
    dataframe["last_review"] = pd.to_datetime(
        dataframe["last_review"],
        errors="coerce",
    )

    LOGGER.info("Cleaned data has %s rows and %s columns", *dataframe.shape)
    dataframe.to_csv("clean_sample.csv", index=False)

    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file("clean_sample.csv")
    run.log_artifact(artifact)
    run.finish()


def parse_args() -> argparse.Namespace:
    """Parse command-line parameters defined by the MLproject file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_artifact", type=str, required=True)
    parser.add_argument("--output_artifact", type=str, required=True)
    parser.add_argument("--output_type", type=str, required=True)
    parser.add_argument("--output_description", type=str, required=True)
    parser.add_argument("--min_price", type=float, required=True)
    parser.add_argument("--max_price", type=float, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    go(parse_args())

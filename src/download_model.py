from __future__ import annotations

import argparse
import logging

from modeling import DEFAULT_MODEL_NAME, normalize_model_name_or_path


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a Hugging Face model snapshot for offline use.")
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--local_dir", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--local_dir_use_symlinks", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    args = parse_args()
    model_name = normalize_model_name_or_path(args.model_name_or_path)

    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id=model_name,
        revision=args.revision,
        cache_dir=args.cache_dir,
        local_dir=args.local_dir,
        local_dir_use_symlinks=args.local_dir_use_symlinks,
    )
    logger.info("Downloaded %s to %s", model_name, path)
    print(path)


if __name__ == "__main__":
    main()

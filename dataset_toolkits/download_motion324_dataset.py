import os
import argparse
import tarfile
from huggingface_hub import snapshot_download

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

REPO_ID = "River-Chen/Motion324"
REPO_TYPE = "dataset"
NUM_PARTS = 17  # train/part_0001.tar.gz ... part_0017.tar.gz


def main():
    p = argparse.ArgumentParser(
        description="Download Motion324 train shards."
    )
    p.add_argument("--parts", type=int, nargs="+", default=None,
                   help=f"Part numbers to download, 1-indexed in [1, {NUM_PARTS}] "
                        f"(e.g., --parts 1 4 5 10). Default: all parts.")
    p.add_argument("--local-dir", type=str, required=True,
                   help="Directory to download and extract into.")
    p.add_argument("--no-extract", action="store_true", default=False,
                   help="Skip extraction; just download .tar.gz shards.")
    p.add_argument("--delete-tars", action="store_true", default=True,
                   help="Remove .tar.gz after successful extraction.")
    args = p.parse_args()

    if args.parts is not None:
        bad = [k for k in args.parts if not (1 <= k <= NUM_PARTS)]
        if bad:
            raise ValueError(f"parts out of range [1, {NUM_PARTS}]: {bad}")
        my_parts = sorted(set(args.parts))
    else:
        my_parts = list(range(1, NUM_PARTS + 1))
    my_files = [f"train/part_{k:04d}.tar.gz" for k in my_parts]

    print(f"Downloading {len(my_files)} shard(s):")
    for f in my_files:
        print(f"  {f}")

    os.makedirs(args.local_dir, exist_ok=True)
    snapshot_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        allow_patterns=my_files,
        local_dir=args.local_dir,
    )
    print(f"Download complete -> {args.local_dir}")

    if args.no_extract:
        return

    train_dir = os.path.join(args.local_dir, "train")
    marker_dir = os.path.join(args.local_dir, ".extracted")
    os.makedirs(marker_dir, exist_ok=True)
    for f in my_files:
        tar_path = os.path.join(args.local_dir, f)
        marker = os.path.join(marker_dir, os.path.basename(f) + ".done")
        if not os.path.exists(tar_path):
            if os.path.exists(marker):
                print(f"  SKIP (already extracted, tar deleted): {f}")
                continue
            print(f"  MISSING: {tar_path}")
            continue
        if os.path.exists(marker):
            print(f"  SKIP (already extracted): {f}")
            if args.delete_tars:
                os.remove(tar_path)
            continue
        print(f"Extracting {tar_path} ...")
        try:
            with tarfile.open(tar_path, "r:gz") as tf:
                tf.extractall(train_dir, filter="data")
        except Exception as e:
            print(f"  FAILED: {tar_path}: {e}")
            continue
        with open(marker, "w") as mf:
            mf.write("done\n")
        if args.delete_tars:
            os.remove(tar_path)

    print(f"Done -> {train_dir}")


if __name__ == "__main__":
    main()

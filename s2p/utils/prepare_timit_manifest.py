import os
import os.path as osp
import argparse
import random
import glob
import soundfile as sf
import multiprocessing as mp
import tqdm

def get_frames(file_path):
    
    frames = sf.info(file_path).frames
    return frames

def write_manifest(root, manifests, output_fpath, sort=True, relpath=True):
    if relpath:
        manifests = [(osp.relpath(fpath, root), frames) for fpath, frames in manifests]
    if sort:
        manifests = sorted(manifests, key=lambda x: x[0])
    with open(output_fpath, "w") as f:
        f.write(root + "\n")
        for manifest in manifests:
            f.write(f"{manifest[0]}\t{manifest[1]}" + "\n")

def prepare_unmatched_manifest(root, manifest_dir, seed=42, ext="wav"):
    train_count = 3000
    unpaired_text_count = 1000
    valid_count = 620
    total_count = train_count + unpaired_text_count + valid_count
    DR_count = 8
    dir_path = os.path.realpath(root)
    rand = random.Random(seed)
    train_manifests = []
    unpaired_manifests = []
    valid_manifests = []
    pool = mp.Pool()
    for i in range(1, DR_count+1):
        search_path = os.path.join(dir_path, f"DR{i}/**/*." + ext)
        fpaths = [os.path.realpath(fname) for fname in glob.iglob(search_path, recursive=True)]
        frames = list(tqdm.tqdm(pool.imap(get_frames, fpaths), total=len(fpaths)))
        manifests = list(zip(fpaths, frames))
        rand.shuffle(manifests)
        cur_dr_count = len(manifests)
        new_train_counts = int(train_count * cur_dr_count / total_count + 0.5)
        new_train_manifests = manifests[:new_train_counts]
        new_unpaired_counts = int(unpaired_text_count * cur_dr_count / total_count + 0.5)
        new_unpaired_manifests = manifests[new_train_counts:new_train_counts+new_unpaired_counts]
        new_valid_manifests = manifests[new_train_counts+new_unpaired_counts:]
        train_manifests.extend(new_train_manifests)
        unpaired_manifests.extend(new_unpaired_manifests)
        valid_manifests.extend(new_valid_manifests)
    print(len(train_manifests), len(unpaired_manifests), len(valid_manifests))
    all_manifests = train_manifests + unpaired_manifests + valid_manifests
    train_manifests = all_manifests[:train_count]
    unpaired_manifests = all_manifests[train_count:train_count+unpaired_text_count]
    valid_manifests = all_manifests[train_count+unpaired_text_count:]
    write_manifest(dir_path, train_manifests, osp.join(manifest_dir, "train.tsv"))
    write_manifest(dir_path, unpaired_manifests, osp.join(manifest_dir, "unpaired.tsv"))
    write_manifest(dir_path, valid_manifests, osp.join(manifest_dir, "valid.tsv"))


def main(args):
    root = args.root
    prepare_unmatched_manifest(root, args.manifest_dir, seed=args.seed, ext=args.ext)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="/work/c/timit/data/TRAIN",
        help="root directory of timit dataset",
    )
    parser.add_argument(
        "--manifest_dir",
        default="/work/c/timit/manifest",
        help="directory to store manifest files",
    )
    parser.add_argument(
        "--ext",
        default="wav",
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
    )
    args = parser.parse_args()

    main(args)
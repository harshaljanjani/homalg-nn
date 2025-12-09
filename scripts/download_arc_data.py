# https://github.com/fchollet/ARC-AGI
# data/arc/
# >> training/
# >>> *.json (400 train tasks)
# >> evaluation/
# >>> *.json (400 eval tasks)
import sys
import json
import shutil
import subprocess
from pathlib import Path
ARC_REPO_URL = "https://github.com/fchollet/ARC-AGI.git"
DATA_DIR = Path(__file__).parent.parent / "data" / "arc"
TEMP_CLONE_DIR = Path(__file__).parent.parent / "data" / "arc_temp"


def clone_repository():
    """Clone ARC-AGI repository to temporary directory."""
    print("\nCloning ARC-AGI repository.")
    print(f"URL: {ARC_REPO_URL}")
    if TEMP_CLONE_DIR.exists():
        print(f"Removing existing temp directory: {TEMP_CLONE_DIR}")
        shutil.rmtree(TEMP_CLONE_DIR)
    try:
        _ = subprocess.run(
            ["git", "clone", "--depth", "1", ARC_REPO_URL, str(TEMP_CLONE_DIR)],
            capture_output=True,
            text=True,
            check=True
        )
        print("Repository cloned successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error cloning repository: {e.stderr}")
        return False
    except FileNotFoundError:
        print("Error: git command not found. Please install git:")
        print("  Windows: https://git-scm.com/download/win")
        print("  Or download data manually from: https://github.com/fchollet/ARC-AGI")
        return False


def copy_data():
    """Copy training and evaluation data to target directory."""
    print("\nCopying data files.")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    src_training = TEMP_CLONE_DIR / "data" / "training"
    dst_training = DATA_DIR / "training"
    if src_training.exists():
        if dst_training.exists():
            shutil.rmtree(dst_training)
        shutil.copytree(src_training, dst_training)
        num_training = len(list(dst_training.glob("*.json")))
        print(f"  Copied {num_training} training tasks")
    else:
        print(f"  Warning: Training directory not found: {src_training}")
        return False
    src_evaluation = TEMP_CLONE_DIR / "data" / "evaluation"
    dst_evaluation = DATA_DIR / "evaluation"
    if src_evaluation.exists():
        if dst_evaluation.exists():
            shutil.rmtree(dst_evaluation)
        shutil.copytree(src_evaluation, dst_evaluation)
        num_evaluation = len(list(dst_evaluation.glob("*.json")))
        print(f"  Copied {num_evaluation} evaluation tasks")
    else:
        print(f"  Warning: Evaluation directory not found: {src_evaluation}")
        return False
    return True


def cleanup():
    """Remove temporary clone directory."""
    print("\nCleaning up temporary files.")
    if TEMP_CLONE_DIR.exists():
        shutil.rmtree(TEMP_CLONE_DIR)
        print("  Removed temporary clone directory")


def verify_dataset():
    """Verify downloaded dataset is complete."""
    print("\nVerifying dataset.")
    checks = []
    training_dir = DATA_DIR / "training"

    # training split
    if training_dir.exists():
        training_files = list(training_dir.glob("*.json"))
        num_training = len(training_files)
        checks.append(("Training tasks", num_training == 400, num_training))
        print(f"  Training: {num_training} tasks (expected 400)")
        if training_files:
            try:
                with open(training_files[0], 'r') as f:
                    data = json.load(f)
                    has_train = 'train' in data
                    has_test = 'test' in data
                    checks.append(("Training file format", has_train and has_test, None))
                    if has_train and has_test:
                        print(f"  Format: Valid (has 'train' and 'test' fields)")
                    else:
                        print(f"  Format: Invalid (missing fields)")
            except Exception as e:
                checks.append(("Training file format", False, None))
                print(f"  Format: Error - {e}")
    else:
        checks.append(("Training tasks", False, 0))
        print("  Training: MISSING")

    # eval split
    evaluation_dir = DATA_DIR / "evaluation"
    if evaluation_dir.exists():
        evaluation_files = list(evaluation_dir.glob("*.json"))
        num_evaluation = len(evaluation_files)
        checks.append(("Evaluation tasks", num_evaluation == 400, num_evaluation))
        print(f"  Evaluation: {num_evaluation} tasks (expected 400)")
    else:
        checks.append(("Evaluation tasks", False, 0))
        print("  Evaluation: MISSING")
    
    # summary
    all_good = all(check[1] for check in checks)
    if all_good:
        print("\nDataset verification passed!")
        return True
    else:
        print("\nDataset verification failed:")
        for name, passed, value in checks:
            if not passed:
                if value is not None:
                    print(f"  - {name}: got {value}")
                else:
                    print(f"  - {name}")
        return False


def create_subset(num_tasks: int = 20):
    """Create small subset for quick testing."""
    print(f"\nCreating subset of {num_tasks} tasks for quick testing.")
    training_dir = DATA_DIR / "training"
    subset_dir = DATA_DIR / "subset"
    subset_dir.mkdir(parents=True, exist_ok=True)
    if not training_dir.exists():
        print("Training data not found. Run download first.")
        return False
    training_files = sorted(training_dir.glob("*.json"))[:num_tasks]
    if len(training_files) < num_tasks:
        print(f"Warning: Only found {len(training_files)} training files")
    for task_file in training_files:
        shutil.copy(task_file, subset_dir / task_file.name)
    print(f"Created subset with {len(training_files)} tasks at {subset_dir}")
    return True


def main():
    """Download ARC-AGI dataset."""
    print("\nARC-AGI Dataset Downloader")
    if not clone_repository():
        print("\nDownload failed. Please check errors above.")
        return 1
    if not copy_data():
        print("\nCopy failed. Please check errors above.")
        cleanup()
        return 1
    cleanup()
    if not verify_dataset():
        print("\nVerification failed. Please check errors above.")
        return 1
    create_subset(num_tasks=20)
    print("\nDownload complete!")
    print(f"Dataset location: {DATA_DIR}")
    return 0


if __name__ == '__main__':
    sys.exit(main())

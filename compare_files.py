import os
import filecmp
import hashlib
import torch
import pickle


def compare_files(file1, file2, method="hash"):
    """
    Compare two files to check if they are identical.

    Args:
        file1 (str): Path to the first file.
        file2 (str): Path to the second file.
        method (str): Method to compare files. Options are:
                      - "hash": Compare files based on their hash (default).
                      - "byte": Compare files byte by byte.
                      - "metadata": Compare based on file metadata (size, timestamp).

    Returns:
        bool: True if files are identical, False otherwise.
    """

    # Check if both files exist
    if not os.path.exists(file1) or not os.path.exists(file2):
        raise FileNotFoundError(f"One or both files do not exist: {file1}, {file2}")

    # if both files are .pth files, compare using torch
    if file1.endswith('.pth') and file2.endswith('.pth'):
        state_dict1 = torch.load(file1)
        state_dict2 = torch.load(file2)
        return compare_state_dicts(state_dict1, state_dict2)

    if file1.endswith('.pkl') and file2.endswith('.pkl'):
        return compare_pkl(file1, file2)

    # Method 1: Compare using file hash (MD5, SHA-256, etc.)
    if method == "hash":
        return compare_files_by_hash(file1, file2)

    # Method 2: Compare byte by byte
    elif method == "byte":
        return compare_files_byte_by_byte(file1, file2)

    # Method 3: Compare file metadata (size and modification time)
    elif method == "metadata":
        return compare_files_by_metadata(file1, file2)

    else:
        raise ValueError(f"Unknown comparison method: {method}")


def compare_state_dicts(state_dict1, state_dict2):
    for key in state_dict1:
        if not torch.equal(state_dict1[key], state_dict2[key]):
            return False, f"Diff in {key}"
    return True, None

def compare_pkl(pkl1, pkl2):
    with open(pkl1, 'rb') as f:
        data1 = pickle.load(f)
    with open(pkl2, 'rb') as f:
        data2 = pickle.load(f)
    for d1, d2 in zip(data1, data2):
        if not torch.equal(d1, d2):
            return False
    return True


def compare_files_by_hash(file1, file2, hash_algorithm="sha256"):
    """
    Compare two files by generating and comparing their hash digests.

    Args:
        file1 (str): Path to the first file.
        file2 (str): Path to the second file.
        hash_algorithm (str): Hashing algorithm to use (e.g., 'md5', 'sha256').

    Returns:
        bool: True if files are identical, False otherwise.
    """
    hash1 = generate_file_hash(file1, hash_algorithm)
    hash2 = generate_file_hash(file2, hash_algorithm)
    return hash1 == hash2


def generate_file_hash(file_path, hash_algorithm="sha256"):
    """
    Generate a hash digest for a file.

    Args:
        file_path (str): Path to the file.
        hash_algorithm (str): Hashing algorithm to use (e.g., 'md5', 'sha256').

    Returns:
        str: Hexadecimal hash digest of the file.
    """
    hash_function = hashlib.new(hash_algorithm)
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            hash_function.update(chunk)
    return hash_function.hexdigest()


def compare_files_byte_by_byte(file1, file2):
    """
    Compare two files byte by byte to check if they are identical.

    Args:
        file1 (str): Path to the first file.
        file2 (str): Path to the second file.

    Returns:
        bool: True if files are identical, False otherwise.
    """
    with open(file1, "rb") as f1, open(file2, "rb") as f2:
        while True:
            chunk1 = f1.read(8192)
            chunk2 = f2.read(8192)
            if chunk1 != chunk2:
                return False
            if not chunk1:  # Reached the end of both files
                break
    return True


def compare_files_by_metadata(file1, file2):
    """
    Compare two files based on their metadata: size and modification time.

    Args:
        file1 (str): Path to the first file.
        file2 (str): Path to the second file.

    Returns:
        bool: True if files are identical based on metadata, False otherwise.
    """
    stat1 = os.stat(file1)
    stat2 = os.stat(file2)
    return (stat1.st_size == stat2.st_size) and (stat1.st_mtime == stat2.st_mtime)

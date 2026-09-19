"""Verify all distributed files listed in provenance/SHA256SUMS.csv."""
import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    failures = []
    with (ROOT/'provenance/SHA256SUMS.csv').open(encoding='utf-8',newline='') as stream:
        records = list(csv.DictReader(stream))
    for row in records:
        path = (ROOT/row['path']).resolve()
        if not path.is_relative_to(ROOT):
            failures.append(row['path'] + ': invalid relative path')
            continue
        if not path.is_file():
            failures.append(row['path'] + ': missing')
            continue
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream,'sha256').hexdigest()
        if digest != row['sha256']: failures.append(row['path'] + ': changed')
    if failures:
        raise SystemExit('\n'.join(failures))
    print(f'Verified {len(records)} distributed files. New files under runs/ are not part of the manifest.')

if __name__ == '__main__': main()

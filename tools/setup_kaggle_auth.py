"""Write ~/.kaggle/kaggle.json from an API token, without exposing the key.

The key is read with getpass (not echoed to the screen, not stored in shell
history) and written straight to the credentials file.

Run:  python tools/setup_kaggle_auth.py
"""
from __future__ import annotations

import getpass
import json
import os
import pathlib
import stat


def main() -> None:
    home = pathlib.Path(os.path.expanduser("~"))
    cred_dir = home / ".kaggle"
    cred = cred_dir / "kaggle.json"

    if cred.exists():
        print(f"{cred} already exists.")
        if input("overwrite? [y/N] ").strip().lower() != "y":
            print("kept the existing file.")
            return

    print("Kaggle username is the name in your profile URL "
          "(kaggle.com/<username>), e.g. hidehikofukaya")
    username = input("Kaggle username: ").strip()
    if not username:
        raise SystemExit("username is required")
    key = getpass.getpass("API token/key (input is hidden): ").strip()
    if not key:
        raise SystemExit("key is required")

    cred_dir.mkdir(parents=True, exist_ok=True)
    cred.write_text(json.dumps({"username": username, "key": key}),
                    encoding="utf-8")
    try:                       # kaggle CLI warns if the file is world-readable
        cred.chmod(stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass
    print(f"\nwrote {cred}  (key length {len(key)}, not shown)")
    print("next: ask Claude to run the connection test")


if __name__ == "__main__":
    main()

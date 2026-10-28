"""Load .env tolerantly: strip whitespace, inline '# comments', and a leading 'Bearer '."""
import os, re
from pathlib import Path

def load_env(path=Path(__file__).resolve().parent.parent / ".env"):
    for line in Path(path).read_text().splitlines():
        if "=" not in line or line.lstrip().startswith("#"):
            continue
        k, v = line.split("=", 1)
        v = re.sub(r"\s+#.*$", "", v).strip().strip('"').strip("'")
        v = re.sub(r"^Bearer\s+", "", v)
        os.environ.setdefault(k.strip(), v)

load_env()

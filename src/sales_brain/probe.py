"""Explicit anonymous provider check; never sends a customer/sales message."""
import json
from .app import transport


def main():
    print(json.dumps(transport().probe_v21(),ensure_ascii=False))


if __name__ == '__main__':
    main()

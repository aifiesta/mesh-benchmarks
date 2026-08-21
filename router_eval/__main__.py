"""Enable `python -m router_eval` as an alias for `python -m router_eval.replay`."""

from router_eval.replay import main

if __name__ == "__main__":
    raise SystemExit(main())

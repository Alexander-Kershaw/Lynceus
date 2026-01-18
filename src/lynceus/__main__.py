from __future__ import annotations
from lynceus.config import SimConfig
from termcolor import colored


# Entry point for the LYNCEUS simulation framework

def main() -> None:
    cfg = SimConfig()
    print(colored("LYNCEUS boot successful", "light_red"))
    print(colored(f"dt={cfg.dt}, steps={cfg.steps}, seed={cfg.seed}", "light_yellow"))


if __name__ == "__main__":
    main()

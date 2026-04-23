"""Stage 2 driver: parquets -> char_panel_raw -> panel.pickle (+ Z.parquet).

Idempotent. Run after any refresh of Data/parquet/*.
"""
from rsgp.characteristics import build_char_panel
from rsgp.preprocess import build_and_save
from rsgp.macro import build_macro_panel


def main() -> None:
    build_char_panel()
    build_and_save()
    build_macro_panel()


if __name__ == "__main__":
    main()

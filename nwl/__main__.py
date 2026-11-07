"""`python -m nwl` komutunun giriş noktası."""

from nwl import __version__


def main() -> None:
    print(f"NWL {__version__} hazır.")


if __name__ == "__main__":
    main()

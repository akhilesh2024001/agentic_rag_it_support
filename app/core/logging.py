import logging
def configure_logging()->Name:
    logging.basicConfig(
        level=logging.INFO,
          format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
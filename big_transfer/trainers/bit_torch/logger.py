import logging, os


def setup_logger(log_path):
    """Creates and returns a fancy logger."""

    # Why is setting up proper logging so !@?#! ugly?

    folder_name = os.path.dirname(log_path)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


    # os.makedirs(os.path.join(args.logdir, args.name), exist_ok=True)
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
                },
            },
            "handlers": {
                "stderr": {
                    "level": "INFO",
                    "formatter": "standard",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                },
                "logfile": {
                    "level": "DEBUG",
                    "formatter": "standard",
                    "class": "logging.FileHandler",
                    "filename": log_path,
                    "mode": "a",
                },
            },
            "loggers": {
                "": {
                    "handlers": ["stderr", "logfile"],
                    "level": "DEBUG",
                    "propagate": True,
                },
            },
        }
    )
    logger = logging.getLogger("train")
    logger.flush = lambda: [h.flush() for h in logger.handlers]
    # logger.info(args)
    return logger

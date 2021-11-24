import os
import logging


def init_logger(args):
    dir_name = './log/'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    logfilepath = os.path.join(dir_name, f'{args.task_name}.log')

    filefmt = "%(asctime)-15s %(levelname)s %(message)s"
    filedatefmt = "%d %b %Y %H:%M:%S"
    fileformatter = logging.Formatter(filefmt, filedatefmt)

    sfmt = "%(asctime)-15s %(levelname)s %(message)s"
    sdatefmt = "%d %b %H:%M:%S"
    sformatter = logging.Formatter(sfmt, sdatefmt)
    if args.state is None or args.state.lower() == 'info':
        level = logging.INFO
    elif args.state.lower() == 'debug':
        level = logging.DEBUG
    elif args.state.lower() == 'error':
        level = logging.ERROR
    elif args.state.lower() == 'warning':
        level = logging.WARNING
    elif args.state.lower() == 'critical':
        level = logging.CRITICAL
    else:
        level = logging.INFO
    fh = logging.FileHandler(logfilepath)
    fh.setLevel(level)
    fh.setFormatter(fileformatter)

    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(sformatter)

    logging.basicConfig(level=level, handlers=[fh, sh])
import sys, logging


def get_logger(timestamp='today', name='keyphrase-generate'):
    logger = logging.getLogger(name)
    fmt = logging.Formatter(fmt='%(message)s', datefmt='%Y%m%d:%H%M%S')

    fhdler = logging.FileHandler('./result_%s.log' % timestamp)
    fhdler.setFormatter(fmt=fmt)

    shdler = logging.StreamHandler(sys.stdout)
    shdler.setFormatter(fmt)

    logger.addHandler(shdler)
    logger.addHandler(fhdler)

    logger.setLevel(logging.INFO)

    return logger


if __name__ == '__main__':
    logger = get_logger('2019')
    for i in range(12):
        logger.info('%s-%s hahahhhhh', i, i + 1)

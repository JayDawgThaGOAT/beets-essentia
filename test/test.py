import logging
import sys

from beets.library import Item

from beetsplug.essentia import EssentiaInterface
import confuse

def do_stuff():
    config = confuse.Configuration('beets-essentia', __name__)
    config.set_file('../beetsplug/essentia/config_default.yml')

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    es = EssentiaInterface(config, logger)
    es.analyse([Item.from_path('/mnt/idata/local/users/auridh/Music/Picard/GOGAE.flac')])

if __name__ == '__main__':
    do_stuff()

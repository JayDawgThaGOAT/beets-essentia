import logging
import sys

import mediafile
from beets.library import Item
from confuse import ConfigView

from beetsplug.essentia import EssentiaInterface
import confuse

def add_media_fields(config: ConfigView):
    mood_mapping = config['tags']['mood']['mapping'].get(str)
    mood = mediafile.MediaField(
        mediafile.MP3DescStorageStyle(mood_mapping),
        mediafile.StorageStyle(mood_mapping)
    )
    mediafile.MediaFile.add_field('mood', mood)
    Item._media_fields.add('mood')

def do_stuff():
    config = confuse.Configuration('beets-essentia', __name__)
    config.set_file('../beetsplug/essentia/config_default.yml')

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    add_media_fields(config)

    es = EssentiaInterface(config, logger)
    gogae = Item.from_path('/mnt/idata/local/users/auridh/Music/Picard/GOGAE.flac')
    thunder = Item.from_path('/mnt/idata/local/users/auridh/Music/Picard/THUNDER.flac')

    es.analyse([gogae, thunder])

if __name__ == '__main__':
    do_stuff()

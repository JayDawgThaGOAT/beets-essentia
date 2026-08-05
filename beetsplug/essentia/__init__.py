import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import essentia
essentia.log.infoActive = False

import mediafile
from beets.importer import ImportTask
from beets.plugins import BeetsPlugin
from beets.ui import Subcommand
from confuse import ConfigSource, load_yaml
from beetsplug.essentia.interface import EssentiaInterface
from beetsplug.essentia.command import EssentiaCommand


class EssentiaPlugin(BeetsPlugin):
    _default_plugin_config_file_name_ = 'config_default.yml'

    def __init__(self) -> None:
        super(EssentiaPlugin, self).__init__()

        config_file_path = os.path.join(os.path.dirname(__file__), self._default_plugin_config_file_name_)
        source = ConfigSource(load_yaml(config_file_path) or {}, config_file_path)
        self.config.add(source)

        mood_mapping = self.config['tags']['mood']['mapping'].get(str)
        mood = mediafile.MediaField(
            mediafile.MP3DescStorageStyle(mood_mapping),
            mediafile.StorageStyle(mood_mapping)
        )
        self.add_media_field('mood', mood)

        if self.config['auto'].get(bool):
            self.import_stages = [self.imported]

    def imported(self, _, task: ImportTask) -> None:
        if not self.config['auto'].get(bool):
            return

        items = task.imported_items()
        if not items:
            return

        es = EssentiaInterface(self.config, self._log)
        es.analyse(items)

    def commands(self) -> list[Subcommand]:
        return [EssentiaCommand(self.config, self._log)]

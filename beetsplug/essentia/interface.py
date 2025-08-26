import json
import os.path
import urllib.request
from logging import Logger

from os import path
from pathlib import Path

from beets.library import Item
from confuse import ConfigView

import essentia.standard as es

class EssentiaModel:
    def __init__(
            self,
            log: Logger,
            mpath: str,
            config: ConfigView,
            name: str
    ):
        self.name = name
        self.config = config
        self._log = log
        self._path = mpath

        model_weight_file, model_metadata = self._load(self.config['model'].get(str))
        self._model = model_weight_file
        self.model_metadata = model_metadata

        embedding_model = (self.model_metadata['inference']['embedding_model']['link']
                           .replace('https://essentia.upf.edu/models/', '')
                           .replace('.pb', '')) \
            if 'embedding_model' in self.model_metadata['inference'] else None
        model_weight_file, model_metadata = self._load(embedding_model)
        self._embedding_model = model_weight_file
        self.embedding_model_metadata = model_metadata

        handler, embedding_handler = self._load_handlers()
        self._handler = handler
        self._embedding_handler = embedding_handler

    def _get_handler_input(self, embedding: bool = False) -> str|None:
        if embedding:
            return next(
                (entry['name'] for entry in self.embedding_model_metadata['schema']['inputs']),
                None
            )

        return next(
            (entry['name'] for entry in self.model_metadata['schema']['inputs']),
            None
        )

    def _get_handler_output(self, embedding: bool = False) -> str|None:
        if embedding:
            return next(
                (entry['name'] for entry in self.embedding_model_metadata['schema']['outputs'] if entry['output_purpose'] == 'embeddings'),
                None
            )

        return next(
            (entry['name'] for entry in self.model_metadata['schema']['outputs'] if entry['output_purpose'] == 'predictions'),
            None
        )

    def _load_handler(self, embedding: bool = False) -> callable:
        model = self._embedding_model if embedding else self._model
        if model is None:
            return None

        handler_class = self.model_metadata['inference']['algorithm'] \
            if not embedding else self.embedding_model_metadata['inference']['algorithm']

        handler_in = self._get_handler_input(embedding)
        handler_out = self._get_handler_output(embedding)

        if handler_in and handler_out:
            return getattr(es, handler_class)(graphFilename=model, input=handler_in, output=handler_out)
        if handler_in:
            return getattr(es, handler_class)(graphFilename=model, input=handler_in)
        if handler_out:
            return getattr(es, handler_class)(graphFilename=model, output=handler_out)

        return getattr(es, handler_class)(graphFilename=model)

    def _load_handlers(self) -> (callable, callable):
        return self._load_handler(), self._load_handler(embedding=True)

    def _load(self, model: str|None) -> (str, str):
        if not model:
            return None, None

        model_path = model
        is_path = path.isabs(model_path)

        if not is_path:
            if self._path == 'auto':
                home = Path.home() / 'essentia'
                home_base = Path(os.path.dirname(home / model_path))
                if not home_base.exists():
                    home_base.mkdir(parents=True)
                model_path = path.join(home, model_path)
            else:
                model_path = path.join(self._path, model_path)

        meta_file = f'{model}.json'
        meta_file_path = f'{model_path}.json'

        if not path.isfile(meta_file_path):
            self._log.info(f'Downloading model: {meta_file}')
            urllib.request.urlretrieve(f'https://essentia.upf.edu/models/{meta_file}', meta_file_path)

        metadata_file = open(meta_file_path, 'r')
        metadata = json.load(metadata_file)
        metadata_file.close()

        weight_file = f'{model}.pb'
        weight_file_path = f'{model_path}.pb'

        if not path.isfile(weight_file_path):
            self._log.info(f'Downloading model: {weight_file}')
            urllib.request.urlretrieve(metadata['link'], weight_file_path)

        return weight_file_path, metadata

    def embed(self, audio):
        return self._embedding_handler(audio)

    def analyse(self, embedding):
        return self._handler(embedding)

class EssentiaInterface:
    def __init__(self, config: ConfigView, log: Logger):
        self._config = config
        self._log = log

        self._bpm_model = self._load_bpm_model()
        self._mood_models = self._load_mood_models()

    def _load_bpm_model(self) -> EssentiaModel|None:
        model_folder = self._config['path'].get(str)
        if self._config['tags']['bpm']['enabled'].get(bool):
            return EssentiaModel(
                self._log,
                model_folder,
                self._config['tags']['bpm'],
                'bpm'
            )

        return None

    def _load_mood_models(self) -> [EssentiaModel]:
        model_folder = self._config['path'].get(str)
        models = []

        if self._config['tags']['mood']['enabled'].get(bool):
            for mood_name, mood in self._config['tags']['mood']['moods'].items():
                if mood['enabled'].get(bool):
                    models.append(EssentiaModel(
                        self._log,
                        model_folder,
                        mood,
                        mood_name
                    ))

        return models

    def _analyze_bpm(self, loader: es.MonoLoader, item: Item):
        if self._bpm_model is None:
            return

        loader.configure(
            filename=item.path.decode('utf-8'),
            sampleRate=self._bpm_model.model_metadata['inference']['sample_rate'],
            resampleQuality=4
        )
        audio = loader()

        global_bpm, local_bpm, local_probs = self._bpm_model.analyse(audio)
        confidence = sum(local_probs) / len(local_probs)

        if  1 - confidence <= self._config['tags']['bpm']['threshold'].get(float):
            self._log.info(f'BPM success {item.path}: {global_bpm} / {confidence:.4f}')
            item['bpm'] = global_bpm
        else:
            self._log.info(f'BPM below threshold {item.path}: {global_bpm} / {confidence:.4f}')

    def _analyze_mood(self, loader: es.MonoLoader, item: Item):
        for model in self._mood_models:
            loader.configure(
                filename=item.path.decode('utf-8'),
                sampleRate=model.model_metadata['inference']['sample_rate'],
                resampleQuality=4
            )
            audio = loader()
            embedding = model.embed(audio)
            predictions = [entry[0] for entry in model.analyse(embedding)]
            self._log.info(f'Things {predictions}')
            confidence = sum(predictions) / len(predictions)

            if  1 - confidence <= model.config['threshold'].get(float):
                moods = ';'.join([mood.get() for mood in model.config['mapping'].sequence()])
                if 'mood' in item and item['mood']:
                    item['mood'] = f'{item['mood']}{self._config['tags']['mood']['separator']}{moods}'
                else:
                    item['mood'] = moods
                self._log.info(f'Mood success {item.path}: {model.model_metadata['name']} / {item['mood']} / {confidence:.4f}')
            else:
                self._log.info(f'Mood below threshold {item.path}: {model.model_metadata['name']} / {confidence:.4f}')

    def _analyse_item(self, item: Item) -> None:
        if not path.isfile(item.path):
            self._log.error(f'Item {item.path} not found!')
            return

        loader = es.MonoLoader()

        self._analyze_bpm(loader, item)
        self._analyze_mood(loader, item)
        item.write()

    def analyse(self, items: [Item]):
        for item in items:
            self._analyse_item(item)

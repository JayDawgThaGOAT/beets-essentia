import json
import multiprocessing
import os
import os.path
import time
import urllib.error
import urllib.request
from concurrent import futures
from enum import Enum
from logging import Logger

from os import path
from pathlib import Path
from typing import Callable

import numpy as np
from beets.library import Item
from beets.ui import UserError
from confuse import ConfigView, ConfigTypeError

import essentia.standard as es


def _download(
        url: str,
        dest: str,
        label: str,
        log: Logger,
        timeout: float = 60.0,
        retries: int = 3,
) -> None:
    """Download *url* to *dest*, logging progress via the beets plugin logger."""
    label_display = path.basename(label)
    chunk_size = 64 * 1024

    def _cleanup_partial() -> None:
        if path.isfile(dest):
            try:
                os.remove(dest)
            except OSError:
                pass

    def _attempt() -> None:
        last_percent = -1
        last_mb = -1

        def _log_progress(downloaded: int, total_size: int) -> None:
            nonlocal last_percent, last_mb
            if total_size > 0:
                downloaded = min(downloaded, total_size)
                percent = downloaded * 100 // total_size
                # Log at 0%, every +10%, and on completion.
                if (
                    last_percent >= 0
                    and percent < last_percent + 10
                    and downloaded < total_size
                ):
                    return
                last_percent = percent
                done_mb = downloaded / (1024 * 1024)
                total_mb = total_size / (1024 * 1024)
                log.info(f'{label_display}  {percent:3d}%  {done_mb:.2f} / {total_mb:.2f} MB')
            else:
                done_mb_i = int(downloaded / (1024 * 1024))
                if done_mb_i <= last_mb:
                    return
                last_mb = done_mb_i
                log.info(f'{label_display}  {done_mb_i:.2f} MB')

        with urllib.request.urlopen(url, timeout=timeout) as resp:
            total_size = int(resp.headers.get('Content-Length') or 0)
            downloaded = 0
            _log_progress(downloaded, total_size)
            with open(dest, 'wb') as out:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    out.write(chunk)
                    downloaded += len(chunk)
                    _log_progress(downloaded, total_size)
            if total_size <= 0:
                log.info(
                    f'{label_display}  done  {downloaded / (1024 * 1024):.2f} MB'
                )

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        if attempt == 1:
            log.info(f'Connecting: {label_display}')
        else:
            log.info(f'Retrying ({attempt}/{retries}): {label_display}')

        try:
            _attempt()
            return
        except (TimeoutError, urllib.error.URLError, OSError) as exc:
            last_error = exc
            _cleanup_partial()
            if attempt < retries:
                # Brief backoff; essentia.upf.edu TLS handshakes are flaky.
                time.sleep(2 * attempt)
                continue
            break

    raise UserError(
        f'Failed to download model {label_display} after {retries} attempts: {last_error}'
    )


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

    def _load_handler(self, embedding: bool = False) -> Callable|None:
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

    def _load_handlers(self) -> tuple[Callable, Callable|None]:
        return self._load_handler(), self._load_handler(embedding=True)

    def _load(self, model: str|None) -> tuple[str|None, dict|None]:
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
            _download(
                f'https://essentia.upf.edu/models/{meta_file}',
                meta_file_path,
                meta_file,
                self._log,
            )

        metadata_file = open(meta_file_path, 'r')
        metadata = json.load(metadata_file)
        metadata_file.close()

        weight_file = f'{model}.pb'
        weight_file_path = f'{model_path}.pb'

        if not path.isfile(weight_file_path):
            self._log.info(f'Downloading model: {weight_file}')
            _download(metadata['link'], weight_file_path, weight_file, self._log)

        return weight_file_path, metadata

    def sample_rate(self) -> int:
        return self.model_metadata['inference']['sample_rate']

    def embed(self, audio):
        return self._embedding_handler(audio)

    def analyze(self, embedding):
        return self._handler(embedding)

class BPMModel(EssentiaModel):
    def __init__(
            self,
            log: Logger,
            mpath: str,
            config: ConfigView,
            name: str
    ):
        super().__init__(log, mpath, config, name)

    def analyze(self, embedding) -> tuple[float, list[float], list[float]]:
        return super().analyze(embedding)


class MoodModelType(Enum):
    aggressive = 'aggressive'
    happy = 'happy'
    party = 'party'
    relaxed = 'relaxed'
    sad = 'sad'
    mirex = 'mirex'
    jamendo = 'jamendo'

class MoodPrediction:
    def __init__(self, moods: set[str], confidence: float, mtype: MoodModelType):
        self.confidence = confidence
        self.moods = moods
        self.name = self._get_name(mtype)

    def _get_name(self, mtype: MoodModelType) -> str:
        if mtype == MoodModelType.mirex or mtype == MoodModelType.jamendo:
            return f'{mtype.name}+{next(iter(self.moods))}'

        return mtype.name

    def __repr__(self) -> str:
        return f'{self.moods} / {self.confidence}'

    def __str__(self) -> str:
        return f'{self.moods} / {self.confidence}'

class MoodModel(EssentiaModel):
    def __init__(
            self,
            log: Logger,
            mpath: str,
            config: ConfigView,
            name: str
    ):
        super().__init__(log, mpath, config, name)
        self.model_type = MoodModelType(self.name)
        self._moods = self._get_moods()

    def _get_moods(self) -> list[list[str|None]] | set[str]:
        if self.model_type == MoodModelType.mirex or self.model_type == MoodModelType.jamendo:
            moods = []
            mappings = [mood for category, mood in sorted(self.config['mapping'].items(), key=lambda t: t[0])]
            for mapping in mappings:
                try:
                    moods.append([m.get() for m in mapping.sequence()])
                except ConfigTypeError:
                    moods.append(None)
            return moods

        return set(mood.get() for mood in self.config['mapping'].sequence())

    def analyze(self, embedding) -> list[MoodPrediction]:
        predictions = super().analyze(embedding)

        if self.model_type == MoodModelType.mirex or self.model_type == MoodModelType.jamendo:
            sum_count = len(predictions)
            sum_categories = np.array(predictions).sum(axis=0)
            category_confidence = [i / sum_count for i in sum_categories]
            return [MoodPrediction(set(moods), category_confidence[idx], self.model_type) for idx, moods in enumerate(self._moods) if moods is not None]

        predictions_positive = [entry[0] for entry in predictions]
        confidence = sum(predictions_positive) / len(predictions_positive)
        return [MoodPrediction(self._moods, confidence, self.model_type)]


class EssentiaInterface:
    def __init__(self, config: ConfigView, log: Logger):
        self._config = config
        self._logger = log
        self._max_threads = config['threads'].get() \
            if config['threads'].get() != 'auto' else multiprocessing.cpu_count()
        self._dry_run: bool = config['dry-run'].get(bool)
        self._quiet: bool = config['quiet'].get(bool)
        self._force: bool = config['force'].get(bool)
        self._write: bool = config['write'].get(bool)

        self._bpm_model = self._load_bpm_model()
        self._mood_models = self._load_mood_models()

    def _log(self, msg: str):
        if not self._quiet:
            self._logger.info(msg)

    def _load_bpm_model(self) -> BPMModel|None:
        model_folder = self._config['path'].get(str)
        if self._config['tags']['bpm']['enabled'].get(bool):
            return BPMModel(
                self._logger,
                model_folder,
                self._config['tags']['bpm'],
                'bpm'
            )

        return None

    def _load_mood_models(self) -> list[MoodModel]:
        model_folder = self._config['path'].get(str)
        models = []

        if self._config['tags']['mood']['enabled'].get(bool):
            for mood_name, mood in self._config['tags']['mood']['moods'].items():
                if mood['enabled'].get(bool):
                    models.append(MoodModel(
                        self._logger,
                        model_folder,
                        mood,
                        mood_name
                    ))

        return models

    def _analyze_bpm(self, loader: es.MonoLoader, item: Item) -> None:
        if self._bpm_model is None:
            return

        loader.configure(
            filename=item.path.decode('utf-8'),
            sampleRate=self._bpm_model.sample_rate(),
            resampleQuality=4
        )
        audio = loader()

        global_bpm, local_bpm, local_probs = self._bpm_model.analyze(audio)
        confidence = sum(local_probs) / len(local_probs)

        if  1 - confidence <= self._config['tags']['bpm']['threshold'].get(float):
            if 'bpm' in item and item['bpm']:
                if self._force:
                    self._log(f'[BPM][OVERWRITE][{item.path.decode('utf-8')}] {item['bpm']} -> {global_bpm} / {confidence:.4f}')
                    item['bpm'] = global_bpm
                else:
                    self._log(f'[BPM][SKIP][{item.path.decode('utf-8')}] {item['bpm']} -> {global_bpm} / {confidence:.4f}')
            else:
                self._log(f'[BPM][ADD][{item.path.decode('utf-8')}] {global_bpm} / {confidence:.4f}')
                item['bpm'] = global_bpm
        else:
            self._log(f'[BPM][THRESHOLD][{item.path.decode('utf-8')}] {global_bpm} / {confidence:.4f}')

    def _analyze_mood(self, loader: es.MonoLoader, item: Item) -> None:
        if self._force and self._config['tags']['mood']['force_overwrite'].get(bool):
            self._log(f'[Mood][OVERWRITE][{item.path.decode('utf-8')}] {item['mood']}')
            item['mood'] = None

        separator = self._config['tags']['mood']['separator'].get(str)

        for model in self._mood_models:
            loader.configure(
                filename=item.path.decode('utf-8'),
                sampleRate=model.sample_rate(),
                resampleQuality=4
            )
            audio = loader()
            embedding = model.embed(audio)
            predictions = model.analyze(embedding)

            for p in predictions:
                if len(p.moods) == 0:
                    continue

                if  1 - p.confidence <= model.config['threshold'].get(float):
                    if 'mood' in item and item['mood']:
                        existing_list = item['mood'].split(separator)
                        item_moods_list = sorted(set(existing_list) | p.moods)
                        item_moods = separator.join(item_moods_list)
                        if len(item_moods_list) == len(existing_list):
                            self._log(f'[Mood][SKIP][{item.path.decode('utf-8')}] {p.name} / {item['mood']} == {item_moods} {p.confidence:.4f}')
                        else:
                            self._log(f'[Mood][APPEND][{item.path.decode('utf-8')}] {p.name} / {item['mood']} +> {item_moods} {p.confidence:.4f}')
                            item['mood'] = item_moods
                    else:
                        item_moods = separator.join(sorted(p.moods))
                        self._log(f'[Mood][ADD][{item.path.decode('utf-8')}] +> {item_moods} / {p.confidence:.4f}')
                        item['mood'] = item_moods
                else:
                    self._log(f'[Mood][THRESHOLD][{item.path.decode('utf-8')}] {p.name} / {p.confidence:.4f}')

    def _analyse_item(self, item: Item) -> None:
        if not path.isfile(item.path):
            self._logger.error(f'[{item.path.decode('utf-8')}] not found!')
            return

        loader = es.MonoLoader()

        self._analyze_bpm(loader, item)
        self._analyze_mood(loader, item)

        if self._write and not self._dry_run:
            success = item.try_write()
            if success:
                self._log(f'[{item.path.decode('utf-8')}] tags written successfully')
            else:
                self._logger.error(f'[{item.path.decode('utf-8')}] failed to write tags')
        elif self._write and self._dry_run:
            self._log(f'[{item.path.decode('utf-8')}] would write tags')

    def analyse(self, items: [Item]) -> None:
        with futures.ThreadPoolExecutor(max_workers=self._max_threads) as executor:
            executor.map(self._analyse_item, items)

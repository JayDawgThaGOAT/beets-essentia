import http.client
import json
import multiprocessing
import os
import os.path
import threading
import time
import urllib.error
from concurrent import futures
from enum import Enum
from logging import DEBUG, Logger
from os import path
from pathlib import Path
from typing import Callable
from urllib.parse import urljoin, urlparse

import numpy as np
from beets.library import Item
from beets.ui import UserError
from confuse import ConfigView, ConfigTypeError

import essentia.standard as es

MODELS_BASE_URL = 'https://essentia.upf.edu/models/'


class _HttpsSession:
    """Keep-alive HTTPS client; reconnects after errors (UPF TLS is flaky)."""

    def __init__(self, timeout: float = 60.0):
        self._timeout = timeout
        self._conn: http.client.HTTPSConnection | None = None
        self._host: str | None = None

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except OSError:
                pass
            self._conn = None
            self._host = None

    def _ensure_conn(self, host: str) -> http.client.HTTPSConnection:
        if self._conn is None or self._host != host:
            self.close()
            self._conn = http.client.HTTPSConnection(host, timeout=self._timeout)
            self._host = host
        return self._conn

    def get(self, url: str) -> http.client.HTTPResponse:
        """GET *url* and return an open response (caller must read the body)."""
        current = url
        for _ in range(6):
            parsed = urlparse(current)
            if parsed.scheme != 'https':
                raise urllib.error.URLError(f'Only HTTPS is supported: {current}')

            host = parsed.netloc
            req_path = parsed.path or '/'
            if parsed.query:
                req_path = f'{req_path}?{parsed.query}'

            conn = self._ensure_conn(host)
            try:
                conn.request('GET', req_path, headers={'Connection': 'keep-alive'})
                resp = conn.getresponse()
            except (TimeoutError, OSError, http.client.HTTPException):
                self.close()
                raise

            if resp.status in (301, 302, 303, 307, 308):
                location = resp.getheader('Location')
                resp.read()
                if not location:
                    raise urllib.error.URLError(f'Redirect without Location for {current}')
                current = urljoin(current, location)
                # Host may change; drop pooled connection.
                if urlparse(current).netloc != host:
                    self.close()
                continue

            if resp.status != 200:
                resp.read()
                raise urllib.error.URLError(
                    f'HTTP {resp.status} {resp.reason} for {current}'
                )

            return resp

        raise urllib.error.URLError(f'Too many redirects for {url}')


def _download(
        url: str,
        dest: str,
        label: str,
        log: Logger,
        session: _HttpsSession | None = None,
        timeout: float = 60.0,
        retries: int = 3,
) -> None:
    """Download *url* to *dest*, logging progress via the beets plugin logger."""
    label_display = path.basename(label)
    chunk_size = 64 * 1024
    partial = f'{dest}.partial'
    owns_session = session is None
    if session is None:
        session = _HttpsSession(timeout=timeout)

    def _cleanup_partial() -> None:
        for candidate in (partial, dest):
            if path.isfile(candidate):
                try:
                    os.remove(candidate)
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

        resp = session.get(url)
        try:
            total_size = int(resp.getheader('Content-Length') or 0)
            downloaded = 0
            _log_progress(downloaded, total_size)
            Path(os.path.dirname(dest) or '.').mkdir(parents=True, exist_ok=True)
            with open(partial, 'wb') as out:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    out.write(chunk)
                    downloaded += len(chunk)
                    _log_progress(downloaded, total_size)
            os.replace(partial, dest)
            if total_size <= 0:
                log.info(
                    f'{label_display}  done  {downloaded / (1024 * 1024):.2f} MB'
                )
        finally:
            # Ensure the body is drained/closed so keep-alive can reuse the socket.
            try:
                resp.close()
            except OSError:
                pass

    last_error: Exception | None = None
    try:
        for attempt in range(1, retries + 1):
            if attempt == 1:
                log.info(f'Connecting: {label_display}')
            else:
                log.info(
                    f'Retrying ({attempt}/{retries}): {label_display} ({last_error})'
                )

            try:
                _attempt()
                return
            except (TimeoutError, urllib.error.URLError, OSError, http.client.HTTPException) as exc:
                last_error = exc
                session.close()
                _cleanup_partial()
                if attempt < retries:
                    # Brief backoff; essentia.upf.edu TLS handshakes are flaky.
                    time.sleep(1.5 * attempt)
                    continue
                break

        raise UserError(
            f'Failed to download model {label_display} after {retries} attempts: {last_error}'
        )
    finally:
        if owns_session:
            session.close()


class ModelRepository:
    """Resolve, download, and cache Essentia TF models for a library path."""

    def __init__(self, log: Logger, mpath: str):
        self._log = log
        self._path = mpath
        self._session = _HttpsSession()
        self._metadata: dict[str, dict] = {}
        self._weight_paths: dict[str, str] = {}
        self._handlers: dict[tuple, Callable] = {}

    def close(self) -> None:
        self._session.close()

    def _resolve(self, model: str) -> tuple[bool, str, str, str]:
        """Return (is_relative_id, model_id, meta_path, weight_path)."""
        if path.isabs(model):
            return False, model, f'{model}.json', f'{model}.pb'

        if self._path == 'auto':
            home = Path.home() / 'essentia'
            model_path = path.join(home, model)
        else:
            model_path = path.join(self._path, model)

        return True, model, f'{model_path}.json', f'{model_path}.pb'

    @staticmethod
    def embedding_model_id(metadata: dict | None) -> str | None:
        if not metadata:
            return None
        inference = metadata.get('inference') or {}
        embedding = inference.get('embedding_model')
        if not embedding:
            return None
        link = embedding.get('link', '')
        return link.replace(MODELS_BASE_URL, '').replace('.pb', '') or None

    def ensure(self, model: str | None) -> tuple[str | None, dict | None]:
        """Ensure model files exist on disk; return (weight_path, metadata)."""
        if not model:
            return None, None

        if model in self._weight_paths and model in self._metadata:
            return self._weight_paths[model], self._metadata[model]

        is_relative, model_id, meta_path, weight_path = self._resolve(model)
        Path(os.path.dirname(meta_path) or '.').mkdir(parents=True, exist_ok=True)

        if not path.isfile(meta_path):
            if not is_relative:
                raise UserError(f'Model metadata not found: {meta_path}')
            self._log.info(f'Downloading model: {model_id}.json')
            _download(
                f'{MODELS_BASE_URL}{model_id}.json',
                meta_path,
                f'{model_id}.json',
                self._log,
                session=self._session,
            )

        with open(meta_path, 'r') as metadata_file:
            metadata = json.load(metadata_file)

        if not path.isfile(weight_path):
            if not is_relative and not metadata.get('link'):
                raise UserError(f'Model weights not found: {weight_path}')
            weight_url = metadata.get('link') or f'{MODELS_BASE_URL}{model_id}.pb'
            self._log.info(f'Downloading model: {model_id}.pb')
            _download(
                weight_url,
                weight_path,
                f'{model_id}.pb',
                self._log,
                session=self._session,
            )

        self._metadata[model] = metadata
        self._weight_paths[model] = weight_path
        # Also cache under resolved id when they differ.
        self._metadata[model_id] = metadata
        self._weight_paths[model_id] = weight_path
        return weight_path, metadata

    def prefetch(self, models: list[str | None]) -> None:
        """Download all configured models (and embedding deps) before TF load."""
        pending: list[str] = []
        seen: set[str] = set()
        for model in models:
            if model and model not in seen:
                pending.append(model)
                seen.add(model)

        if not pending:
            return

        self._log.info(f'Ensuring {len(pending)} model(s) are available…')
        idx = 0
        while idx < len(pending):
            model = pending[idx]
            idx += 1
            _, metadata = self.ensure(model)
            embedding_id = self.embedding_model_id(metadata)
            if embedding_id and embedding_id not in seen:
                pending.append(embedding_id)
                seen.add(embedding_id)

        # Drop the HTTP session before long-lived TF graphs hold memory.
        self.close()

    def get_handler(
            self,
            weight_path: str,
            metadata: dict,
            *,
            embedding: bool = False,
    ) -> Callable:
        inputs = metadata['schema']['inputs']
        outputs = metadata['schema']['outputs']
        handler_class = metadata['inference']['algorithm']
        handler_in = next((entry['name'] for entry in inputs), None)
        if embedding:
            handler_out = next(
                (
                    entry['name'] for entry in outputs
                    if entry.get('output_purpose') == 'embeddings'
                ),
                None,
            )
        else:
            handler_out = next(
                (
                    entry['name'] for entry in outputs
                    if entry.get('output_purpose') == 'predictions'
                ),
                None,
            )

        cache_key = (weight_path, handler_class, handler_in, handler_out, embedding)
        if cache_key in self._handlers:
            return self._handlers[cache_key]

        self._log.info(f'Loading model: {path.basename(weight_path)}')
        algo = getattr(es, handler_class)
        if handler_in and handler_out:
            handler = algo(graphFilename=weight_path, input=handler_in, output=handler_out)
        elif handler_in:
            handler = algo(graphFilename=weight_path, input=handler_in)
        elif handler_out:
            handler = algo(graphFilename=weight_path, output=handler_out)
        else:
            handler = algo(graphFilename=weight_path)

        self._handlers[cache_key] = handler
        return handler


class EssentiaModel:
    def __init__(
            self,
            log: Logger,
            mpath: str,
            config: ConfigView,
            name: str,
            repo: ModelRepository | None = None,
    ):
        self.name = name
        self.config = config
        self._log = log
        self._path = mpath
        self._repo = repo or ModelRepository(log, mpath)

        model_weight_file, model_metadata = self._repo.ensure(self.config['model'].get(str))
        self._model = model_weight_file
        self.model_metadata = model_metadata

        embedding_model = self._repo.embedding_model_id(self.model_metadata)
        model_weight_file, model_metadata = self._repo.ensure(embedding_model)
        self._embedding_model = model_weight_file
        self.embedding_model_metadata = model_metadata

        self._handler = self._repo.get_handler(
            self._model, self.model_metadata, embedding=False
        )
        self._embedding_handler = (
            self._repo.get_handler(
                self._embedding_model, self.embedding_model_metadata, embedding=True
            )
            if self._embedding_model and self.embedding_model_metadata
            else None
        )

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
            name: str,
            repo: ModelRepository | None = None,
    ):
        super().__init__(log, mpath, config, name, repo=repo)

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
            name: str,
            repo: ModelRepository | None = None,
    ):
        super().__init__(log, mpath, config, name, repo=repo)
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

        model_folder = self._config['path'].get(str)
        self._repo = ModelRepository(self._logger, model_folder)
        self._repo.prefetch(self._configured_model_ids())

        self._bpm_model = self._load_bpm_model()
        self._mood_models = self._load_mood_models()
        self._store_lock = threading.Lock()

    def _log(self, msg: str):
        if not self._quiet:
            self._logger.info(msg)

    @staticmethod
    def _item_label(item: Item) -> str:
        title = item.get('title')
        if title:
            return str(title)
        return path.splitext(path.basename(item.path.decode('utf-8')))[0]

    def _should_log_rejects(self) -> bool:
        if self._quiet:
            return False
        if self._config['tags']['mood']['log_rejects'].get(bool):
            return True
        return self._logger.isEnabledFor(DEBUG)

    def _configured_model_ids(self) -> list[str]:
        ids: list[str] = []
        if self._config['tags']['bpm']['enabled'].get(bool):
            ids.append(self._config['tags']['bpm']['model'].get(str))
        if self._config['tags']['mood']['enabled'].get(bool):
            for _mood_name, mood in self._config['tags']['mood']['moods'].items():
                if mood['enabled'].get(bool):
                    ids.append(mood['model'].get(str))
        return ids

    def _load_bpm_model(self) -> BPMModel|None:
        model_folder = self._config['path'].get(str)
        if self._config['tags']['bpm']['enabled'].get(bool):
            return BPMModel(
                self._logger,
                model_folder,
                self._config['tags']['bpm'],
                'bpm',
                repo=self._repo,
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
                        mood_name,
                        repo=self._repo,
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
        if not self._mood_models:
            return

        if self._force and self._config['tags']['mood']['force_overwrite'].get(bool):
            self._log(f'[Mood][OVERWRITE][{self._item_label(item)}] {item.get('mood')}')
            item['mood'] = None

        separator = self._config['tags']['mood']['separator'].get(str)
        # Reuse embeddings when multiple heads share the same embedder + sample rate.
        embedding_cache: dict[tuple[str | None, int], object] = {}
        accepted: dict[str, float] = {}
        rejected: list[tuple[str, float]] = []

        for model in self._mood_models:
            emb_key = (model._embedding_model, model.sample_rate())
            if emb_key not in embedding_cache:
                loader.configure(
                    filename=item.path.decode('utf-8'),
                    sampleRate=model.sample_rate(),
                    resampleQuality=4
                )
                audio = loader()
                embedding_cache[emb_key] = model.embed(audio)

            predictions = model.analyze(embedding_cache[emb_key])

            for p in predictions:
                if len(p.moods) == 0:
                    continue

                label = separator.join(sorted(p.moods))
                if 1 - p.confidence <= model.config['threshold'].get(float):
                    for mood in p.moods:
                        accepted[mood] = max(accepted.get(mood, 0.0), p.confidence)

                    if 'mood' in item and item['mood']:
                        existing_list = item['mood'].split(separator)
                        item_moods_list = sorted(set(existing_list) | p.moods)
                        item['mood'] = separator.join(item_moods_list)
                    else:
                        item['mood'] = separator.join(sorted(p.moods))
                else:
                    rejected.append((label, p.confidence))

        track = self._item_label(item)
        if accepted:
            score_parts = ' '.join(
                f'{mood}={accepted[mood]:.4f}' for mood in sorted(accepted)
            )
            final = item.get('mood') or '_(none)_'
            self._log(f'[Mood][WRITE][{track}] {score_parts} → {final}')
        else:
            self._log(f'[Mood][WRITE][{track}] _(none)_')

        if self._should_log_rejects() and rejected:
            reject_count = self._config['tags']['mood']['reject_count'].get(int)
            rejected.sort(key=lambda entry: entry[1], reverse=True)
            reject_parts = ' '.join(
                f'{label}={confidence:.4f}'
                for label, confidence in rejected[:reject_count]
            )
            self._log(f'[Mood][REJECT][{track}] {reject_parts}')

    def _persist_item(self, item: Item) -> None:
        """Write file tags and/or store library fields after analysis."""
        decoded_path = item.path.decode('utf-8')

        if self._dry_run:
            # Mood/BPM status lines already describe the dry-run result.
            return

        if self._write:
            success = item.try_write()
            if success:
                mood = item.get('mood')
                if mood:
                    self._log(f'[{decoded_path}] wrote mood={mood}')
                else:
                    self._log(f'[{decoded_path}] tags written successfully')
            else:
                self._logger.error(f'[{decoded_path}] failed to write tags')

        # Persist to the beets DB so skip queries see updated bpm/mood.
        if getattr(item, '_db', None) is not None and item.id is not None:
            with self._store_lock:
                item.store()

    def _analyse_item(self, item: Item) -> None:
        if not path.isfile(item.path):
            self._logger.error(f'[{item.path.decode('utf-8')}] not found!')
            return

        loader = es.MonoLoader()

        self._analyze_bpm(loader, item)
        self._analyze_mood(loader, item)
        self._persist_item(item)

    def analyse(self, items: list[Item]) -> None:
        with futures.ThreadPoolExecutor(max_workers=self._max_threads) as executor:
            # Exhaust the iterator so work completes before the pool shuts down.
            list(executor.map(self._analyse_item, items))

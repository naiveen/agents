# Copyright 2023 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import asyncio
import base64
import json
import os
import weakref
from dataclasses import dataclass
from typing import Final

import aiohttp

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tts,
    utils,
)
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    NotGivenOr,
)
from livekit.agents.utils import is_given

from .log import logger
from .models import LMNTAudioFormats, LMNTLanguages, LMNTModels, LMNTSampleRate

LMNT_BASE_URL: Final[str] = "https://api.lmnt.com/v1/ai/speech/bytes"
LMNT_STREAM_URL: Final[str] = "https://api.lmnt.com/v1/ai/speech/stream"
NUM_CHANNELS: Final[int] = 1

def ws_url(path: str) -> str:
    return f"{LMNT_STREAM_URL}{path}"

@dataclass
class _TTSOptions:
    sample_rate: LMNTSampleRate
    model: LMNTModels
    format: LMNTAudioFormats
    language: LMNTLanguages
    num_channels: int
    voice: str
    api_key: str


class TTS(tts.TTS):
    """
    Text-to-Speech (TTS) plugin for LMNT.
    """
    def __init__(
        self,
        *,
        model: LMNTModels = "blizzard",
        voice: str = "lily",
        language: NotGivenOr[LMNTLanguages] = NOT_GIVEN,
        format: LMNTAudioFormats = "mp3",
        sample_rate: LMNTSampleRate = 24000,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """
        Create a new instance of LMNT TTS.

        See (https://docs.lmnt.com/api-reference/speech/synthesize-speech-bytes) for more details on the API.

        Args:
            model: The model to use for synthesis. Default is "blizzard". Learn more at (https://docs.lmnt.com/guides/models).
            voice: The voice id of the voice to use; Default is "ava".
            language: The desired language. Two letter ISO 639-1 code. Defaults to auto language detection. See (https://docs.lmnt.com/api-reference/speech/synthesize-speech-bytes#body-language) for the full list of supported languages.
            format: The file format of the audio output. Available options: aac, mp3, mulaw, raw, wav. Default is "mp3".
            sample_rate: The desired output sample rate in Hz. Defaults to 24000. See (https://docs.lmnt.com/api-reference/speech/synthesize-speech-bytes#body-sample-rate) for more details.
            api_key: The API key to use for authentication. If not provided, api key is read from the environment variable defined at `LMNT_API_KEY`.
            http_session: An existing aiohttp ClientSession to use. If not provided, a new session will be created.
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=sample_rate,
            num_channels=NUM_CHANNELS,
        )
        api_key = api_key if is_given(api_key) else os.getenv("LMNT_API_KEY")
        if not is_given(api_key):
            raise ValueError("LMNT API key is required. " \
            "Set it via environment variable or pass it as an argument.")

        if not is_given(language):
            language = 'auto' if model == "blizzard" else "en"

        self._opts = _TTSOptions(
            model = model,
            sample_rate = sample_rate,
            num_channels = NUM_CHANNELS,
            language = language,
            voice = voice,
            format = format,
            api_key = api_key
        )

        self._session = http_session
        self._pool = utils.ConnectionPool[aiohttp.ClientWebSocketResponse](
            connect_cb=self._connect_ws,
            close_cb=self._close_ws,
            max_session_duration=90,
            mark_refreshed_on_get=True,
        )
        self._streams = weakref.WeakSet[SynthesizeStream]()

    async def _connect_ws(self) -> aiohttp.ClientWebSocketResponse:
        session = self._ensure_session()
        return await asyncio.wait_for(session.ws_connect(LMNT_STREAM_URL), self._conn_options.timeout)

    async def _close_ws(self, ws: aiohttp.ClientWebSocketResponse):
        await ws.close()

    def synthesize(
            self,
            text: str,
            *,
            conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> ChunkedStream:
        return ChunkedStream(
            tts=self,
            input_text=text,
            conn_options=conn_options,
            opts=self._opts,
            session=self._ensure_session(),
        )

    def update_options(self,
                       *,
            model: NotGivenOr[LMNTModels] = NOT_GIVEN,
            voice: NotGivenOr[str] = NOT_GIVEN,
            language: NotGivenOr[LMNTLanguages] = NOT_GIVEN,
            format: NotGivenOr[LMNTAudioFormats] = NOT_GIVEN,
            sample_rate: NotGivenOr[LMNTSampleRate] = NOT_GIVEN,
    ) -> None:
        """
        Update the TTS options.
        Args:
            model: The model to use for synthesis. Default is "blizzard". Learn more at (https://docs.lmnt.com/guides/models).
            voice: The voice id of the voice to use; Default is "lily".
            language: The desired language. Two letter ISO 639-1 code. Defaults to auto language detection. See (https://docs.lmnt.com/api-reference/speech/synthesize-speech-bytes#body-language) for the full list of supported languages.
            format: The file format of the audio output. Available options: aac, mp3, mulaw, raw, wav. Default is "mp3".
            sample_rate: The desired output sample rate in Hz. Defaults to 24000. 
        """
        if is_given(model):
            self._opts.model = model
        if is_given(voice):
            self._opts.voice = voice
        if is_given(language):
            self._opts.language = language
        if is_given(format):
            self._opts.format = format
        if is_given(sample_rate):
            self._opts.sample_rate = sample_rate

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()

        return self._session

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> SynthesizeStream:
        stream = SynthesizeStream(
            tts=self,
            opts=self._opts,
            pool=self._pool,
        )
        self._streams.add(stream)
        return stream

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()
        await super().aclose()


class ChunkedStream(tts.ChunkedStream):
    """Synthesize text to speech in chunks."""

    def __init__(
        self,
        *,
        tts: TTS,
        input_text: str,
        opts: _TTSOptions,
        session: aiohttp.ClientSession,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._opts, self._session = opts, session

    async def _run(self) -> None:
        request_id = utils.shortuuid()
        audio_bytestream = utils.audio.AudioByteStream(
            sample_rate=self._opts.sample_rate,
            num_channels=self._opts.num_channels,
        )

        headers = {
            'Content-Type': 'application/json',
            'X-API-Key': self._opts.api_key,
        }
        data = {
            'text': self._input_text,
            'voice': self._opts.voice,
            'language': self._opts.language,
            'sample_rate': self._opts.sample_rate,
            'model': self._opts.model,
            'format': self._opts.format,
        }

        try:
            async with self._session.post(
                LMNT_BASE_URL,
                headers=headers,
                json=data,
                timeout=aiohttp.ClientTimeout(
                    total=30,
                    sock_connect=self._conn_options.timeout,
                ),
            ) as resp:
                resp.raise_for_status()
                emitter = tts.SynthesizedAudioEmitter(
                    event_ch=self._event_ch,
                    request_id=request_id,
                )
                async for data, _ in resp.content.iter_chunks():
                    for frame in audio_bytestream.write(data):
                        emitter.push(frame)

                for frame in audio_bytestream.flush():
                    emitter.push(frame)
                emitter.flush()
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message,
                status_code=e.status,
                request_id=None,
                body=None,
            ) from None
        except Exception as e:
            raise APIConnectionError() from e


class SynthesizeStream(tts.SynthesizeStream):
    """Synthesize text to speech in a stream using websockets"""

    def __init__(
        self,
        *,
        tts: TTS,
        opts: _TTSOptions,
        pool: utils.ConnectionPool[aiohttp.ClientWebSocketResponse],
    ) -> None:
        super().__init__(tts=tts)
        self._opts, self._pool = opts, pool

    async def _run(self) -> None:
        request_id = utils.shortuuid()

        async def _input_task(ws: aiohttp.ClientWebSocketResponse):
            async for text in self._input_ch:
                if isinstance(text, self._FlushSentinel):
                    await ws.send_str('{"flush": true}')
                    continue

                await ws.send_str(json.dumps({"text": text}))
            await ws.send_str('{"eof": true}')

        async def _recv_task(ws: aiohttp.ClientWebSocketResponse):
            audio_bstream = utils.audio.AudioByteStream(
                sample_rate=self._opts.sample_rate,
                num_channels=NUM_CHANNELS,
            )
            emitter = tts.SynthesizedAudioEmitter(
                event_ch=self._event_ch,
                request_id=request_id,
            )

            while True:
                try:
                    msg = await ws.receive()
                except Exception as e:
                    raise APIStatusError(
                        "LMNT connection closed unexpectedly",
                        request_id=request_id,
                    ) from e

                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    break
                else:
                    logger.warning("Received LMNT message: %s", msg)

                if msg.type != aiohttp.WSMsgType.TEXT:
                    logger.warning("Unexpected LMNT message type %s", msg.type)
                    continue

                data = json.loads(msg.data)

                if data.get("data"):
                    b64data = base64.b64decode(data["data"])
                    for frame in audio_bstream.write(b64data):
                        emitter.push(frame)
                else:
                    logger.error("Unexpected LMNT message %s", data)

        async with self._pool.connection() as ws:
            init_msg = {
                'X-API-Key': self._opts.api_key,
                'voice': self._opts.voice,
                'format': self._opts.format,
                'language': 'en' if self._opts.language == 'auto' else self._opts.language,
                'sample_rate': self._opts.sample_rate,
                'model': self._opts.model,
            }
            await ws.send_str(json.dumps(init_msg))

            tasks = [
                asyncio.create_task(_input_task(ws)),
                asyncio.create_task(_recv_task(ws)),
            ]
            try:
                await asyncio.gather(*tasks)
            except asyncio.TimeoutError as e:
                raise APITimeoutError() from e
            except aiohttp.ClientResponseError as e:
                raise APIStatusError(
                    message=e.message,
                    status_code=e.status,
                    request_id=request_id,
                    body=None,
                ) from e
            except Exception as e:
                raise APIConnectionError() from e
            finally:
                await utils.aio.gracefully_cancel(*tasks)


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
import os
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

from .models import LMNTAudioFormats, LMNTLanguages, LMNTModels, LMNTSampleRate

LMNT_BASE_URL: Final[str] = "https://api.lmnt.com/v1/ai/speech/bytes"
NUM_CHANNELS: Final[int] = 1


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
        voice: str = "zeke",
        language: NotGivenOr[LMNTLanguages] = NOT_GIVEN,
        format: LMNTAudioFormats = "mp3",
        sample_rate: LMNTSampleRate = 24000,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """
        Create a new instance of LMNT TTS.

        See: https://docs.lmnt.com/api-reference/speech/synthesize-speech-bytes

        Args:
            model: The model to use for synthesis. Default is "blizzard".
                Learn more at: https://docs.lmnt.com/guides/models
            voice: The voice ID to use. Default is "zeke". Find more amazing voices at https://app.lmnt.com/
            language: Two-letter ISO 639-1 language code. Defaults to auto-detection.
                See: https://docs.lmnt.com/api-reference/speech/synthesize-speech-bytes#body-language
            format: Output file format. Options: aac, mp3, mulaw, raw, wav. Default is "mp3".
            sample_rate: Output sample rate in Hz. Default is 24000.
                See: https://docs.lmnt.com/api-reference/speech/synthesize-speech-bytes#body-sample-rate
            api_key: API key for authentication. Defaults to the LMNT_API_KEY environment variable.
            http_session: Optional aiohttp ClientSession. A new session is created if not provided.
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=sample_rate,
            num_channels=NUM_CHANNELS,
        )
        api_key = api_key if is_given(api_key) else os.getenv("LMNT_API_KEY")
        if not is_given(api_key):
            raise ValueError(
                "LMNT API key is required. "
                "Set it via environment variable or pass it as an argument."
            )

        if not is_given(language):
            language = "auto" if model == "blizzard" else "en"

        self._opts = _TTSOptions(
            model=model,
            sample_rate=sample_rate,
            num_channels=NUM_CHANNELS,
            language=language,
            voice=voice,
            format=format,
            api_key=api_key,
        )

        self._session = http_session

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

    def update_options(
        self,
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
            model: The model to use for synthesis. Learn more at: https://docs.lmnt.com/guides/models
            voice: The voice ID to update.
            language: Two-letter ISO 639-1 code.
                See: https://docs.lmnt.com/api-reference/speech/synthesize-speech-bytes#body-language
            format: Audio output format. Options: aac, mp3, mulaw, raw, wav.
            sample_rate: Output sample rate in Hz.
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
            "Content-Type": "application/json",
            "X-API-Key": self._opts.api_key,
        }
        data = {
            "text": self._input_text,
            "voice": self._opts.voice,
            "language": self._opts.language,
            "sample_rate": self._opts.sample_rate,
            "model": self._opts.model,
            "format": self._opts.format,
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

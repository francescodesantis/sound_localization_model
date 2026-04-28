import brian2 as b2
import brian2hears as b2h
import numpy as np


DEFAULT_SOUND_DURATION = 25 * b2.ms
DEFAULT_SILENCE_DURATION = DEFAULT_BURST_SINGLE_DURATION = 25 * b2.ms
DEFAULT_BURST_REP = 3
DEFAULT_CLICKS_NUMBER = 10
DEFAULT_CLICKS_DURATION = 10
DEFAULT_CLICKS_INTERVAL = 50
DEFAULT_SEED = 42


def gate_and_append_silence(
    sound: b2h.Sound,
    ramp_ms: float = 10.0,
    offset_silence_duration=0 * b2.ms,
):
    fs = int(sound.samplerate)
    n_samples = sound.nsamples
    ramp_samples = int((ramp_ms / 1000.0) * fs)

    if ramp_samples > 1:
        ramp = 0.5 * (1 - np.cos(np.pi * np.arange(ramp_samples) / ramp_samples))
        envelope = np.ones(n_samples)
        envelope[:ramp_samples] = ramp
        envelope[-ramp_samples:] = ramp[::-1]

        data = sound.data * envelope.reshape(-1, 1)
        sound = b2h.Sound(data, samplerate=fs * b2.Hz)

    # Append silence if requested
    if offset_silence_duration > 0 * b2.ms:
        silence = b2h.Sound.silence(offset_silence_duration)
        sound = b2h.Sound.sequence([sound, silence])

    return sound



class Tone:
    frequency: b2.Quantity
    sound: b2h.Sound

    def __init__(
        self,
        frequency: b2.Quantity,
        duration=DEFAULT_SOUND_DURATION,
        level=None,
        ramp_ms: float = 10.0,
        offset_silence_duration=0 * b2.ms,
        **kwargs,
    ):
        self.frequency = frequency

        sound = b2h.Sound.tone(frequency, duration, **kwargs)

        if level is not None:
            sound.level = level

        self.sound = gate_and_append_silence(
            sound,
            ramp_ms=ramp_ms,
            offset_silence_duration=offset_silence_duration,
        )

class ToneBurst:
    frequency: b2.Quantity
    sound: b2h.Sound
    burst_num: int

    def __init__(
        self,
        frequency: b2.Quantity,
        single_duration=DEFAULT_BURST_SINGLE_DURATION,
        burst_num=DEFAULT_BURST_REP,
        silence_duration=DEFAULT_SILENCE_DURATION,
        level=None,
        ramp_ms: float = 10.0,
        offset_silence_duration=0 * b2.ms,
        **kwargs,
    ):
        self.frequency = frequency
        self.burst_num = burst_num

        tone = b2h.Sound.tone(frequency, single_duration, **kwargs)

        if level is not None:
            tone.level = level

        tone = gate_and_append_silence(tone, ramp_ms=ramp_ms)

        silence = b2h.Sound.silence(silence_duration)

        pattern = b2h.Sound.sequence([tone, silence])
        sound = pattern.repeat(burst_num)

        if offset_silence_duration > 0 * b2.ms:
            sound = b2h.Sound.sequence(
                [sound, b2h.Sound.silence(offset_silence_duration)]
            )

        self.sound = sound

class WhiteNoise:
    sound: b2h.Sound

    def __init__(
        self,
        duration=DEFAULT_SOUND_DURATION,
        level=None,
        seed=DEFAULT_SEED,
        ramp_ms: float = 10.0,
        offset_silence_duration=0 * b2.ms,
        **kwargs,
    ):
        np.random.seed(seed)
        sound = b2h.Sound.whitenoise(duration, **kwargs)

        if level is not None:
            sound.level = level

        self.sound = gate_and_append_silence(
            sound,
            ramp_ms=ramp_ms,
            offset_silence_duration=offset_silence_duration,
        )

class HarmonicComplex:
    frequency: b2.Quantity
    sound: b2h.Sound

    def __init__(
        self,
        frequency: b2.Quantity,
        duration=DEFAULT_SOUND_DURATION,
        level=None,
        ramp_ms: float = 10.0,
        offset_silence_duration=0 * b2.ms,
        **kwargs,
    ):
        self.frequency = frequency
        sound = b2h.Sound.harmoniccomplex(frequency, duration, **kwargs)

        if level is not None:
            sound.level = level

        self.sound = gate_and_append_silence(
            sound,
            ramp_ms=ramp_ms,
            offset_silence_duration=offset_silence_duration,
        )

class Click:
    sound: b2h.Sound

    def __init__(self, duration=DEFAULT_SOUND_DURATION, click_duration = DEFAULT_CLICKS_DURATION, level=None, **kwargs):
        if level is not None:
            self.peak = level
        if isinstance(click_duration, b2.Quantity):
            c = b2h.click(click_duration, self.peak)
            i = b2h.silence(duration - click_duration)
        else:
            c = b2h.click(click_duration, self.peak)
            click_duration_ms = click_duration*(1/(c.samplerate / b2.Hz))*1000*b2.ms
            i = b2h.silence(duration - click_duration_ms)
        p = c + i
        self.sound = p

class Click_Train:
    sound: b2h.Sound

    def __init__(self, duration=DEFAULT_SOUND_DURATION, click_duration = DEFAULT_CLICKS_DURATION, interval = DEFAULT_CLICKS_INTERVAL, level=None, **kwargs):
        if level is not None:
            self.peak = level

        if isinstance(click_duration, b2.Quantity) and isinstance(interval, b2.Quantity):
            c = b2h.click(click_duration, self.peak)
            i = b2h.silence(interval)
            p = b2h.Sound.sequence(c,i)
            n = round(duration/(click_duration + interval))
            train = p.repeat(n)
        else:
            c = b2h.click(click_duration, self.peak)
            click_duration_ms = click_duration*(1/(c.samplerate / b2.Hz))*1000*b2.ms
            interval_ms = interval*(1/(c.samplerate / b2.Hz))*1000*b2.ms
            i = b2h.silence(interval_ms)
            p = b2h.Sound.sequence(c,i)
            n = round(int(duration/(click_duration_ms + interval_ms)))
            train = p.repeat(n)
    
        self.sound = train
        self.number = n

class DualTone:
    """Sum of two pure tones, each with independent frequency and level."""

    sound: b2h.Sound

    def __init__(
        self,
        frequency1: b2.Quantity,
        frequency2: b2.Quantity,
        duration=DEFAULT_SOUND_DURATION,
        level1=None,
        level2=None,
        ramp_ms: float = 10.0,
        offset_silence_duration=0 * b2.ms,
        **kwargs,
    ):
        self.frequency1 = frequency1
        self.frequency2 = frequency2

        tone1 = b2h.Sound.tone(frequency1, duration, **kwargs)
        tone2 = b2h.Sound.tone(frequency2, duration, **kwargs)

        if level1 is not None:
            tone1.level = level1
        if level2 is not None:
            tone2.level = level2

        # Sum the two tones
        combined = b2h.Sound(
            np.array(tone1.data) + np.array(tone2.data),
            samplerate=tone1.samplerate
        )

        self.sound = gate_and_append_silence(
            combined,
            ramp_ms=ramp_ms,
            offset_silence_duration=offset_silence_duration,
        )

class ToneFromAngle(Tone):
    # x = ToneFromAngle(20, 20 * b2.Hz)
    angle: int

    def __init__(self, angle, *args):
        self.angle = angle
        super().__init__(*args)

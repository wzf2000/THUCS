import wave
import contextlib
import numpy as np
import argparse


class BinauralSound():
    def __init__(self, hrir_path):
        self.hrir_l, self.hrir_r = self.load_hrir(hrir_path)

    def load_hrir(self, hrir_path):
        """
        Load HRIR data from given file path.
        HRIR data for each ear is a NumPy array of shape (25, 50, 200).
        """
        hrir_lr = np.load(hrir_path)
        return hrir_lr['hrir_l'], hrir_lr['hrir_r']

    def load_wav(self, path):
        """
        Load 1-channel 16-bit uncompressed WAV given file path.
        Returns sample rate and audio sample data. The audio sample data is a 1-D NumPy array of type np.int16.
        """
        with contextlib.closing(wave.open(path,'rb')) as f:
            if f.getcomptype() != 'NONE':
                raise ValueError('The input audio must not be compressed.')
            n_channels = f.getnchannels()
            if n_channels != 1:
                raise ValueError('The input audio must be 1-channel WAV.')
            sample_width = f.getsampwidth()
            if sample_width != 2:
                raise ValueError('The input audio must be 16-bit WAV.')
            sample_rate = f.getframerate()
            n_samples = f.getnframes()
            dat = f.readframes(n_samples * n_channels)
        audio = np.frombuffer(dat, dtype=np.int16)
        print(f'Loaded audio from {path}.')
        return sample_rate, audio

    def save_wav(self, path, sample_rate, samples):
        """
        Save 2-channel 16-bit WAV to the given path.
        """
        if isinstance(samples, np.ndarray) and samples.dtype == np.int16 and samples.ndim == 2 and samples.shape[1] == 2:
            with contextlib.closing(wave.open(path, 'w')) as f:
                f.setparams((2, 2, sample_rate, samples.shape[0], 'NONE', 'not compressed'))
                f.writeframes(samples.tobytes('C'))
        else:
            raise TypeError('The audio data must be numpy array with dtype int16 and shape (N, 2).')
        print(f'Saved audio to {path}.')
    
    def convolve_fft(self, data, filter):
        """
        data * filter => ifft(fft(data) x filter(data)), where * denotes convolution and x denotes multiplication.
        """
        L = len(data) + len(filter) - 1
        return np.fft.ifft(np.fft.fft(data, L) * np.fft.fft(filter, L)).real[:len(data)]

    def gen_demo(self, input_audio_path):
        sample_rate, audio = self.load_wav(input_audio_path)
        audio = audio.astype(np.float32)
        audio_l, audio_r = self.convolve_fft(audio, self.hrir_l[0, 8]), self.convolve_fft(audio, self.hrir_r[0, 8])
        audio_lr = np.stack((audio_l, audio_r), axis=1)
        # normalize the amplitude to avoid overflow
        i16min, i16max = np.iinfo(np.int16).min, np.iinfo(np.int16).max
        audio_lr = (audio_lr * i16max / np.abs(audio_lr).max()).clip(i16min, i16max).astype(np.int16) # [-32768, 32767]
        self.save_wav('left-side.wav', sample_rate, audio_lr) 

    def gen_360(self, input_audio_path):
        """
        Implement this function to generate binaural audio with time-varying space locations.
        For example, to generate an audio with the sound source rotating around.
        You could also try other audios for fun.
        """
        sample_rate, audio = self.load_wav(input_audio_path)
        audio = audio.astype(np.float32)
        n = 30
        block = audio.size // n
        if audio.size % n != 0:
            block += 1
        audio_l_list = []
        audio_r_list = []
        for i in range(n):
            audio_range = audio[(i - 2) * block: (i + 3) * block]
            audio_l_range, audio_r_range = self.convolve_fft(audio_range, self.hrir_l[min(i, 24), min(i, 24)]), self.convolve_fft(audio_range, self.hrir_r[min(i, 24), min(i, 24)])
            audio_l_list.append(audio_l_range[2 * block: 2 * block + min(block, len(audio) - i * block)])
            audio_r_list.append(audio_r_range[2 * block: 2 * block + min(block, len(audio) - i * block)])
        audio_l = np.concatenate(audio_l_list, axis=0)
        audio_r = np.concatenate(audio_r_list, axis=0)
        audio_lr = np.stack((audio_l, audio_r), axis=1)
        # normalize the amplitude to avoid overflow
        i16min, i16max = np.iinfo(np.int16).min, np.iinfo(np.int16).max
        audio_lr = (audio_lr * i16max / np.abs(audio_lr).max()).clip(i16min, i16max).astype(np.int16) # [-32768, 32767]
        self.save_wav('test.wav', sample_rate, audio_lr) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio', type=str, default='assets/nokia.wav')
    parser.add_argument('--demo', action='store_true', help='Run demo program.')
    args = parser.parse_args()

    bs = BinauralSound('assets/hrir.npz')

    if args.demo:
        bs.gen_demo(args.audio)
    else:
        bs.gen_360(args.audio)

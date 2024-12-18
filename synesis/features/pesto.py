"""
PESTO model for pitch estimation
Code from: https://github.com/SonyCSLParis/pesto/
GPLv3
"""

import warnings
from functools import partial
from typing import Any, Mapping, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import get_window

OUTPUT_TYPE = Union[
    Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
]


class CropCQT(nn.Module):
    def __init__(self, min_steps: int, max_steps: int):
        super(CropCQT, self).__init__()
        self.min_steps = min_steps
        self.max_steps = max_steps

        # lower bin
        self.lower_bin = self.max_steps

    def forward(self, spectrograms: torch.Tensor) -> torch.Tensor:
        # WARNING: didn't check that it works, it may be dangerous
        return spectrograms[..., self.max_steps : self.min_steps]

        # old implementation
        batch_size, _, input_height = spectrograms.size()

        output_height = input_height - self.max_steps + self.min_steps
        assert output_height > 0, (
            f"With input height {input_height:d} and output height {output_height:d}, "
            f"impossible to have a range of {self.max_steps - self.min_steps:d} bins."
        )

        return spectrograms[..., self.lower_bin : self.lower_bin + output_height]


def broadcast_dim(x):
    """
    Auto broadcast input so that it can fits into a Conv1d
    """

    if x.dim() == 2:
        x = x[:, None, :]
    elif x.dim() == 1:
        # If nn.DataParallel is used, this broadcast doesn't work
        x = x[None, None, :]
    elif x.dim() == 3:
        pass
    else:
        raise ValueError(
            "Only support input with shape = (batch, len) or shape = (len)"
        )
    return x


def nextpow2(A):
    """A helper function to calculate the next nearest number to the power of 2.

    Parameters
    ----------
    A : float
        A float number that is going to be rounded up to the nearest power of 2

    Returns
    -------
    int
        The nearest power of 2 to the input number ``A``

    Examples
    --------

    >>> nextpow2(6)
    3
    """

    return int(np.ceil(np.log2(A)))


def get_window_dispatch(window, N, fftbins=True):
    if isinstance(window, str):
        return get_window(window, N, fftbins=fftbins)
    elif isinstance(window, tuple):
        if window[0] == "gaussian":
            assert window[1] >= 0
            sigma = np.floor(-N / 2 / np.sqrt(-2 * np.log(10 ** (-window[1] / 20))))
            return get_window(("gaussian", sigma), N, fftbins=fftbins)
        else:
            Warning("Tuple windows may have undesired behaviour regarding Q factor")
    elif isinstance(window, float):
        Warning(
            "You are using Kaiser window with beta factor "
            + str(window)
            + ". Correct behaviour not checked."
        )
    else:
        raise Exception(
            "The function get_window from scipy only supports strings, tuples and floats."
        )


def create_cqt_kernels(
    Q,
    fs,
    fmin: float,
    n_bins=84,
    bins_per_octave=12,
    norm=1,
    window="hann",
    fmax: Optional[float] = None,
    topbin_check=True,
    gamma=0,
    pad_fft=True,
):
    """
    Automatically create CQT kernels in time domain
    """
    if (fmax is not None) and (n_bins is None):
        n_bins = np.ceil(
            bins_per_octave * np.log2(fmax / fmin)
        )  # Calculate the number of bins
        freqs = fmin * 2.0 ** (np.r_[0:n_bins] / float(bins_per_octave))

    elif (fmax is None) and (n_bins is not None):
        freqs = fmin * 2.0 ** (np.r_[0:n_bins] / float(bins_per_octave))

    else:
        warnings.warn("If fmax is given, n_bins will be ignored", SyntaxWarning)
        n_bins = np.ceil(
            bins_per_octave * np.log2(fmax / fmin)
        )  # Calculate the number of bins
        freqs = fmin * 2.0 ** (np.r_[0:n_bins] / float(bins_per_octave))

    if np.max(freqs) > fs / 2 and topbin_check:
        raise ValueError(
            "The top bin {}Hz has exceeded the Nyquist frequency, \
                          please reduce the n_bins".format(np.max(freqs))
        )

    alpha = 2.0 ** (1.0 / bins_per_octave) - 1.0
    lengths = np.ceil(Q * fs / (freqs + gamma / alpha))

    # get max window length depending on gamma value
    max_len = int(max(lengths))
    fftLen = int(2 ** (np.ceil(np.log2(max_len))))

    tempKernel = np.zeros((int(n_bins), int(fftLen)), dtype=np.complex64)
    # specKernel = np.zeros((int(n_bins), int(fftLen)), dtype=np.complex64)

    for k in range(0, int(n_bins)):
        freq = freqs[k]
        l = lengths[k]

        # Centering the kernels
        if l % 2 == 1:  # pad more zeros on RHS
            start = int(np.ceil(fftLen / 2.0 - l / 2.0)) - 1
        else:
            start = int(np.ceil(fftLen / 2.0 - l / 2.0))

        window_dispatch = get_window_dispatch(window, int(l), fftbins=True)
        sig = (
            window_dispatch
            * np.exp(np.r_[-l // 2 : l // 2] * 1j * 2 * np.pi * freq / fs)
            / l
        )

        if norm:  # Normalizing the filter # Trying to normalize like librosa
            tempKernel[k, start : start + int(l)] = sig / np.linalg.norm(sig, norm)
        else:
            tempKernel[k, start : start + int(l)] = sig
        # specKernel[k, :] = fft(tempKernel[k])

    # return specKernel[:,:fftLen//2+1], fftLen, torch.tensor(lenghts).float()
    return tempKernel, fftLen, torch.tensor(lengths).float(), freqs


class CQT(nn.Module):
    """This function is to calculate the CQT of the input signal.
    Input signal should be in either of the following shapes.\n
    1. ``(len_audio)``\n
    2. ``(num_audio, len_audio)``\n
    3. ``(num_audio, 1, len_audio)``

    The correct shape will be inferred autommatically if the input follows these 3
        shapes.
    Most of the arguments follow the convention from librosa.
    This class inherits from ``nn.Module``, therefore, the usage is same as
        ``nn.Module``.

    This alogrithm uses the method proposed in [1]. I slightly modify it so that it
        runs faster
    than the original 1992 algorithm, that is why I call it version 2.
    [1] Brown, Judith C.C. and Miller Puckette. “An efficient algorithm for the
        calculation of a
    constant Q transform.” (1992).

    Parameters
    ----------
    sr : int
        The sampling rate for the input audio. It is used to calucate the correct
            ``fmin`` and ``fmax``.
        Setting the correct sampling rate is very important for calculating the correct
            frequency.

    hop_length : int
        The hop (or stride) size. Default value is 512.

    fmin : float
        The frequency for the lowest CQT bin. Default is 32.70Hz, which coresponds to
            the note C0.

    fmax : float
        The frequency for the highest CQT bin. Default is ``None``, therefore the higest
            CQT bin is
        inferred from the ``n_bins`` and ``bins_per_octave``.
        If ``fmax`` is not ``None``, then the argument ``n_bins`` will be ignored and
            ``n_bins``
        will be calculated automatically. Default is ``None``

    n_bins : int
        The total numbers of CQT bins. Default is 84. Will be ignored if ``fmax`` is
            not ``None``.

    bins_per_octave : int
        Number of bins per octave. Default is 12.

    filter_scale : float > 0
        Filter scale factor. Values of filter_scale smaller than 1 can be used to
            improve the time resolution at the
        cost of degrading the frequency resolution. Important to note is that setting
            for example filter_scale = 0.5 and
        bins_per_octave = 48 leads to exactly the same time-frequency resolution
            trade-off as setting filter_scale = 1
        and bins_per_octave = 24, but the former contains twice more frequency bins per
            octave. In this sense, values
        filter_scale < 1 can be seen to implement oversampling of the frequency axis,
            analogously to the use of zero
        padding when calculating the DFT.

    norm : int
        Normalization for the CQT kernels. ``1`` means L1 normalization, and ``2``
            means L2 normalization.
        Default is ``1``, which is same as the normalization used in librosa.

    window : string, float, or tuple
        The windowing function for CQT. If it is a string, It uses
            ``scipy.signal.get_window``. If it is a
        tuple, only the gaussian window wanrantees constant Q factor. Gaussian window
            hould be given as a
        tuple ('gaussian', att) where att is the attenuation in the border given in dB.
        Please refer to scipy documentation for possible windowing functions. The
            default value is 'hann'.

    center : bool
        Putting the CQT keneral at the center of the time-step or not. If ``False``,
            the time index is
        the beginning of the CQT kernel, if ``True``, the time index is the center of
            the CQT kernel.
        Default value if ``True``.

    pad_mode : str
        The padding method. Default value is 'reflect'.

    trainable : bool
        Determine if the CQT kernels are trainable or not. If ``True``, the gradients
            for CQT kernels
        will also be caluclated and the CQT kernels will be updated during model
            training.
        Default value is ``False``.

    output_format : str
        Determine the return type.
        ``Magnitude`` will return the magnitude of the STFT result, shape = ``
            num_samples, freq_bins,time_steps)``;
        ``Complex`` will return the STFT result in complex number, shape = ``
            num_samples, freq_bins,time_steps, 2)``;
        ``Phase`` will return the phase of the STFT reuslt, shape = ``(num_samples, freq
            bins,time_steps, 2)``.
        The complex number is stored as ``(real, imag)`` in the last axis. Default value
            is 'Magnitude'.

    Returns
    -------
    spectrogram : torch.Tensor
    It returns a tensor of spectrograms.
    shape = ``(num_samples, freq_bins,time_steps)`` if ``output_format='Magnitude'``;
    shape = ``(num_samples, freq_bins,time_steps, 2)`` if ``output_format='Complex' or
        'Phase'``;

    Examples
    --------
    >>> spec_layer = CQT()
    >>> specs = spec_layer(x)
    """

    def __init__(
        self,
        sr=22050,
        hop_length=512,
        fmin=32.70,
        fmax=None,
        n_bins=84,
        bins_per_octave=12,
        filter_scale=1,
        norm=1,
        window="hann",
        center=True,
        pad_mode="reflect",
        trainable=False,
        output_format="Magnitude",
    ):
        super().__init__()

        self.trainable = trainable
        self.hop_length = hop_length
        self.center = center
        self.pad_mode = pad_mode
        self.output_format = output_format

        # creating kernels for CQT
        Q = float(filter_scale) / (2 ** (1 / bins_per_octave) - 1)

        cqt_kernels, self.kernel_width, lenghts, freqs = create_cqt_kernels(
            Q, sr, fmin, n_bins, bins_per_octave, norm, window, fmax
        )

        self.register_buffer("lenghts", lenghts)
        self.frequencies = freqs

        cqt_kernels_real = torch.tensor(cqt_kernels.real).unsqueeze(1)
        cqt_kernels_imag = torch.tensor(cqt_kernels.imag).unsqueeze(1)

        if trainable:  # NOTE: can't it be factorized?
            cqt_kernels_real = nn.Parameter(cqt_kernels_real, requires_grad=trainable)
            cqt_kernels_imag = nn.Parameter(cqt_kernels_imag, requires_grad=trainable)
            self.register_parameter("cqt_kernels_real", cqt_kernels_real)
            self.register_parameter("cqt_kernels_imag", cqt_kernels_imag)
        else:
            self.register_buffer("cqt_kernels_real", cqt_kernels_real)
            self.register_buffer("cqt_kernels_imag", cqt_kernels_imag)

    def forward(self, x, output_format=None, normalization_type="librosa"):
        """
        Convert a batch of waveforms to CQT spectrograms.

        Parameters
        ----------
        x : torch tensor
            Input signal should be in either of the following shapes.\n
            1. ``(len_audio)``\n
            2. ``(num_audio, len_audio)``\n
            3. ``(num_audio, 1, len_audio)``
            It will be automatically broadcast to the right shape

        normalization_type : str
            Type of the normalization. The possible options are: \n
            'librosa' : the output fits the librosa one \n
            'convolutional' : the output conserves the convolutional inequalities of the
                wavelet transform:\n
            for all p ϵ [1, inf] \n
                - || CQT ||_p <= || f ||_p || g ||_1 \n
                - || CQT ||_p <= || f ||_1 || g ||_p \n
                - || CQT ||_2 = || f ||_2 || g ||_2 \n
            'wrap' : wraps positive and negative frequencies into positive frequencies.
                      This means that the CQT of a
            sinus (or a cosine) with a constant amplitude equal to 1 will have the value
                    1 in the bin corresponding to
            its frequency.
        """
        output_format = output_format or self.output_format

        x = broadcast_dim(x)
        if self.center:
            if self.pad_mode == "constant":
                padding = nn.ConstantPad1d(self.kernel_width // 2, 0)
            elif self.pad_mode == "reflect":
                padding = nn.ReflectionPad1d(self.kernel_width // 2)

            x = padding(x)

        # CQT
        CQT_real = F.conv1d(x, self.cqt_kernels_real, stride=self.hop_length)
        CQT_imag = -F.conv1d(x, self.cqt_kernels_imag, stride=self.hop_length)

        if normalization_type == "librosa":
            CQT_real *= torch.sqrt(self.lenghts.view(-1, 1))
            CQT_imag *= torch.sqrt(self.lenghts.view(-1, 1))
        elif normalization_type == "convolutional":
            pass
        elif normalization_type == "wrap":
            CQT_real *= 2
            CQT_imag *= 2
        else:
            raise ValueError(
                "The normalization_type %r is not part of our current options."
                % normalization_type
            )

        if output_format == "Magnitude":
            margin = 1e-8 if self.trainable else 0
            return torch.sqrt(CQT_real.pow(2) + CQT_imag.pow(2) + margin)

        elif output_format == "Complex":
            return torch.stack((CQT_real, CQT_imag), -1)

        elif output_format == "Phase":
            phase_real = torch.cos(torch.atan2(CQT_imag, CQT_real))
            phase_imag = torch.sin(torch.atan2(CQT_imag, CQT_real))
            return torch.stack((phase_real, phase_imag), -1)


class HarmonicCQT(nn.Module):
    r"""Harmonic CQT layer, as described in Bittner et al. (20??)"""

    def __init__(
        self,
        harmonics,
        sr: int = 22050,
        hop_length: int = 512,
        fmin: float = 32.7,
        fmax: Optional[float] = None,
        bins_per_semitone: int = 1,
        n_bins: int = 84,
        center_bins: bool = True,
    ):
        super(HarmonicCQT, self).__init__()

        if center_bins:
            fmin = fmin / 2 ** ((bins_per_semitone - 1) / (24 * bins_per_semitone))

        self.cqt_kernels = nn.ModuleList(
            [
                CQT(
                    sr=sr,
                    hop_length=hop_length,
                    fmin=h * fmin,
                    fmax=fmax,
                    n_bins=n_bins,
                    bins_per_octave=12 * bins_per_semitone,
                    output_format="Complex",
                )
                for h in harmonics
            ]
        )

    def forward(self, audio_waveforms: torch.Tensor):
        r"""Converts a batch of waveforms into a batch of HCQTs.

        Args:
            audio_waveforms (torch.Tensor): Batch of waveforms, shape (batch_size,
                num_samples)

        Returns:
            Harmonic CQT, shape (batch_size, num_harmonics, num_freqs, num_timesteps, 2)
        """
        return torch.stack([cqt(audio_waveforms) for cqt in self.cqt_kernels], dim=1)


class ToLogMagnitude(nn.Module):
    def __init__(self):
        super(ToLogMagnitude, self).__init__()
        self.eps = torch.finfo(torch.float32).eps

    def forward(self, x):
        x = x.abs()
        x.clamp_(min=self.eps).log10_().mul_(20)
        return x


class Preprocessor(nn.Module):
    r"""

    Args:
        hop_size (float): step size between consecutive CQT frames (in milliseconds)
    """

    def __init__(
        self, hop_size: float, sampling_rate: Optional[int] = None, **hcqt_kwargs
    ):
        super(Preprocessor, self).__init__()

        # HCQT
        self.hcqt_sr = None
        self.hcqt_kernels = None
        self.hop_size = hop_size

        self.hcqt_kwargs = hcqt_kwargs

        # log-magnitude
        self.to_log = ToLogMagnitude()

        # register a dummy tensor to get implicit access to the module's device
        self.register_buffer("_device", torch.zeros(()), persistent=False)

        # if the sampling rate is provided, instantiate the CQT kernels
        if sampling_rate is not None:
            self.hcqt_sr = sampling_rate
            self._reset_hcqt_kernels()

    def forward(self, x: torch.Tensor, sr: Optional[int] = None) -> torch.Tensor:
        r"""

        Args:
            x (torch.Tensor): audio waveform or batch of audio waveforms, any sampling
                              rate,
                shape (batch_size?, num_samples)
            sr (int, optional): sampling rate

        Returns:
            torch.Tensor: log-magnitude CQT of batch of CQTs,
                shape (batch_size?, num_timesteps, num_harmonics, num_freqs)
        """
        # compute CQT from input waveform, and invert dims for (time_steps,
        # num_harmonics, freq_bins)
        # in other words, time becomes the batch dimension, enabling efficient
        # processing for long audios.
        complex_cqt = torch.view_as_complex(self.hcqt(x, sr=sr)).permute(0, 3, 1, 2)
        complex_cqt.squeeze_(0)

        # convert to dB
        return self.to_log(complex_cqt)

    def hcqt(self, audio: torch.Tensor, sr: Optional[int] = None) -> torch.Tensor:
        r"""Compute the Harmonic CQT of the input audio after eventually recreating the
            kernels
        (in case the sampling rate has changed).

        Args:
            audio (torch.Tensor): mono audio waveform, shape (batch_size, num_samples)
            sr (int): sampling rate of the audio waveform.
                If not specified, we assume it is the same as the previous processed
                audio waveform.

        Returns:
            torch.Tensor: Complex Harmonic CQT (HCQT) of the input,
                shape (batch_size, num_harmonics, num_freqs, num_timesteps, 2)
        """
        # compute HCQT kernels if it does not exist or if the sampling rate has changed
        if sr is not None and sr != self.hcqt_sr:
            self.hcqt_sr = sr
            self._reset_hcqt_kernels()

        return self.hcqt_kernels(audio)

    def _reset_hcqt_kernels(self) -> None:
        hop_length = int(self.hop_size * self.hcqt_sr / 1000 + 0.5)
        self.hcqt_kernels = HarmonicCQT(
            sr=self.hcqt_sr, hop_length=hop_length, **self.hcqt_kwargs
        ).to(self._device.device)


class ToeplitzLinear(nn.Conv1d):
    def __init__(self, in_features, out_features):
        super(ToeplitzLinear, self).__init__(
            in_channels=1,
            out_channels=1,
            kernel_size=in_features + out_features - 1,
            padding=out_features - 1,
            bias=False,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return super(ToeplitzLinear, self).forward(input.unsqueeze(-2)).squeeze(-2)


class Resnet1d(nn.Module):
    """
    Basic CNN similar to the one in Johannes Zeitler's report,
    but for longer HCQT input (always stride 1 in time)
    Still with 75 (-1) context frames, i.e. 37 frames padded to each side
    The number of input channels, channels in the hidden layers, and output
    dimensions (e.g. for pitch output) can be parameterized.
    Layer normalization is only performed over frequency and channel dimensions,
    not over time (in order to work with variable length input).
    Outputs one channel with sigmoid activation.

    Args (Defaults: BasicCNN by Johannes Zeitler but with 6 input channels):
        n_chan_input:     Number of input channels (harmonics in HCQT)
        n_chan_layers:    Number of channels in the hidden layers (list)
        n_prefilt_layers: Number of repetitions of the prefiltering layer
        residual:         If True, use residual connections for prefiltering
                          (default: False)
        n_bins_in:        Number of input bins (12 * number of octaves)
        n_bins_out:       Number of output bins (12 for pitch class, 72 for pitch,
                          num_octaves * 12)
        a_lrelu:          alpha parameter (slope) of LeakyReLU activation function
        p_dropout:        Dropout probability
    """

    def __init__(
        self,
        n_chan_input=1,
        n_chan_layers=(20, 20, 10, 1),
        n_prefilt_layers=1,
        prefilt_kernel_size=15,
        residual=False,
        n_bins_in=216,
        output_dim=128,
        activation_fn: str = "leaky",
        a_lrelu=0.3,
        p_dropout=0.2,
    ):
        super(Resnet1d, self).__init__()

        self.hparams = dict(
            n_chan_input=n_chan_input,
            n_chan_layers=n_chan_layers,
            n_prefilt_layers=n_prefilt_layers,
            prefilt_kernel_size=prefilt_kernel_size,
            residual=residual,
            n_bins_in=n_bins_in,
            output_dim=output_dim,
            activation_fn=activation_fn,
            a_lrelu=a_lrelu,
            p_dropout=p_dropout,
        )

        if activation_fn == "relu":
            activation_layer = nn.ReLU
        elif activation_fn == "silu":
            activation_layer = nn.SiLU
        elif activation_fn == "leaky":
            activation_layer = partial(nn.LeakyReLU, negative_slope=a_lrelu)
        else:
            raise ValueError

        n_in = n_chan_input
        n_ch = n_chan_layers
        if len(n_ch) < 5:
            n_ch.append(1)

        # Layer normalization over frequency and channels (harmonics of HCQT)
        self.layernorm = nn.LayerNorm(normalized_shape=[n_in, n_bins_in])

        # Prefiltering
        prefilt_padding = prefilt_kernel_size // 2
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=n_in,
                out_channels=n_ch[0],
                kernel_size=prefilt_kernel_size,
                padding=prefilt_padding,
                stride=1,
            ),
            activation_layer(),
            nn.Dropout(p=p_dropout),
        )
        self.n_prefilt_layers = n_prefilt_layers
        self.prefilt_layers = nn.ModuleList(
            *[
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=n_ch[0],
                        out_channels=n_ch[0],
                        kernel_size=prefilt_kernel_size,
                        padding=prefilt_padding,
                        stride=1,
                    ),
                    activation_layer(),
                    nn.Dropout(p=p_dropout),
                )
                for _ in range(n_prefilt_layers - 1)
            ]
        )
        self.residual = residual

        conv_layers = []
        for i in range(len(n_chan_layers) - 1):
            conv_layers.extend(
                [
                    nn.Conv1d(
                        in_channels=n_ch[i],
                        out_channels=n_ch[i + 1],
                        kernel_size=1,
                        padding=0,
                        stride=1,
                    ),
                    activation_layer(),
                    nn.Dropout(p=p_dropout),
                ]
            )
        self.conv_layers = nn.Sequential(*conv_layers)

        self.flatten = nn.Flatten(start_dim=1)
        self.fc = ToeplitzLinear(n_bins_in * n_ch[-1], output_dim)

        self.final_norm = nn.Softmax(dim=-1)

    def forward(self, x):
        r"""

        Args:
            x (torch.Tensor): shape (batch, channels, freq_bins)
        """
        x = self.layernorm(x)

        x = self.conv1(x)
        for p in range(0, self.n_prefilt_layers - 1):
            prefilt_layer = self.prefilt_layers[p]
            if self.residual:
                x_new = prefilt_layer(x)
                x = x_new + x
            else:
                x = prefilt_layer(x)

        x = self.conv_layers(x)
        x = self.flatten(x)

        y_pred = self.fc(x)

        return self.final_norm(y_pred)


class PESTO(nn.Module):
    def __init__(
        self,
        crop_kwargs: Optional[Mapping[str, Any]] = None,
        reduction: str = "alwa",
        feature_extractor: bool = False,
    ):
        super(PESTO, self).__init__()

        # constant shift to get absolute pitch from predictions
        self.register_buffer(
            "shift", torch.zeros((), dtype=torch.float), persistent=True
        )

    def forward(
        self,
        audio_waveforms: torch.Tensor,
        sr: Optional[int] = None,
        convert_to_freq: bool = False,
    ) -> OUTPUT_TYPE:
        r"""

        Args:
            audio_waveforms (torch.Tensor): mono audio waveform or batch of mono audio
                waveforms, shape (batch_size?, num_samples)
            sr (int, optional): sampling rate, defaults to the previously used sampling
                rate
            convert_to_freq (bool): whether to convert the result to frequencies or
                return fractional semitones instead.
            return_activations (bool): whether to return activations or pitch
                predictions only

        Returns:
            preds (torch.Tensor): pitch predictions in SEMITONES, shape (batch_size?,
                num_timesteps)
                where `num_timesteps` ~= `num_samples` / (`self.hop_size` * `sr`)
            confidence (torch.Tensor): confidence of whether frame is voiced or unvoiced
                in [0, 1],
                shape (batch_size?, num_timesteps)
            activations (torch.Tensor): activations of the model, shape (batch_size?,
                num_timesteps, output_dim)
        """
        # squeeze channel dimension
        audio_waveforms = audio_waveforms.squeeze(1)

        batch_size = audio_waveforms.size(0) if audio_waveforms.ndim == 2 else None
        x = self.preprocessor(audio_waveforms, sr=sr)
        x = self.crop_cqt(x)  # the CQT has to be cropped beforehand

        # for now, confidence is computed very naively just based on energy in the CQT
        confidence = x.mean(dim=-2).max(dim=-1).values
        conf_min, conf_max = (
            confidence.min(dim=-1, keepdim=True).values,
            confidence.max(dim=-1, keepdim=True).values,
        )
        confidence = (confidence - conf_min) / (conf_max - conf_min)

        # flatten batch_size and time_steps since anyway predictions are made on CQT
        # frames independently
        if batch_size:
            x = x.flatten(0, 1)

        activations = self.encoder(x)
        if batch_size:
            activations = activations.view(batch_size, -1, activations.size(-1))

        activations = activations.roll(
            -round(self.shift.cpu().item() * self.bins_per_semitone), -1
        )

        # preds = reduce_activations(activations, reduction=self.reduction)

        # if convert_to_freq:
        #     preds = 440 * 2 ** ((preds - 69) / 12)

        # At this point, activations are (b, time_steps, feature_dim), feature_dim=384.
        # We will mean features over time to get a single feature vector per input frame
        activations = activations.mean(dim=1)

        return activations

        # return preds, confidence

    def load_state_dict(self, state_dict, strict: bool = False):
        """Intercept the loading of the state dict to load other stuff as well"""
        # load checkpoint
        hparams = state_dict["hparams"]
        hcqt_params = state_dict["hcqt_params"]
        state_dict = state_dict["state_dict"]

        # instantiate preprocessor
        self.preprocessor = Preprocessor(
            hop_size=10.0, sampling_rate=22050, **hcqt_params
        )

        # instantiate PESTO encoder
        self.encoder = Resnet1d(**hparams["encoder"])

        # crop CQT
        if hparams["pitch_shift"] is None:
            hparams["pitch_shift"] = {}
        self.crop_cqt = CropCQT(**hparams["pitch_shift"])

        self.reduction = hparams["reduction"]

        super().load_state_dict(state_dict, strict=strict)
        super().eval()

    @property
    def bins_per_semitone(self) -> int:
        return self.preprocessor.hcqt_kwargs["bins_per_semitone"]

    @property
    def hop_size(self) -> float:
        r"""Returns the hop size of the model (in milliseconds)"""
        return self.preprocessor.hop_size

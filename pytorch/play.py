# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# 
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
# 
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 
# *****************************************************************************
import os
from scipy.io.wavfile import write
import torch
import nv_wavenet
import utils
from threading import Thread
import pyaudio
import numpy as np
import time

WIDTH = 2 # bytewidth of 16 linear PCM
CHANNELS = 1
RATE = 22050

def play(buffer_size, callback):
    p = pyaudio.PyAudio()

    stream = p.open(format=p.get_format_from_width(WIDTH),
                    channels=CHANNELS,
                    rate=RATE,
                    output=True,
                    frames_per_buffer=buffer_size,
                    stream_callback=callback)

    stream.start_stream()

    while stream.is_active():
        time.sleep(0.1)

    stream.stop_stream()
    stream.close()

    p.terminate()

def main(mel_file, model_filename, buffer_size, implementation):
    model = torch.load(model_filename)['model']
    wavenet = nv_wavenet.NVWaveNet(**(model.export_weights()))

    mel = torch.load(mel_file)
    mel = utils.to_gpu(mel)
    mels = [torch.unsqueeze(mel, 0)]
    cond_input = model.get_cond_input(torch.cat(mels, 0))

    sample_count = cond_input.size(3)
    buffer = wavenet.infer_streaming(cond_input, implementation, torch.IntTensor(1, sample_count), buffer_size)
    def buffer_callback(in_data, frame_count, time_info, status):
        #print('fetching buffer')
        try:
            audio_chunk = next(buffer)
            audio_chunk = utils.mu_law_decode_numpy(audio_chunk[0,:].cpu().numpy(), 256)
            audio_chunk = utils.MAX_WAV_VALUE * audio_chunk
            if audio_chunk.shape[0] < frame_count:
                audio_chunk = np.pad(audio_chunk, (0, frame_count-audio_chunk.shape[0]), 'constant', constant_values=0)
            data = audio_chunk.astype('int16').tostring()
            return (data, pyaudio.paContinue)
        except:
            return ('', pyaudio.paComplete)

    # def chatter():
    #     print("not blocked")

    # chatter_t = Thread(target=chatter)
    # chatter_t.start()

    play(buffer_size, buffer_callback)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--file_path", required=True)
    parser.add_argument('-c', "--checkpoint_path", required=True)
    parser.add_argument('-s', "--buffer_size", type=int, default=0)
    parser.add_argument('-i', "--implementation", type=str, default="persistent",
                        help="""Which implementation of NV-WaveNet to use.
                        Takes values of single, dual, or persistent""" )
    
    args = parser.parse_args()
    if args.implementation == "auto":
        implementation = nv_wavenet.Impl.AUTO
    elif args.implementation == "single":
        implementation = nv_wavenet.Impl.SINGLE_BLOCK
    elif args.implementation == "dual":
        implementation = nv_wavenet.Impl.DUAL_BLOCK
    elif args.implementation == "persistent":
        implementation = nv_wavenet.Impl.PERSISTENT
    else:
        raise ValueError("implementation must be one of auto, single, dual, or persistent")
    
    main(args.file_path, args.checkpoint_path, args.buffer_size, implementation)

from flask import Flask, Blueprint, request, render_template, jsonify, Response
from flask import current_app
import os
from tflite_runtime.interpreter import Interpreter
import librosa
import python_speech_features
import numpy as np
# from db import c_firebase as fb
import datetime
from collections import Counter
# from matplotlib.figure import Figure
# import random
# from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
# import io

mod = Blueprint('prediction', __name__,
                template_folder='templates', static_folder='static')

# model_path = 'src/model/mfcc_13_13.tflite'
model_path = 'model/mfcc_13_13.tflite'
UPLOAD_URL = 'app/audio/'
# UPLOAD_URL = 'server/src/audio/'


@mod.route('/')
def home():
    return render_template('index.html')


@mod.route('/prediction', methods=['POST'])
def prediction():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return "someting went wrong 1"

        # Get file name from URL
        user_file = request.files['file']
        temp = request.files['file']
        if user_file.filename == '':
            return "file name not found ..."

        else:

            path_file = os.path.join(
                current_app.config['UPLOAD_FOLDER'], user_file.filename)

            user_file.save(path_file)

            # PREDICTION RESULTS ## 🙌🥂

            pred, conf = identifyAudio(path_file)

            label_class = winner(pred)
            confidence = winner(conf)

            print('class : {}, confidence : {:.2f}'.format(
                label_class[0][0], confidence[0][0] * 100))

            time_result = datetime.datetime.utcnow() + datetime.timedelta(hours=7)
            x_time = time_result.strftime('%d/%m/%y - %X')

            # fb.set_data(
            #     'prediction',
            #     'keng',
            #     user_file.filename,
            #     str(label_class[0][0]),
            #     str(round(confidence[0][0] * 100, 2)),
            #     x_time,
            #     UPLOAD_URL + user_file.filename)

            rm_file(path_file)
            # return render_template('result.html')
            return jsonify({
                'LABEL': str(label_class[0][0]),
                "ACC": str(round(confidence[0][0] * 100, 2)) + ' %',
                'TIME': x_time
            })

# def create_figure(xs):
#     fig = Figure()
#     axis = fig.add_subplot(1, 1, 1)
#     # xs = range(8000)
#     # ys = [random.randint(1, 50) for x in xs]
#     axis.plot(xs)
#     return fig

# collect for the most of word


def winner(input):
    votes = Counter(input)
    return votes.most_common(1)


def rm_file(path):
    """Removes a file from the file system.

    Args:
        path (str): the absolute path of the file to be removed

    Returns:
        True on success
    """

    from traceback import print_exc

    if os.path.isfile(path):
        try:
            os.remove(path)
            return True
        except:
            print_exc()
    return False


def envelope(y, rate, threshold):
    import pandas as pd

    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate/20),
                       min_periods=1,
                       center=True).max()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask, y_mean


def check_sample(signal, fs):
    delta_sample = int(fs)

    if signal.shape[0] < delta_sample:
        # print('This is in loop if.')
        sample = np.zeros(shape=(delta_sample, ), dtype=np.float32)
        sample[: signal.shape[0]] = signal
        return sample, fs

    else:
        trunc = signal.shape[0] % delta_sample
        # print('This is in loop else.')
        for cnt, i in enumerate(np.arange(0, signal.shape[0] - trunc, delta_sample)):
            start = int(i)
            stop = int(i + delta_sample)
            sample = signal[start:stop]
            return sample, fs


# TODO Extract segmets
def extract_segments(clip, frames):
    FRAMES_PER_SEGMENT = frames - 1
    WINDOW_SIZE = 150 * FRAMES_PER_SEGMENT
    STEP_SIZE = 150 * FRAMES_PER_SEGMENT  # // 2
    BANDS = 13

    segments = []
    s = 0

    normalization_factor = 1 / np.max(np.abs(clip))
    clip = clip * normalization_factor

    while len(clip[s * STEP_SIZE:s * STEP_SIZE + WINDOW_SIZE]) == WINDOW_SIZE:
        signal = clip[s * STEP_SIZE:s * STEP_SIZE + WINDOW_SIZE]

        envo, y_mean = envelope(signal, 8000, threshold=0.4)
        signal = signal[envo]

        if len(signal) == 0:
            pass

        else:
            sample, fs = check_sample(signal, 1100)

            # Comput features
            mfccs = python_speech_features.mfcc(sample,
                                                samplerate=8000,
                                                winlen=0.02,
                                                winstep=0.01,
                                                numcep=BANDS,
                                                nfilt=26,
                                                nfft=1024,
                                                preemph=0.0,
                                                ceplifter=0,
                                                appendEnergy=False,
                                                winfunc=np.hamming)

            mfccs = mfccs.transpose()

            segments.append(mfccs)

        s = s + 1

    return segments


def identifyAudio(path):
    interpreter = Interpreter(model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    signal, fs = librosa.load(path, sr=8000, mono=True)

    mfccs = extract_segments(signal, 10)
    pred = []
    confidence = []

    for index, _ in enumerate(mfccs):
        for _ in range(index):
            # Make prediction input (1, 13, 13, 1)
            in_tensor = np.float32(mfccs[index].reshape(
                1, mfccs[index].shape[0], mfccs[index].shape[1], 1))

            interpreter.set_tensor(input_details[0]['index'], in_tensor)
            interpreter.invoke()

            labels = ['ripe', 'unripe', 'mid-ripe']

            output_data = interpreter.get_tensor(output_details[0]['index'])
            val = output_data[0]

            v = max(val)

            if v > 0.5:  # percent of accuracy
                for i, j in enumerate(val):
                    if j == v:
                        pred.append(labels[i])
                        confidence.append(v)

    return pred, confidence

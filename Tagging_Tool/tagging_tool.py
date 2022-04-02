#import pandas as pd
#import streamlit as st
import os
from moviepy.editor import VideoFileClip
import librosa
import keras
from keras import backend as K
import sklearn
import numpy as np

FRAME_SIZE = 10 #seconds
eps = np.finfo(np.float).eps

#st.set_page_config(page_icon="⚙️", page_title="Tagging Tool", layout='wide')

def convert_video_to_audio_moviepy(video_file, output_ext="wav"):
    """Converts video to audio using MoviePy library
    that uses `ffmpeg` under the hood"""
    filename, ext = os.path.splitext(video_file)
    clip = VideoFileClip(video_file)
    output_path = f"{filename}.{output_ext}"
    clip.audio.write_audiofile(output_path)
    return f"{filename}.{output_ext}"

def class_mae(y_true, y_pred):
    return K.mean(
        K.abs(
            K.argmax(y_pred, axis=-1) - K.argmax(y_true, axis=-1)
        ),
        axis=-1
    )


def load_model(model_path):
    model = keras.models.load_model(
        model_path,
        custom_objects={
            'class_mae': class_mae,
            'exp': K.exp
        }
    )
    return model


def count(audio, model, scaler):
    # compute STFT
    X = np.abs(librosa.stft(audio, n_fft=400, hop_length=160)).T

    # apply global (featurewise) standardization to mean1, var0
    X = scaler.transform(X)

    # cut to input shape length (500 frames x 201 STFT bins)
    X = X[:500, :]

    # apply l2 normalization
    Theta = np.linalg.norm(X, axis=1) + eps
    X /= np.mean(Theta)

    # add sample dimension
    X = X[np.newaxis, ...]

    if len(model.input_shape) == 4:
        X = X[:, np.newaxis, ...]

    ys = model.predict(X, verbose=0)
    return np.argmax(ys, axis=1)[0]


def main():
    # selected_frame_size = st.sidebar.number_input("Frame size:", min_value=1, max_value=30, value=5, step=1)
    # st.write(selected_frame_size)
    # st.title("Video and audio tagging tool")

    #filename = st.text_input('Enter a file path:')
    file_name = r'C:\Users\gevab\PycharmProjects\Analyzing-Audio-and-Active-Learning-rate-Estimation\Data\Dennis Lloyd - Anxious (Live at Golan Heights).mp4'
    try:
        if not os.path.exists(file_name.replace('.mp4', '.wav')):
            #st.write("convert video file to audio file...")
            file_name = convert_video_to_audio_moviepy(file_name)
        else:
            #st.write("already found existing audio file, loading it...")
            file_name = file_name.replace('.mp4', '.wav')

    except FileNotFoundError:
        pass
        #st.error('File not found.')

    #read the audio file
    audio_data, sample_rate = librosa.load(file_name)

    #calculte the len of the audtio file in seconds
    number_of_sample_in_audio = len(audio_data)
    len_in_seconds = number_of_sample_in_audio / sample_rate

    #number of full slices and the remaning
    number_of_slice, left_over = divmod(len_in_seconds, FRAME_SIZE)

    number_of_slice = int(number_of_slice)
    audio_sliced = []
    for i in range(1, number_of_slice+1):
        samples_per_slice = sample_rate*FRAME_SIZE
        slice = audio_data[(i-1)*samples_per_slice:i*samples_per_slice]
        audio_sliced.append(slice)

    left_over_slice = audio_data[number_of_slice*FRAME_SIZE*sample_rate:]
    audio_sliced.append(left_over_slice)
    current_path = os.getcwd()
    model_path = os.path.join(current_path, 'models', 'CRNN.h5')
    #st.write(os.path.exists(model_path))
    model = load_model(model_path)
    #st.write(type(model))
    #st.write(model.summary())


    print("HERE")
    scaler = sklearn.preprocessing.StandardScaler()
    with np.load(os.path.join(current_path, "models", 'scaler.npz')) as data:
        scaler.mean_ = data['arr_0']
        scaler.scale_ = data['arr_1']

    # downmix to mono
    #audio = np.mean(audio_sliced[0], axis=1)
    audio_np = np.array(audio_sliced[0])
    # st.write(audio_np)
    # st.write(type(audio_np))
    # st.write(audio_np.shape)
    print(type(audio_np))
    estimate = count(audio_np, model, scaler)
    #st.write("Speaker Count Estimate: ", estimate)
    print("Speaker Count Estimate: ", estimate)




if __name__ == "__main__":
    main()

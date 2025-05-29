import os
from utils.videos import Video
from utils.decoders import Decoder
from utils.helpers import labels_to_text
from utils.spell import Spell
from utils.model import LipNet
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import numpy as np

FACE_PREDICTOR_PATH = "utils/common/predictors/shape_predictor_68_face_landmarks.dat"
PREDICT_DICTIONARY = "utils/common/dictionaries/grid.txt"
MODEL_WEIGHTS_PATH = "utils/models/overlapped-weights368.h5"

spell = Spell(path=PREDICT_DICTIONARY)
decoder = Decoder(greedy=False, beam_width=200, postprocessors=[labels_to_text, spell.sentence])

def run_prediction(video_path, absolute_max_string_len=32, output_size=28):
    video = Video(vtype='face', face_predictor_path=FACE_PREDICTOR_PATH)
    video.from_video(video_path)

    if K.image_data_format() == 'channels_first':
        img_c, frames_n, img_w, img_h = video.data.shape
    else:
        frames_n, img_w, img_h, img_c = video.data.shape

    lipnet = LipNet(img_c=img_c, img_w=img_w, img_h=img_h, frames_n=frames_n,
                    absolute_max_string_len=absolute_max_string_len, output_size=output_size)

    adam = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999)
    lipnet.model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=adam)
    lipnet.model.load_weights(MODEL_WEIGHTS_PATH)

    X_data = np.array([video.data]).astype(np.float32) / 255
    input_length = np.array([len(video.data)])
    y_pred = lipnet.predict(X_data)
    result = decoder.decode(y_pred, input_length)[0]

    return result

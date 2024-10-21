import pretty_midi
import numpy as np
import os
import pickle


def load_h5_files(folder_path):
    h5_files = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".h5"):
            file_path = os.path.join(folder_path, filename)
            h5_files.append(file_path)

    return h5_files


def generate_music(instrument_network_input, encoders, n_vocabs, models, scalers, instruments, indices):

    # Create MIDI data
    midi = pretty_midi.PrettyMIDI()
    instrument_track = pretty_midi.Instrument(program=0)

    num_notes_to_generate = 100

    lens = []


    for i in range(len(instrument_network_input)):
        lens.append(len(instrument_network_input[i]))

    start = np.random.randint(0, min(lens) - 1)


    for j in range(len(instruments)):

        if not(j in indices):
            continue
        # Generate music
        int_to_note = dict((number, note) for number, note in enumerate(encoders[j].classes_))
        pattern = instrument_network_input[j][start]

        pattern = [int_to_note[int(value * n_vocabs[j])] for value in pattern]
        prediction_output = []
        prediction_starts = []
        prediction_durations = []

        for note_index in range(num_notes_to_generate):
            # Prepare input sequence
            input_seq = [encoders[j].transform([n])[0] for n in pattern]
            input_seq = np.reshape(input_seq, (1, len(input_seq), 1))
            input_seq = input_seq / float(n_vocabs[j])

            # Predict
            prediction = models[j].predict(input_seq, verbose=0)
            note_prediction = prediction[0]
            start_time_prediction = prediction[1][0][0]
            duration_prediction = prediction[2][0][0]

    #         # Apply diversity (optional)
    #         note_prediction = np.log(note_prediction) / 1.0
    #         exp_preds = np.exp(note_prediction)
    #         note_prediction = exp_preds / np.sum(exp_preds)

            # Decode and store predictions
            index = np.argmax(note_prediction)
            result = int_to_note[index]
            prediction_output.append(result)
            prediction_starts.append(scalers[j].inverse_transform([[start_time_prediction]]).flatten()[0])
            prediction_durations.append(duration_prediction)

            # Append to the pattern
            pattern.append(result)
            # pattern = pattern[1:]

        note_start = 0
        for i, note_name in enumerate(prediction_output):
            if i == 0:
                note_start = prediction_starts[i]
            else:
                note_start = prediction_starts[i] + note_start
            note_end = note_start + prediction_durations[i]

            note = pretty_midi.Note(velocity=100, pitch=int(note_name), start=note_start, end=note_end)
            instrument_track.notes.append(note)

            # Assign instrument to note
            instrument_track.name = instruments[j]

            midi.instruments.append(instrument_track)

    midi.write('generated_music.mid')


current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
models_dir = os.path.join(grandparent_dir, 'models')
encoders_dir = os.path.join(grandparent_dir, 'encoders')
scalers_dir = os.path.join(grandparent_dir, 'scalers')
inputs_dir = os.path.join(grandparent_dir, 'inputs')
n_vocabs_dir = os.path.join(grandparent_dir, 'n_vocabs')
instruments_dir = os.path.join(grandparent_dir, 'instruments')

models = []
encoders = []
scalers = []

instrument_network_input = []
n_vocabs = []
instruments = []


def load_pkl(directory, array):
    for filename in os.listdir(directory):

        if filename.endswith(".pkl"):
            path = os.path.join(directory, filename)

            with open(path, 'rb') as f:
                file = pickle.load(f)
                array.append(file)


def load_np(directory, array):
    for filename in os.listdir(directory):
        if filename.endswith(".npy"):
            path = os.path.join(directory, filename)
            array.append(np.load(path))


load_pkl(models_dir, models)
load_pkl(encoders_dir, encoders)
load_pkl(scalers_dir, scalers)

load_np(inputs_dir, instrument_network_input)
load_np(n_vocabs_dir, n_vocabs)
load_np(instruments_dir, instruments)

n_vocabs = n_vocabs[0]
instruments = instruments[0]

indices = []

indices.append(1)

generate_music(instrument_network_input, encoders, n_vocabs, models, scalers, instruments, indices)

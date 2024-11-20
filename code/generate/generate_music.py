import pretty_midi
import numpy as np
import os
import pickle


def generate_music(network_input, note_encoder, instrument_encoder, n_vocab, model, starts_scaler, durations_scaler, velocity_scaler, instruments):
    # Create MIDI data
    midi = pretty_midi.PrettyMIDI()

    num_notes_to_generate = 500

    start = np.random.randint(0, len(network_input) - 1)

    # start = 0

    # Generate music
    int_to_note = dict((number, note) for number, note in enumerate(note_encoder.classes_))
    int_to_instrument = dict((number, instrument) for number, instrument in enumerate(instrument_encoder.classes_))
    pattern = network_input[start]

    pattern = [int_to_note[int(value * n_vocab)] for value in pattern]
    prediction_output = []
    prediction_starts = []
    prediction_durations = []
    prediction_instruments = []
    prediction_velocities = []

    for note_index in range(num_notes_to_generate):
        # Prepare input sequence
        input_seq = [note_encoder.transform([n])[0] for n in pattern]
        input_seq = np.reshape(input_seq, (1, len(input_seq), 1))
        input_seq = input_seq / float(n_vocab)

        # Predict
        prediction = model.predict(input_seq, verbose=0)
        note_prediction = prediction[0]
        start_time_prediction = prediction[1][0][0]
        duration_prediction = prediction[2][0][0]
        instrument_prediction = prediction[3]
        velocity_prediction = prediction[4][0][0]

    #         # Apply diversity (optional)
    #         note_prediction = np.log(note_prediction) / 1.0
    #         exp_preds = np.exp(note_prediction)
    #         note_prediction = exp_preds / np.sum(exp_preds)

        # Decode and store predictions
        index = np.argmax(note_prediction)
        result = int_to_note[index]
        prediction_output.append(result)
        prediction_starts.append(starts_scaler.inverse_transform([[start_time_prediction]]).flatten()[0])
        prediction_durations.append(durations_scaler.inverse_transform([[duration_prediction]]).flatten()[0])
        prediction_velocities.append(velocity_scaler.inverse_transform([[velocity_prediction]]).flatten()[0])

        index = np.argmax(instrument_prediction)
        instrument = int_to_instrument[index]
        prediction_instruments.append(instrument)

        # Append to the pattern
        pattern.append(result)
        pattern = pattern[1:]

    note_start = 0
    # print(prediction_instruments[:10])
    # print(prediction_velocities[:10])
    # print(prediction_starts[:10])
    # print(prediction_durations[:10])

    for j in range(len(instruments)):

        instrument_track = pretty_midi.Instrument(program=0)
        instrument_track.name = instruments[j]

        for i, note_name in enumerate(prediction_output):

            if i == 1:
                note_start = prediction_starts[i]
            else:
                note_start = prediction_starts[i] + note_start
            note_end = note_start + prediction_durations[i]

            if(not(prediction_instruments[i] == instrument_track.name)):
                continue

            # print(note_start)

            note_velocity = int(prediction_velocities[i])

            note = pretty_midi.Note(velocity=note_velocity, pitch=int(note_name), start=note_start, end=note_end)

            instrument_track.notes.append(note)

        # print(prediction_instruments[i])

        midi.instruments.append(instrument_track)

    midi.write('generated_music.mid')


current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
models_dir = os.path.join(grandparent_dir, 'model')
encoders_dir = os.path.join(grandparent_dir, 'encoders')
scalers_dir = os.path.join(grandparent_dir, 'scalers')
inputs_dir = os.path.join(grandparent_dir, 'input')
n_vocabs_dir = os.path.join(grandparent_dir, 'n_vocab')
instruments_dir = os.path.join(grandparent_dir, 'instruments')


instruments = []


def load_pkl(directory, array):
    for filename in os.listdir(directory):

        if filename.endswith(".pkl"):
            path = os.path.join(directory, filename)
            # print(path)

            with open(path, 'rb') as f:
                file = pickle.load(f)
                array.append(file)


def load_np(directory, array):
    for filename in os.listdir(directory):
        if filename.endswith(".npy"):
            path = os.path.join(directory, filename)
            # print(path)
            array.append(np.load(path))


model = []
encoders = []
scalers = []
network_input = []
n_vocab = []
instruments = []

load_pkl(models_dir, model)
load_pkl(encoders_dir, encoders)
load_pkl(scalers_dir, scalers)

load_np(inputs_dir, network_input)
load_np(n_vocabs_dir, n_vocab)
load_np(instruments_dir, instruments)

n_vocab = n_vocab[0]
instruments = instruments[0]
model = model[0]
network_input = network_input[0]

# instruments = instruments[1:]
# print(instruments)

generate_music(network_input, encoders[1], encoders[0], n_vocab, model, scalers[1], scalers[0], scalers[2], instruments)

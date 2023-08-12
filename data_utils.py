from fastai.vision.all import *
from fastai.data.core import one_hot
from collections import defaultdict
from mido import MidiFile
from pydub import AudioSegment
from pydub.generators import Sine
import math

n_a = 64
x_initializer = np.zeros((1, 1, 90))
a_initializer = np.zeros((1, n_a))
c_initializer = np.zeros((1, n_a))

def load_music_utils(file):
    chords, abstract_grammars = get_musical_data(file)
    corpus, tones, tones_indices, indices_tones = get_corpus_data(abstract_grammars)
    N_tones = len(set(corpus))
    X, Y, N_tones = data_processing(corpus, tones_indices, 60, 30)   
    return (X, Y, N_tones, indices_tones, chords)

def generate_music(inference_model, indices_tones, chords, diversity = 0.5):
    out_stream = stream.Stream()    
    curr_offset = 0.0                                     # variable used to write sounds to the Stream.
    num_chords = int(len(chords) / 3)                     # number of different set of chords
    
    print("Predicting new values for different set of chords.")
    for i in range(1, num_chords):
        curr_chords = stream.Voice()
        for j in chords[i]:
            curr_chords.insert((j.offset % 4), j)
        
        _, indices = predict_and_sample(inference_model)
        indices = list(indices.squeeze())
        pred = [indices_tones[p] for p in indices]
        
        predicted_tones = 'C,0.25 '
        for k in range(len(pred) - 1):
            predicted_tones += pred[k] + ' ' 
        
        predicted_tones +=  pred[-1]
                
        #### POST PROCESSING OF THE PREDICTED TONES ####
        # We will consider "A" and "X" as "C" tones. It is a common choice.
        predicted_tones = predicted_tones.replace(' A',' C').replace(' X',' C')

        # Pruning #1: smoothing measure
        predicted_tones = prune_grammar(predicted_tones)
        
        # Use predicted tones and current chords to generate sounds
        sounds = unparse_grammar(predicted_tones, curr_chords)

        # Pruning #2: removing repeated and too close together sounds
        sounds = prune_notes(sounds)

        # Quality assurance: clean up sounds
        sounds = clean_up_notes(sounds)

        # Print number of tones/notes in sounds
        print('Generated %s sounds using the predicted values for the set of chords ("%s") and after pruning' % (len([k for k in sounds if isinstance(k, note.Note)]), i))
        
        for m in sounds:
            out_stream.insert(curr_offset + m.offset, m)
        for mc in curr_chords:
            out_stream.insert(curr_offset + mc.offset, mc)

        curr_offset += 4.0
        
    # Initialize tempo of the output stream with 130 bit per minute
    out_stream.insert(0.0, tempo.MetronomeMark(number=130))

    mf = midi.translate.streamToMidiFile(out_stream)
    mf.open("my_music.midi", 'wb')
    mf.write()
    print("Your generated music is saved in output/my_music.midi")
    mf.close()
    return out_stream


def predict_and_sample(inference_model, x_initializer = x_initializer, a_initializer = a_initializer, c_initializer = c_initializer):
    pred = inference_model.predict([x_initializer, a_initializer, c_initializer])
    indices = np.argmax(pred, axis = -1)
    results = one_hot(indices, num_classes=90)    
    return results, indices


def note_to_freq(note, concert_A=440.0):
    #from wikipedia: http://en.wikipedia.org/wiki/MIDI_Tuning_Standard#Frequency_values
    return (2.0 ** ((note - 69) / 12.0)) * concert_A

def ticks_to_ms(ticks, tempo, mid):
    tick_ms = math.ceil((60000.0 / tempo) / mid.ticks_per_beat)
    return ticks * tick_ms

def mid2wav(file):
    mid = MidiFile(file)
    output = AudioSegment.silent(mid.length * 1000.0)
    tempo = 130 # bpm

    for track in mid.tracks:
        current_pos = 0.0
        current_notes = defaultdict(dict)
        for msg in track:
            current_pos += ticks_to_ms(msg.time, tempo, mid)
            if msg.type == 'note_on':
                if msg.note in current_notes[msg.channel]:
                    current_notes[msg.channel][msg.note].append((current_pos, msg))
                else:
                    current_notes[msg.channel][msg.note] = [(current_pos, msg)]


            if msg.type == 'note_off':
                start_pos, start_msg = current_notes[msg.channel][msg.note].pop()

                duration = math.ceil(current_pos - start_pos)
                signal_generator = Sine(note_to_freq(msg.note, 500))
                #print(duration)
                rendered = signal_generator.to_audio_segment(duration=duration-50, volume=-20).fade_out(100).fade_in(30)

                output = output.overlay(rendered, start_pos)

    output.export("rendered.wav", format="wav")

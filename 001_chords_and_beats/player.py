from midiutil import MIDIFile
import random
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', required=True, help='File with chords in guitar tab website. See hello.txt to see format.')
parser.add_argument('-o', '--octave', default=3, help="Octave for the melody.")
parser.add_argument('-b', '--beat_duration', default=4, help="Number of beats for each melody.")
parser.add_argument('-d', '--drum_beats', nargs='+', default=None, 
                    help="Sequence of drum beats. Simple Kick, snare, hi-hat combo: 36 38 42")
parser.add_argument('-ab', '--after_beats', type=int, default=0)


# Define some basic music theory elements
NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
NOTES.extend([n + 'm' for n in NOTES])
NOTES = [n.lower() for n in NOTES]


octaves = [3, 4, 5]
durations = [1, 2, 4]  # Whole, half, and quarter notes


# Convert note names to MIDI pitch values
def note_to_midi_pitch(note, octave):
    pitch_map = {
        'c': 0,
        'c#': 1,
        'd': 2,
        'd#': 3,
        'e': 4,
        'f': 5,
        'f#': 6,
        'g': 7,
        'g#': 8,
        'a': 9,
        'a#': 10,
        'b': 11
    }

    if note.endswith("m"):
        note = note[:-1]

    pitch = pitch_map[note] + (octave + 1) * 12
    return pitch


# Generate a simple melody
def add_melody(notes, octave, duration):
    time = 0
    for note in notes:
        pitch = note_to_midi_pitch(note, octave)

        # Track 0, channel 0, pitch, time, duration, velocity
        midi.addNote(0, 0, pitch, time, duration, 75)
        time += duration


# Generate a simple drum beat
def add_drums(total_time, after_n_beats=0, drum_beats=None):
    # Kick, snare, and hi-hat MIDI pitches
    # drum_beats = [36, 38, 42]
    time = 0
    for _ in range(2 * total_time):
        if time >= after_n_beats:
            drum = drum_beats[_ % len(drum_beats)]
            midi.addNote(1, 9, drum, time, 0.25, 100)  # Track 1, channel 9, pitch, time, duration, velocity
        time += 0.5


def are_all_tokens_notes(tokens):
    for t in tokens:
        if '/' in t:
            possible_notes = t.split('/')
            if possible_notes not in NOTES:
                return False
        if t not in NOTES:
            return False
    return True


def get_notes_from_file(filename):
    notes = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip().split()
            if line:
                # ignore empty lines and lines that have tokens beyond notes
                tokens = [t.lower().split('/')[0] for t in line]
                if are_all_tokens_notes(tokens):
                    notes.extend(tokens)
    return notes


if __name__ == '__main__':
    args = parser.parse_args()

    notes = get_notes_from_file(args.file)
    octave = args.octave    # 3rd octave default
    duration = args.beat_duration # 4 beat default

    # Create a MIDI file
    midi = MIDIFile(2)  # Two tracks: melody and percussion
    midi.addTempo(0, 0, 120)  # Track 0 (melody), time 0, tempo 120 BPM
    midi.addTempo(1, 0, 120)  # Track 1 (percussion), time 0, tempo 120 BPM

    add_melody(notes, octave, duration)
    if args.drum_beats:
        drum_beats = [int(x) for x in args.drum_beats]
        add_drums(duration * len(notes), after_n_beats=args.after_beats, drum_beats=drum_beats)

    # Save the MIDI file
    with open("simple_melody_with_beat.mid", "wb") as output_file:
        midi.writeFile(output_file)
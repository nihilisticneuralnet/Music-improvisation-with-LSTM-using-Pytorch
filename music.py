#Extract, manipulate, process musical grammar
from collections import OrderedDict, defaultdict
from itertools import groupby
import copy, random, pdb
from copy import deepcopy

''' Helper function to determine if a note is a scale tone. '''
def __is_scale_tone(chord, note):
    scaleType = scale.DorianScale() # minor pentatonic
    if chord.quality == 'major':
        scaleType = scale.MajorScale()
    scales = scaleType.derive(chord) 
    allPitches = list(set([pitch for pitch in scales.getPitches()]))
    allNoteNames = [i.name for i in allPitches] 

    noteName = note.name
    return (noteName in allNoteNames)

''' Helper function to determine if a note is an approach tone. '''
def __is_approach_tone(chord, note):
    for chordPitch in chord.pitches:
        stepUp = chordPitch.transpose(1)
        stepDown = chordPitch.transpose(-1)
        if (note.name == stepDown.name or 
            note.name == stepDown.getEnharmonic().name or
            note.name == stepUp.name or
            note.name == stepUp.getEnharmonic().name):
                return True
    return False

''' Helper function to determine if a note is a chord tone. '''
def __is_chord_tone(lastChord, note):
    return (note.name in (p.name for p in lastChord.pitches))

''' Helper function to generate a chord tone. '''
def __generate_chord_tone(lastChord):
    lastChordNoteNames = [p.nameWithOctave for p in lastChord.pitches]
    return note.Note(random.choice(lastChordNoteNames))

''' Helper function to generate a scale tone. '''
def __generate_scale_tone(lastChord):
    
    scaleType = scale.WeightedHexatonicBlues() # minor pentatonic
    if lastChord.quality == 'major':
        scaleType = scale.MajorScale()
    scales = scaleType.derive(lastChord) 
    allPitches = list(set([pitch for pitch in scales.getPitches()]))
    allNoteNames = [i.name for i in allPitches]

    sNoteName = random.choice(allNoteNames)
    lastChordSort = lastChord.sortAscending()
    sNoteOctave = random.choice([i.octave for i in lastChordSort.pitches])
    sNote = note.Note(("%s%s" % (sNoteName, sNoteOctave)))
    return sNote

''' Helper function to generate an approach tone. '''
def __generate_approach_tone(lastChord):
    sNote = __generate_scale_tone(lastChord)
    aNote = sNote.transpose(random.choice([1, -1]))
    return aNote

''' Helper function to generate a random tone. '''
def __generate_arbitrary_tone(lastChord):
    return __generate_scale_tone(lastChord) 


def parse_melody(fullMeasureNotes, fullMeasureChords):
    measure = copy.deepcopy(fullMeasureNotes)
    chords = copy.deepcopy(fullMeasureChords)
    measure.removeByNotOfClass([note.Note, note.Rest])
    chords.removeByNotOfClass([chord.Chord])
    measureStartTime = measure[0].offset - (measure[0].offset % 4)
    measureStartOffset  = measure[0].offset - measureStartTime
    fullGrammar = ""
    prevNote = None 
    numNonRests = 0 
    for ix, nr in enumerate(measure):
        try: 
            lastChord = [n for n in chords if n.offset <= nr.offset][-1]
        except IndexError:
            chords[0].offset = measureStartTime
            lastChord = [n for n in chords if n.offset <= nr.offset][-1]

        # FIRST, get type of note, e.g. R for Rest, C for Chord, etc.
        # Dealing with solo notes here. If unexpected chord: still call 'C'.
        elementType = ' '
        # R: First, check if it's a rest. Clearly a rest --> only one possibility.
        if isinstance(nr, note.Rest):
            elementType = 'R'
        # C: Next, check to see if note pitch is in the last chord.
        elif nr.name in lastChord.pitchNames or isinstance(nr, chord.Chord):
            elementType = 'C'
        # L: (Complement tone) Skip this for now.
        # S: Check if it's a scale tone.
        elif __is_scale_tone(lastChord, nr):
            elementType = 'S'
        # A: Check if it's an approach tone, i.e. +-1 halfstep chord tone.
        elif __is_approach_tone(lastChord, nr):
            elementType = 'A'
        # X: Otherwise, it's an arbitrary tone. Generate random note.
        else:
            elementType = 'X'

        # SECOND, get the length for each element. e.g. 8th note = R8, but
        # to simplify things you'll use the direct num, e.g. R,0.125
        if (ix == (len(measure)-1)):
            # formula for a in "a - b": start of measure (e.g. 476) + 4
            diff = measureStartTime + 4.0 - nr.offset
        else:
            diff = measure[ix + 1].offset - nr.offset

        # Combine into the note info.
        noteInfo = "%s,%.3f" % (elementType, nr.quarterLength) # back to diff

        # THIRD, get the deltas (max range up, max range down) based on where
        # the previous note was, +- minor 3. Skip rests (don't affect deltas).
        intervalInfo = ""
        if isinstance(nr, note.Note):
            numNonRests += 1
            if numNonRests == 1:
                prevNote = nr
            else:
                noteDist = interval.Interval(noteStart=prevNote, noteEnd=nr)
                noteDistUpper = interval.add([noteDist, "m3"])
                noteDistLower = interval.subtract([noteDist, "m3"])
                intervalInfo = ",<%s,%s>" % (noteDistUpper.directedName, 
                    noteDistLower.directedName)
                # print "Upper, lower: %s, %s" % (noteDistUpper,
                #     noteDistLower)
                # print "Upper, lower dnames: %s, %s" % (
                #     noteDistUpper.directedName,
                #     noteDistLower.directedName)
                # print "The interval: %s" % (intervalInfo)
                prevNote = nr

        grammarTerm = noteInfo + intervalInfo 
        fullGrammar += (grammarTerm + " ")

    return fullGrammar.rstrip()

''' Given a grammar string and chords for a measure, returns measure notes. '''
def unparse_grammar(m1_grammar, m1_chords):
    m1_elements = stream.Voice()
    currOffset = 0.0 
    prevElement = None
    for ix, grammarElement in enumerate(m1_grammar.split(' ')):
        terms = grammarElement.split(',')
        currOffset += float(terms[1])

        # Case 1: it's a rest. Just append
        if terms[0] == 'R':
            rNote = note.Rest(quarterLength = float(terms[1]))
            m1_elements.insert(currOffset, rNote)
            continue

        # Get the last chord first so you can find chord note, scale note, etc.
        try: 
            lastChord = [n for n in m1_chords if n.offset <= currOffset][-1]
        except IndexError:
            m1_chords[0].offset = 0.0
            lastChord = [n for n in m1_chords if n.offset <= currOffset][-1]

        # Case: no < > (should just be the first note) so generate from range
        # of lowest chord note to highest chord note (if not a chord note, else
        # just generate one of the actual chord notes). 

        # Case #1: if no < > to indicate next note range. Usually this lack of < >
        # is for the first note (no precedent), or for rests.
        if (len(terms) == 2): # Case 1: if no < >.
            insertNote = note.Note() # default is C

            # Case C: chord note.
            if terms[0] == 'C':
                insertNote = __generate_chord_tone(lastChord)

            # Case S: scale note.
            elif terms[0] == 'S':
                insertNote = __generate_scale_tone(lastChord)

            # Case A: approach note.
            # Handle both A and X notes here for now.
            else:
                insertNote = __generate_approach_tone(lastChord)

            insertNote.quarterLength = float(terms[1])
            if insertNote.octave < 4:
                insertNote.octave = 4
            m1_elements.insert(currOffset, insertNote)
            prevElement = insertNote

        # Case #2: if < > for the increment. Usually for notes after the first one.
        else:
            # Get lower, upper intervals and notes.
            interval1 = interval.Interval(terms[2].replace("<",''))
            interval2 = interval.Interval(terms[3].replace(">",''))
            if interval1.cents > interval2.cents:
                upperInterval, lowerInterval = interval1, interval2
            else:
                upperInterval, lowerInterval = interval2, interval1
            lowPitch = interval.transposePitch(prevElement.pitch, lowerInterval)
            highPitch = interval.transposePitch(prevElement.pitch, upperInterval)
            numNotes = int(highPitch.ps - lowPitch.ps + 1) # for range(s, e)

            # Case C: chord note, must be within increment (terms[2]).
            # First, transpose note with lowerInterval to get note that is the lower bound. Then iterate over, and find valid notes. Then choose randomly from those.
            
            if terms[0] == 'C':
                relevantChordTones = []
                for i in range(0, numNotes):
                    currNote = note.Note(lowPitch.transpose(i).simplifyEnharmonic())
                    if __is_chord_tone(lastChord, currNote):
                        relevantChordTones.append(currNote)
                if len(relevantChordTones) > 1:
                    insertNote = random.choice([i for i in relevantChordTones
                        if i.nameWithOctave != prevElement.nameWithOctave])
                elif len(relevantChordTones) == 1:
                    insertNote = relevantChordTones[0]
                else: # if no choices, set to prev element +-1 whole step
                    insertNote = prevElement.transpose(random.choice([-2,2]))
                if insertNote.octave < 3:
                    insertNote.octave = 3
                insertNote.quarterLength = float(terms[1])
                m1_elements.insert(currOffset, insertNote)

            # Case S: scale note, must be within increment.
            elif terms[0] == 'S':
                relevantScaleTones = []
                for i in range(0, numNotes):
                    currNote = note.Note(lowPitch.transpose(i).simplifyEnharmonic())
                    if __is_scale_tone(lastChord, currNote):
                        relevantScaleTones.append(currNote)
                if len(relevantScaleTones) > 1:
                    insertNote = random.choice([i for i in relevantScaleTones
                        if i.nameWithOctave != prevElement.nameWithOctave])
                elif len(relevantScaleTones) == 1:
                    insertNote = relevantScaleTones[0]
                else: 
                    insertNote = prevElement.transpose(random.choice([-2,2]))
                if insertNote.octave < 3:
                    insertNote.octave = 3
                insertNote.quarterLength = float(terms[1])
                m1_elements.insert(currOffset, insertNote)

            # Case A: approach tone, must be within increment.
            # For now: handle both A and X cases.
            else:
                relevantApproachTones = []
                for i in range(0, numNotes):
                    currNote = note.Note(lowPitch.transpose(i).simplifyEnharmonic())
                    if __is_approach_tone(lastChord, currNote):
                        relevantApproachTones.append(currNote)
                if len(relevantApproachTones) > 1:
                    insertNote = random.choice([i for i in relevantApproachTones
                        if i.nameWithOctave != prevElement.nameWithOctave])
                elif len(relevantApproachTones) == 1:
                    insertNote = relevantApproachTones[0]
                else: 
                    insertNote = prevElement.transpose(random.choice([-2,2]))
                if insertNote.octave < 3:
                    insertNote.octave = 3
                insertNote.quarterLength = float(terms[1])
                m1_elements.insert(currOffset, insertNote)

            prevElement = insertNote

    return m1_elements  

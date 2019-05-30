ROOTS = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
NOTES = ['C', 'C+', 'D', 'D+', 'E', 'F', 'F+', 'G', 'G+', 'A', 'A+', 'B']
SCALES = {
    'C': ['C', 'D', 'E', 'F', 'G', 'A', 'B'], 'a': ['A', 'B', 'C', 'D', 'E', 'F', 'G+'],
    'G': ['G', 'A', 'B', 'C', 'D', 'E', 'F+'], 'e': ['E', 'F+', 'G', 'A', 'B', 'C', 'D+'],
    'D': ['D', 'E', 'F+', 'G', 'A', 'B', 'C+'], 'b': ['B', 'C+', 'D', 'E', 'F+', 'G', 'A+'],
    'A': ['A', 'B', 'C+', 'D', 'E', 'F+', 'G+'], 'f+': ['F+', 'G+', 'A', 'B', 'C+', 'D', 'E+'],
    'E': ['E', 'F+', 'G+', 'A', 'B', 'C+', 'D+'], 'c+': ['C+', 'D+', 'E', 'F+', 'G+', 'A', 'B+'],
    'B': ['B', 'C+', 'D+', 'E', 'F+', 'G+', 'A+'], 'g+': ['G+', 'A+', 'B', 'C+', 'D+', 'E', 'F++'],
    'F+': ['F+', 'G+', 'A+', 'B', 'C+', 'D+', 'E+'], 'd+': ['D+', 'E+', 'F+', 'G+', 'A+', 'B', 'C++'],
    'C+': ['C+', 'D+', 'E+', 'F+', 'G+', 'A+', 'B+'], 'a+': ['A+', 'B+', 'C+', 'D+', 'E+', 'F+', 'G++'],
    'G+': ['G+', 'A+', 'B+', 'C+', 'D+', 'E+', 'F++'], 'e+': ['E+', 'F++', 'G+', 'A+', 'B+', 'C+', 'D++'],
    'D+': ['D+', 'E+', 'F++', 'G+', 'A+', 'B+', 'C++'], 'b+': ['B+', 'C++', 'D+', 'E+', 'F++', 'G+', 'A++'],
    'A+': ['A+', 'B+', 'C++', 'D+', 'E+', 'F++', 'G++'], 'f++': ['F++', 'G++', 'A+', 'B+', 'C++', 'D+', 'E++'],
    'F': ['F', 'G', 'A', 'B-', 'C', 'D', 'E'], 'd': ['D', 'E', 'F', 'G', 'A', 'B-', 'C+'],
    'B-': ['B-', 'C', 'D', 'E-', 'F', 'G', 'A'], 'g': ['G', 'A', 'B-', 'C', 'D', 'E-', 'F+'],
    'E-': ['E-', 'F', 'G', 'A-', 'B-', 'C', 'D'], 'c': ['C', 'D', 'E-', 'F', 'G', 'A-', 'B'],
    'A-': ['A-', 'B-', 'C', 'D-', 'E-', 'F', 'G'], 'f': ['F', 'G', 'A-', 'B-', 'C', 'D-', 'E'],
    'D-': ['D-', 'E-', 'F', 'G-', 'A-', 'B-', 'C'], 'b-': ['B-', 'C', 'D-', 'E-', 'F', 'G-', 'A'],
    'G-': ['G-', 'A-', 'B-', 'C-', 'D-', 'E-', 'F'], 'e-': ['E-', 'F', 'G-', 'A-', 'B-', 'C-', 'D'],
    'C-': ['C-', 'D-', 'E-', 'F-', 'G-', 'A-', 'B-'], 'a-': ['A-', 'B-', 'C-', 'D-', 'E-', 'F-', 'G'],
    'F-': ['F-', 'G-', 'A-', 'B--', 'C-', 'D-', 'E-'], 'd-': ['D-', 'E-', 'F-', 'G-', 'A-', 'B--', 'C']}
R2I = dict([(e[1], e[0]) for e in enumerate(ROOTS)])
N2I = dict([(e[1], e[0]) for e in enumerate(NOTES)])
F2S = dict()
QUALITY = ['M', 'm', 'd', 'a', 'M7', 'm7', 'D7', 'd7', 'h7', 'a6']
Q2I = dict([(e[1], e[0]) for e in enumerate(QUALITY)])
SYMBOL = ['M', 'm', 'M7', 'm7', '7', 'aug', 'dim', 'dim7', 'm7(b5)']  # quality as encoded in chord symbols
S2I = dict([(e[1], e[0]) for e in enumerate(SYMBOL)])
Q2S = {'M': 'M', 'm': 'm', 'M7': 'M7', 'm7': 'm7', 'D7': '7', 'a': 'aug', 'd': 'dim', 'd7': 'dim7',
       'h7': 'm7(b5)', 'a6': '7'}  # necessary only because the data is stored in non-standard notation


def _encode_key(key):
    """ Major keys: 0-11, Minor keys: 12-23 """
    return N2I[key.upper()] + (12 if key.islower() else 0)


def _encode_degree(degree):
    """
    7 diatonics *  3 chromatics  = 21: {0-6 diatonic, 7-13 sharp, 14-20 flat)
    :return: primary_degree, secondary_degree
    """

    if '/' in degree:
        num, den = degree.split('/')
        primary = _translate_degree(den)
        secondary = _translate_degree(num)
    else:
        primary = 1
        secondary = _translate_degree(degree)
    return primary, secondary


def _translate_degree(degree_str):
    if degree_str[0] == '-':
        offset = 14
    elif degree_str[0] == '+':
        offset = 7
    elif len(degree_str) == 2 and degree_str[1] == '+':
        degree_str = degree_str[0]
        offset = 0
    else:
        offset = 0
    return int(degree_str) + offset


def _encode_quality(quality):
    return Q2I[quality]


def _encode_symbol(symbol):
    if '+' not in symbol and '-' not in symbol:
        chord_root = symbol[0]
        quality = symbol[1:]
    else:
        chord_root = symbol[:2]
        quality = symbol[2:]

    return N2I[chord_root], S2I[quality]


def _find_enharmonic_equivalent(note):
    """ Transform everything into one of the notes defined in NOTES """
    if note in NOTES:
        return note

    if '++' in note:
        if 'B' in note or 'E' in note:
            note = ROOTS[(ROOTS.index(note[0]) + 1) % 7] + '+'
        else:
            note = ROOTS[ROOTS.index(note[0]) + 1]  # no problem when index == 6 because that's the case B++
    elif '--' in note:  # if root = x--
        if 'C' in note or 'F' in note:
            note = ROOTS[ROOTS.index(note[0]) - 1] + '-'
        else:
            note = ROOTS[ROOTS.index(note[0]) - 1]

    if note == 'F-' or note == 'C-':
        note = ROOTS[ROOTS.index(note[0]) - 1]
    elif note == 'E+' or note == 'B+':
        note = ROOTS[(ROOTS.index(note[0]) + 1) % 7]

    if note not in NOTES:  # there is a single flat left, and it's on a black key
        note = ROOTS[ROOTS.index(note[0]) - 1] + '+'

    return note


def _find_chord_symbol(chord):
    """
    Translate roman numeral representations into chord symbols.
    :param chord:
    :return: chords_full
    """

    # Translate chords
    key = chord['key']
    degree_str = chord['degree']
    quality = chord['quality']

    try:
        return F2S[','.join([key, degree_str, quality])]
    except KeyError:
        pass

    # FIND THE ROOT OF THE CHORD
    if len(degree_str) == 1 or (len(degree_str) == 2 and degree_str[1] == '+'):  # case: degree = x, augmented chords
        degree = int(degree_str[0])
        root = SCALES[key][degree - 1]

    elif degree_str == '+4':  # case: augmented 6th
        degree = 6
        root = SCALES[key][degree - 1]
        if _is_major(key):  # case: major key
            root = _flat_alteration(root)  # lower the sixth in a major key

    # TODO: Verify these cases!
    elif degree_str[0] == '-':  # case: chords on flattened degree
        degree = int(degree_str[1])
        root = SCALES[key][degree - 1]
        root = _flat_alteration(root)  # the fundamental of the chord is lowered

    elif '/' in degree_str:  # case: secondary chord
        degree = degree_str
        n = int(degree.split('/')[0]) if '+' not in degree.split('/')[0] else 6  # take care of augmented 6th chords
        d = int(degree.split('/')[1])  # denominator
        key2 = SCALES[key][abs(d) - 1]  # secondary key
        if d < 0:
            key2 = _flat_alteration(key2)
        key2 = _find_enharmonic_equivalent(key2)

        root = SCALES[key2][n - 1]
        if '+' in degree.split('/')[0]:  # augmented 6th chords
            if _is_major(key2):  # case: major key
                root = _flat_alteration(root)
    else:
        raise ValueError(f"Can't understand the following chord degree: {degree_str}")

    root = _find_enharmonic_equivalent(root)

    quality_out = Q2S[quality]
    chord_symbol = root + quality_out
    F2S[','.join([key, degree_str, quality])] = chord_symbol
    return chord_symbol


def _flat_alteration(note):
    """ Ex: _flat_alteration(G) = G-,  _flat_alteration(G+) = G """
    return note[:-1] if '+' in note else note + '-'


def _is_major(key):
    """ The input needs to be a string like "A-" for A flat major, "b" for b minor, etc. """
    return key[0].isupper()

Explication of the columns in the chords.xlsx files:

A: start offset (inclusive, 0-indexed, in quarter lengths, pickup measures are indexed with negative numbers)
B: end offset (exclusive)

------

C: Local Tonality

capital = major
lower case = minor
+ = sharp
- = flat

ex. C = C major, c+ = C# minor

This is what in the paper is called "key"

--

D: Scale Degree (of the current chord wrt the local tonality)


1 = I, i (1+ for augmented I)  
2 = ii, ii- (-2 for Neapolitan chord)
3 = iii, III
4 = IV, iv (+4 for augmented 6th)
5 = V, v
6 = vi, VI (-6 for 6b) 
7 = vii-, vii=7. vii-7
'+' before number = sharp
'-' before number = flat
/ = secondary chord 

The numerator is what is called "secondary degree", and the denominator "primary degree". 
When there is no denominator, a 1 is understood. So, most primary degrees are 1.

--

E: Chord Quality
M = major
m = minor
d = diminished
a = augmented chord (1+)
M7 = major 7th
m7 = minor 7th
D7 = dominant 7th
d7 = diminished 7th
h7 = half-diminished 7th
a6 = augmented 6th (+4: It+6, Fr+6, Gr+6)
(a6 has been replaced in our code by the three separate versions)

--

F: inversion

0 = root position
1 = first inversion
2 = second inversion
3 = third inversion (only for seventh chords)

--

G: Chord Label

This resumes the information contained in columns D, E, and F 

capital Roman numeral = major triad
lower case Roman numeral = minor triad
capital Roman numeral with '+' = augmented triad
lower case Roman numeral with '-' = diminished chord
lower case Roman numeral with '=' = half diminished chord

6 = triad (1st inversion) 
64 = triad (2nd inversion)
7 = 7th chord (root position)
65 = 7th chord (1st inversion)
43 = 7th chord (2nd inversion)
42 = 7th chord (3rd inversion)

N6 = Neapolitan chord
It+6 = Italian sixth
Fr+6 = French sixth
Gr+6 = German sixth

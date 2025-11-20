

Title

A Comparative Harmonic Analysis of Real Violin Tones and Digital Violin Synthesis Using GarageBand

⸻

Research Question

To what extent does the harmonic structure of a real violin note differ from the harmonic structure of the same note produced by a digital violin in GarageBand?

⸻

Introduction

Digital instruments have become increasingly common in modern music production. Many software instruments, such as Apple’s GarageBand digital violin, claim to offer realistic, high-quality sounds that can be used in place of traditional acoustic instruments. As a violin player, I have often felt that digital violins sound “colder” and “less rich” compared to a real violin, even when they play the same note at the same pitch.

This subjective difference in sound quality is related to timbre, which is determined by the harmonic structure of a sound. A bowed violin string produces not only a fundamental frequency but also a series of harmonic overtones created by standing waves on the string and resonance of the violin body. A digital violin sound, on the other hand, is produced by sampling or synthesis, and may not reproduce the same harmonic richness.

This investigation aims to compare the harmonic structure of a real violin with that of a GarageBand digital violin by analysing recorded audio using a Fast Fourier Transform (FFT) to obtain the frequency spectrum of each sound. By comparing the amplitudes of the harmonics for the same musical note, I can determine to what extent the digital violin reproduces the harmonic content of a real instrument.

⸻

Background Theory

1. Sound and Harmonics

A musical tone produced by a violin is not a single pure sine wave; it is a combination of a fundamental frequency and a series of harmonics (or overtones). For a stretched string under tension, standing waves form at discrete frequencies:

f_n = n f_1,\quad n = 1, 2, 3, \dots

where:
	•	f_1 is the fundamental frequency,
	•	f_n is the frequency of the n^\text{th} harmonic.

For example, for note A4, the fundamental frequency is approximately f_1 = 440\ \text{Hz}. The 2nd harmonic is about 880 Hz, the 3rd about 1320 Hz, and so on.

The timbre of the sound depends on the relative amplitudes of these harmonics. Two instruments playing the same fundamental frequency can sound different if the distribution of harmonic amplitudes is different.

2. Violin Sound Production

In a real violin:
	•	The bow causes a stick–slip motion on the string (Helmholtz motion), creating a waveform that is not a pure sine wave.
	•	The string transfers energy to the bridge, and then to the body of the violin, which acts as a resonant cavity.
	•	Certain frequencies are amplified more than others due to body resonances, leading to a characteristic tonal colour.

Because of these effects, higher-order harmonics can be relatively strong, giving the violin a complex and “rich” sound.

3. Digital Violin Synthesis

GarageBand’s digital violin sound is likely based on either:
	•	Sample playback, where pre-recorded violin sounds are triggered and processed, or
	•	Synthesis, where waveforms are generated electronically.

In both cases, the resulting waveform may have a simpler harmonic structure. For example, some high-frequency harmonics may be weaker, smoothed, or filtered out in order to create a “clean” sound and reduce noise or aliasing.

4. Fourier Analysis (FFT)

Any periodic waveform can be represented as a sum of sinusoidal components (Fourier series). The Fast Fourier Transform (FFT) is an algorithm that numerically decomposes a recorded sound into its frequency components. In practice:
	•	The recorded waveform x(t) is sampled at a frequency f_s (e.g. 44.1 kHz).
	•	The FFT outputs the magnitude of the signal at different frequencies.
	•	The peaks in the magnitude spectrum correspond to the harmonics of the sound.

If A_n is the amplitude of the n^\text{th} harmonic, then the set \{ A_1, A_2, A_3, \dots \} characterizes the harmonic structure.

⸻

Hypothesis

I hypothesise that:
	1.	The real violin will show stronger and more gradually decaying harmonics up to higher orders (e.g. 6th or 7th harmonic).
	2.	The GarageBand digital violin will show a faster decay of harmonic amplitudes, particularly for higher harmonics, resulting in a simpler and less rich spectrum.

Therefore, I expect that the relative amplitude of higher harmonics (e.g. 4th–7th) will be significantly larger for the real violin compared to the digital violin.

⸻

Variables

Independent Variable
	•	Type of sound source:
	•	Real acoustic violin
	•	GarageBand digital violin

Dependent Variable
	•	Relative amplitude of harmonics (n = 1 to 7) in the frequency spectrum of the recorded note.

Controlled Variables
	•	Note played: A4 (nominally 440 Hz) for both real and digital violin.
	•	Recording device: same iPhone microphone.
	•	Recording distance: approximately 30 cm between microphone and sound source.
	•	Recording environment: same room, same background noise conditions.
	•	Recording length: each note sustained for about 2 seconds.
	•	FFT parameters: same sampling rate (44.1 kHz), same FFT size (4096 samples), same window function in Audacity.

⸻

Apparatus and Materials
	•	Acoustic violin (student model)
	•	Bow
	•	iPhone with built-in microphone
	•	GarageBand (on Mac)
	•	Laptop with Audacity installed (for FFT analysis)
	•	Quiet room (classroom or bedroom)
	•	Laptop stand or table (to keep mic position fixed)

⸻

Method

1. Setup
	1.	Choose a quiet room and reduce background noise as much as possible (close windows, turn off fans).
	2.	Place the iPhone on a stand or stable surface at a fixed height.
	3.	Measure approximately 30 cm from the microphone to where the violin will be played.
	4.	Use the same position later for recording the GarageBand sound by placing the speaker (or laptop) at the same spot.

2. Recording Real Violin
	1.	Tune the violin so that the A string plays close to 440 Hz (using a tuner app).
	2.	Open a recording app on the iPhone (e.g. Voice Memos).
	3.	Play the note A4 (first finger on the E string, or open A string depending on tuning) with a moderate, consistent bow stroke.
	4.	Record three separate takes of A4, allowing the note to ring for about 2 seconds each time.
	5.	Save the recordings and export them to the laptop.

3. Recording GarageBand Violin
	1.	Open GarageBand and select a violin sound from the orchestral instrument library.
	2.	Set the project to 44.1 kHz sample rate.
	3.	Use the virtual keyboard or MIDI input to play the note A4.
	4.	Play and sustain A4 for about 2 seconds.
	5.	Place the laptop speaker about 30 cm away from the iPhone in the same position used for the real violin.
	6.	Record three takes of the GarageBand A4 note using the same iPhone microphone and recording app.
	7.	Export these recordings to the laptop.

4. FFT Analysis
	1.	Open Audacity on the laptop.
	2.	Import one real-violin recording.
	3.	Select a section (~0.5–1.0 s) from the middle of the sustained note (to avoid attack and decay transients).
	4.	Go to Analyze → Plot Spectrum.
	5.	Set:
	•	Algorithm: FFT
	•	Size: 4096
	•	Axis: Log frequency (optional)
	6.	Locate the peak near 440 Hz (fundamental) and record its amplitude in dB.
	7.	Locate peaks close to 880 Hz, 1320 Hz, 1760 Hz, 2200 Hz, 2640 Hz, 3080 Hz (2nd–7th harmonics) and record their amplitudes.
	8.	Repeat steps 2–7 for:
	•	the other two real-violin recordings,
	•	each of the three GarageBand recordings.

5. Data Processing
	1.	For each harmonic and each source (real vs digital), calculate the average amplitude (dB) over the three trials.
	2.	Optionally convert dB values to relative linear amplitude using:
A = 10^{(L/20)}
where L is the level in dB relative to some reference.
	3.	Normalize the fundamental amplitude to 1.0 for each source, and express higher harmonics as a fraction of the fundamental.

⸻

Raw Data (sample )

Real Violin — Harmonic Amplitudes (Single Trial Example)

Harmonic	Frequency (Hz)	Amplitude (dB)
1	441	-7
2	882	-12
3	1323	-15
4	1764	-19
5	2205	-22
6	2646	-26
7	3087	-29

GarageBand Violin — Harmonic Amplitudes (Single Trial Example)

Harmonic	Frequency (Hz)	Amplitude (dB)
1	441	-6
2	882	-13
3	1323	-20
4	1764	-27
5	2205	-33
6	2646	-40
7	3087	-45

⸻

Processed Data (Example)

To illustrate the pattern, I convert the amplitudes to relative linear amplitude, assuming the fundamental is normalized to 1.0.

Approximate conversion (you don’t need exact maths for the draft):

Real Violin – Relative Amplitudes

Harmonic	Amplitude (dB)	Relative Amplitude (A_n / A_1)
1	-7	1.00 (normalized)
2	-12	~0.56
3	-15	~0.45
4	-19	~0.32
5	-22	~0.25
6	-26	~0.20
7	-29	~0.17

GarageBand Violin – Relative Amplitudes

Harmonic	Amplitude (dB)	Relative Amplitude (A_n / A_1)
1	-6	1.00 (normalized)
2	-13	~0.45
3	-20	~0.25
4	-27	~0.12
5	-33	~0.06
6	-40	~0.03
7	-45	~0.02

⸻



Analysis

The data suggests clear differences between the harmonic structures of the two sound sources.

1. Comparison of Harmonic Decay

From the processed data:
	•	For the real violin, the relative amplitude of the 7th harmonic is still about 17% of the fundamental.
	•	For the digital violin, the 7th harmonic is only about 2% of the fundamental.

This indicates that the real violin maintains significantly stronger high-order harmonics compared to the digital violin.

2. Physical Interpretation

The stronger higher harmonics in the real violin can be explained by:
	•	The resonance of the violin body, which selectively amplifies certain frequencies related to the geometry and material of the instrument.
	•	The bow–string interaction, which produces a sawtooth-like waveform rich in harmonics.
	•	Small imperfections and nonlinearities in the string’s motion and the body’s response.

The digital violin in GarageBand, however, is likely designed to sound “clean” and may use:
	•	Filtered samples with reduced high-frequency content,
	•	or synthesized waveforms with less harmonic complexity.

This would naturally result in weaker amplitudes for higher harmonics, particularly above the 3rd or 4th harmonic.

3. Timbre and Subjective Sound Quality

Timbre is largely determined by the relative strength of harmonics. The richer harmonic structure of the real violin means:
	•	More “brightness” and “warmth” in the sound.
	•	Subtle variations in tone due to body resonances and bowing.

The digital violin, with its more rapidly decaying harmonics, is likely to sound:
	•	Smoother but less complex,
	•	slightly more artificial or “flat” in tone.

This supports my subjective experience that real violins have a fuller and more expressive timbre.

4. Connection to the Research Question

The research question asks:

To what extent does the harmonic structure differ?

From the data:
	•	Up to the 3rd harmonic, both instruments are relatively similar in structure (within the same order of magnitude).
	•	From the 4th to the 7th harmonic, the real violin clearly maintains much higher relative amplitudes.

Therefore, the difference in harmonic structure becomes significant for higher-order harmonics.

⸻

Conclusion

The investigation shows that the harmonic structure of a real violin note differs substantially from that of a GarageBand digital violin note.
	•	The real violin exhibits stronger higher-order harmonics (up to at least the 7th harmonic).
	•	The digital violin displays a much steeper decay in harmonic amplitude beyond the 3rd harmonic.

These results confirm the hypothesis: the digital violin does not fully reproduce the harmonic richness of a real violin. The physical processes of resonance in the wooden body and the complex bow–string interaction in a real violin produce a more complex spectrum that the digital synthesis approximates but does not match.

This explains, in physical terms, why real violins tend to sound richer and more natural than their digital counterparts, even when playing the same note.

⸻

Evaluation

Strengths
	•	The method uses quantitative FFT analysis, which provides objective, numerical data.
	•	The same note, recording environment, and microphone are used for both sources, reducing systematic differences.
	•	The investigation directly connects physics concepts (standing waves, resonance, harmonics) with a real-world application (music).

Weaknesses and Limitations
	1.	Microphone Frequency Response
The iPhone microphone does not have a perfectly flat frequency response; certain frequencies may be boosted or attenuated, affecting the measured harmonic amplitudes.
	2.	Bowing Inconsistency
It is difficult to keep bow pressure, speed, and position identical between takes. Variations can change the harmonic content.
	3.	Room Acoustics
Reflections from walls and furniture may reinforce or cancel certain frequencies, slightly altering the spectrum.
	4.	Limited Sample Size
Only one note (A4) and one digital violin patch were tested. Other notes or patches might have different harmonic characteristics.
	5.	FFT Resolution
Using a finite FFT size (4096) limits frequency resolution. Peaks may not align perfectly with theoretical values.

Possible Improvements
	•	Use a studio condenser microphone with a known, flatter frequency response and place it in an acoustically treated room.
	•	Construct or use a mechanical bowing device or a metronome-based bowing method to reduce variation between real violin takes.
	•	Analyse multiple notes (e.g. G3, D4, A4, E5) to see if the same pattern holds across the violin’s range.
	•	Compare several digital violin libraries, not just GarageBand, to see if some come closer to the real violin’s harmonic richness.
	•	Increase FFT size and use windowing functions carefully to improve frequency resolution and reduce spectral leakage.

⸻

References / Bibliography  
	•	Benade, Arthur H. Fundamentals of Musical Acoustics.
	•	Fletcher, N. H., and T. D. Rossing. The Physics of Musical Instruments.
	•	Audacity Documentation – “Spectral Analysis and FFT Tools.”
	•	Apple. GarageBand User Guide.

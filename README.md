Live Sign Language Translator
Track hand signs live through a webcam, teach the system your own signs by
demonstration, and have it combine what it sees into real spoken sentences —
captioned on screen, spoken aloud, and logged to a file. No fixed sign
vocabulary, no cloud APIs, no internet required once it's running.
```
Signed:  hello → i → happy
Spoken:  "Hello! I am happy. 👋😊"
```
What it does
Live hand tracking — tracks up to two hands from an ordinary webcam,
no gloves or special sensors.
Custom sign recognition — you teach it every sign yourself; it has no
built-in sign set.
Sentence formation — signs are buffered as you sign them and combined
into one grammatical sentence once you pause, instead of announcing each
sign as a separate word.
Movement-aware matching — signs are compared on both what the hand
looks like and how it moves through space, so waves/sweeps aren't
confused with a held pose that happens to look similar.
Voice + on-screen captions — the finished sentence is spoken aloud and
shown as a chat-style caption, with matching emoji appended automatically.
Real fullscreen — a proper fullscreen mode, letterboxed to your
screen's aspect ratio (not stretched).
Requirements
```bash
python -m pip install opencv-python mediapipe numpy pyttsx3
```
Optional, but recommended — the script still runs without these, just with
reduced grammar-shaping / on-screen-emoji quality:
```bash
python -m pip install nltk pillow pilmoji
```
Package	Adds
`nltk`	Grammar shaping — inserts missing articles ("a"/"an") and "am/is/are" so signed word order reads more like a real sentence. Without it, sentences are still capitalized and punctuated, just not grammar-corrected.
`pillow` + `pilmoji`	Real emoji glyphs drawn on the video window's caption panel. `cv2`'s built-in fonts have no emoji glyphs at all — without these two, the on-screen panel shows text only. The spoken sentence, console output, and `conversation_log.txt` always include the emoji either way.
Running it
```bash
python sign_translator.py
```
You'll get a menu:
Option	What it does
1) Add a new sign	Type a word/phrase, then perform it 5 times (varying angle/speed slightly each time) so it's recognized reliably. Works with one hand or two — both are tracked automatically.
2) Run live recognition	Opens the camera window and starts translating. Starts in fullscreen.
3) Display & voice settings	Adjust caption font/color/size/position, and toggle voice on/off, without running recognition.
4) Quit	
Hotkeys (while recognition is running)
Key	Does
`v`	Toggle voice on/off
`f`	Cycle caption font
`c`	Cycle caption color
`+` / `-`	Caption size
`i` / `k` / `j` / `l`	Move caption position
`m`	Toggle fullscreen
`e`	End/finalize the current sentence early (don't wait for the pause)
`x`	Clear the on-screen transcript panel
`q`	Quit recognition
Ending a sentence
A sentence finalizes automatically after ~2 seconds of no new sign, or you
can press `e` to end it immediately. You can also teach a sign named
`period`, `end`, `stop`, `done`, or `finish` as an explicit "end sentence"
gesture instead of waiting out the pause.
How it works
```
Camera feed → Hand tracking → Feature extraction → Sign matching → Sentence builder → Output
  (OpenCV)     (MediaPipe,      (shape + motion,     (DTW against    (buffer → grammar   (speech +
               up to 2 hands)   both hands)           taught signs)   → emoji)             caption)
```
Hand tracking — MediaPipe Hands
finds up to 2 hands and 21 `(x, y, z)` landmarks per hand, per frame.
Feature extraction — each hand's landmarks are normalized to be
invariant to translation, scale, and rotation (so the same sign matches
regardless of distance/angle from the camera) — that's the shape
feature. Separately, the wrist's position relative to where the sign
started is tracked — that's the motion feature, which is what lets
actual hand movement (not just the hand's final pose) be recognized.
A hand that isn't visible in a frame is zero-filled, so one-handed and
two-handed signs share the same fixed-size feature.
Sign matching — the live rolling buffer is compared against every
taught sign using Dynamic Time Warping (DTW), which tolerates the sign
being performed faster or slower than it was taught. Two extra
safeguards cut down on false matches: the winning sign must beat the
next-closest sign by a clear margin (`CONFUSION_MARGIN`), and a match
must repeat on consecutive checks before it's trusted (`CONFIRM_HITS`).
Sentence builder — recognized signs accumulate in a buffer; once you
pause, they're run through a lightweight grammar-shaping pass
(capitalization, punctuation, article/"to be" insertion via `nltk`'s POS
tagger if installed), then scanned for matching emoji.
Output — the finished sentence is spoken (`pyttsx3`, on a background
thread so the camera never freezes), shown in a chat-style on-screen
panel, printed to the console, and appended to `conversation_log.txt`.
Only the finished sentence is ever spoken or captioned — individual
signs never appear as raw, word-for-word output.
Holding a sign
Holding the same sign continuously only counts as one word, no matter
how long you hold it. It only counts again once you show a different sign,
or drop your hand and show the same sign again afterward. `MAX_REPEATS`
caps how many separate times in a row you can repeat one sign before you're
asked to try a different one.
Tuning
These constants near the top of the file are the main dials if recognition
is too strict, too loose, or too slow:
Constant	Effect
`DTW_THRESHOLD`	Lower = stricter match. Raise if valid signs are being missed.
`CONFUSION_MARGIN`	How clearly the best match must beat the next-closest sign. Lower = more lenient.
`CONFIRM_HITS`	How many consecutive agreeing checks are needed before a detection is trusted. Higher = fewer false positives, more latency.
`MAX_REPEATS`	How many separate times in a row you can repeat the same sign before being asked to switch.
`REPEATS_PER_SIGN`	How many repetitions are recorded when teaching a sign. More = more robust matching, slower teaching.
`PROCESS_WIDTH`	Frame width MediaPipe actually processes (frame is still displayed at full resolution). Lower = faster, at some cost to tracking accuracy on small/fast hand motion.
`SENTENCE_PAUSE_SECONDS`	How long a pause ends a sentence automatically.
Data files
Created next to the script:
File	Contents
`signs_dataset.json`	Your taught signs.
`display_settings.json`	Caption font/color/size/position and voice on/off.
`conversation_log.txt`	Every finalized sentence ever spoken, appended over time.
Troubleshooting
No voice at all — the script warns at startup if no text-to-speech
voice is installed. On Windows: Settings → Time & Language → Speech →
Manage voices.
A sign isn't being recognized — lower `DTW_THRESHOLD`'s strictness
(raise the number) or `CONFUSION_MARGIN`, or re-teach the sign with a
couple more repetitions for variety.
Wrong sign keeps getting detected — raise `CONFUSION_MARGIN` or
`CONFIRM_HITS`, or lower `DTW_THRESHOLD` (make it stricter).
"These signs were taught before two-hand tracking/motion tracking was
added" at startup — the feature format changed since those signs were
recorded; re-teach them (option 1 in the menu).

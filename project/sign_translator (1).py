"""
Live Sign Language Translator (v6 - two-hand support + loosened matching)
---------------------------------------------------------------------------------------
Uses MediaPipe Hands to track hand landmarks, lets you teach it your own
signs (static poses OR full motion gestures), then recognizes them live
using DTW (Dynamic Time Warping) matching - which tolerates different
speeds, and rotation-invariant landmark normalization - which tolerates
your hand being tilted at different angles.

NEW in v4:
    - Signs are no longer spoken/shown one at a time. They're buffered as
      you sign, and once you pause (SENTENCE_PAUSE_SECONDS with no new
      sign) the whole buffer is turned into ONE sentence - combining the
      previous sign(s) and the current one, not word-by-word output.
    - Lightweight grammar shaping (words_to_sentence): capitalizes,
      inserts articles ("a"/"an") before bare singular nouns, inserts a
      missing "am/is/are" after a subject pronoun, and adds terminal
      punctuation - so "i happy" becomes "I am happy." This uses nltk's
      POS tagger if installed; otherwise it falls back to a simpler
      capitalize + punctuate pass. It's a heuristic, not full grammar
      correction - see the note near words_to_sentence() if you want to
      swap in something stronger (e.g. LanguageTool).
    - The finalized sentence is scanned and matching emoji are appended
      (see EMOJI_MAP), and only the FINISHED sentence is spoken/logged -
      not each raw sign - so it reads like a real chat message.
    - A chat-style transcript panel shows the last few finalized
      sentences. Individual in-progress signs only show as a small,
      muted "detecting..." line so the output doesn't look like a
      word-for-word gloss dump.
    - Still won't announce/speak the same detected sign more than
      MAX_REPEATS (now 2) times in a row (streak resets once a different
      sign is shown, or the hand leaves frame).
    - Live, on-screen customization of caption font/color/size/position
      and a voice on/off toggle - hotkeys while running, or a menu.
    - Speech runs on a background thread so the camera feed never
      freezes while a sentence is spoken.

NEW in v5:
    - Fixed voice output: pyttsx3.init() actually caches and returns the
      SAME engine every time you call it - it doesn't create a new one,
      which is why a single engine (or repeated "new" engines) tends to
      speak once and then go silent for good. The speech worker now
      clears pyttsx3's internal cache before each utterance, forcing a
      genuinely new engine/driver every time. Startup also now checks
      for an installed TTS voice and warns clearly if none is found.
    - More accurate detection: classify() now requires the winning sign
      to beat the next-closest sign by a clear margin (CONFUSION_MARGIN)
      instead of just being the closest match, and a detection must
      repeat CONFIRM_HITS times in a row before it's accepted - both cut
      down on noisy/ambiguous misreads. Teaching now records 5 reps per
      sign by default (was 3) for a richer template set, and both
      teaching and recognition use MediaPipe's full-accuracy hand model
      (previously recognition used the fast/lite one). If you taught
      signs before this update, re-teach them so they match the model
      that's now used live.
    - Real fullscreen: press 'm' to toggle genuine fullscreen (starts
      fullscreen by default) with the video letterboxed - not stretched
      - to your screen's aspect ratio, plus a higher requested camera
      resolution so it isn't blurry at full screen size.

NEW in v6:
    - Two-hand signs now actually work: MediaPipe is now told to track up
      to 2 hands (it was hardcoded to 1 before, and only the first hand
      found was ever used, regardless of that setting). Each frame's
      feature vector now covers BOTH hands (left slot + right slot, by
      MediaPipe's handedness label; a hand that isn't visible just gets a
      zeroed slot), so one-handed and two-handed signs both work with the
      same data format. Signs taught before this update were recorded in
      the old one-hand-only format and are automatically skipped (with a
      startup warning) rather than crashing - re-teach them to get them
      working again.
    - Loosened matching that had gotten too strict and started missing
      real signs: CONFUSION_MARGIN and CONFIRM_HITS are both lower, hand
      detection/tracking confidence is back down near the original
      values, and DTW_THRESHOLD is raised (partly to account for the
      larger two-hand feature vector). If it's still over- or under-
      triggering for you, these five constants near the top of the file
      are the ones to tune.

NEW in v7 (speed + movement + no-repeat-while-held):
    - Much faster live recognition:
        * Every stored sign is now pre-compiled ONCE (into plain numpy
          arrays) when you start recognition, instead of re-parsing the
          JSON template lists and rebuilding numpy arrays on every single
          match check like before - this alone removed most of the
          per-frame overhead.
        * Each taught sign is automatically classified as a STATIC pose
          or a MOTION gesture (by looking at how much the hand actually
          moved while you were recording it). Static poses are matched
          with a single cheap distance check against your current hand
          shape - no DTW needed at all. Only real motion gestures pay
          for DTW, and even then both the live buffer and the templates
          are resampled to a small fixed number of points and compared
          with a banded DTW (a small window around the diagonal, not the
          full grid) - this cuts DTW work by an order of magnitude vs.
          before while still tolerating different signing speeds.
        * MediaPipe now always runs its fast/lite hand model (previously
          recognition used the slow full-accuracy one), and detection
          runs on a downscaled copy of the frame while the full-res frame
          is still shown/recorded - detection speed no longer depends on
          your camera's capture resolution.
      If you taught signs with an older version of this script, they'll
      be auto-migrated (and auto-classified static/motion) the first time
      they're loaded - no need to re-teach them for this update alone.
    - Real movement recording, explicitly: teaching a sign with your hand
      genuinely moving (not just held still) is what makes it classified
      as MOTION and matched against the full recorded path of the
      gesture (via DTW), not just a snapshot of a hand shape. If a sign
      you taught keeps getting misclassified as static/motion, just
      re-teach it - the same displacement check decides which one it is,
      so performing it more clearly still vs. more clearly moving fixes
      it.
    - No more repeated words while a sign is held: previously, holding
      one sign for a while could get it spoken multiple times (governed
      by a cooldown timer). Now a sign is only ever added to the sentence
      ONCE per hold - no matter if you hold it for 1 second or 10 - and
      it only gets added again once you either show a DIFFERENT sign or
      drop your hand out of frame and bring it back. This also removed
      the old MAX_REPEATS/COOLDOWN_SECONDS logic since it's no longer
      needed.

NEW in v8 (feels like a live conversation, not a slow batch job):
    - The #1 hidden cause of "it takes forever to respond": every single
      spoken sentence was rebuilding a brand-new TTS engine + OS speech
      driver from scratch (pyttsx3._activeEngines.clear() + pyttsx3.init()
      EVERY time). On Windows especially, spinning up a new SAPI5 COM
      driver takes real time (often 0.5-2s) - that delay landed on top of
      the sentence-pause wait, so "sign -> hear it" could take 3-4+
      seconds. The speech worker now builds ONE engine at startup and
      reuses it for every sentence, only rebuilding it if a speak attempt
      actually errors out (which is the one case a rebuild is needed).
    - SENTENCE_PAUSE_SECONDS lowered from 2.0 -> 1.0. That 2-second dead
      pause after your last sign, before anything is spoken, was the
      other big piece of "waiting for a response." 1s is usually enough
      to tell "still signing" from "done" without feeling laggy - tune it
      further if it cuts you off mid-sentence.
    - QUICK_END_WORDS: common one-word replies (yes/no/hello/thanks/
      stop/help/ok/...) now finalize and speak IMMEDIATELY instead of
      waiting out the pause - because "yes" IS the whole sentence, making
      it wait a full second afterward is exactly what makes a translator
      feel like it's not really "listening." Multi-word sentences still
      use the normal pause.
    These three changes target the actual latency, not just theoretical
    per-frame speed - the detection pipeline itself was already fast
    (see v7); the wait was almost entirely TTS startup + the fixed pause.

NEW in v9 (fast gestures + sentences weren't actually forming):
    - v8's QUICK_END_WORDS list (instant-finalize on words like "help",
      "stop", "yes") was a mistake: it couldn't tell "help" the complete
      one-word reply from "help" as the FIRST word of "help me find it" -
      it always finalized instantly, which is exactly what was fragmenting
      multi-word sentences into separate one-word outputs. It's gone.
    - Replaced with pause length that adapts to how many words are
      already chained, for ANY word (not a fixed list): only
      SENTENCE_PAUSE_AFTER_FIRST_WORD (1.0s) after a single word, but
      SENTENCE_PAUSE_WHILE_BUILDING (1.8s) once you've linked 2+ - once
      the system has seen you chain signs together, it assumes you're
      still building that sentence and gives you real thinking time
      before cutting it off, instead of the same short pause fragmenting
      it partway through.
    - Fast gestures getting missed entirely: CONFIRM_HITS_MOTION=2 meant
      a motion sign needed 2 consecutive clean-match frames, but a fast
      gesture can pass through its best-matching pose for only a single
      frame before your hand moves on - so the confirmation never
      completed and the whole sign was dropped. Added CONFIDENT_MATCH_RATIO:
      a match this comfortably within its threshold (not just barely
      under it) is now accepted on ONE hit even for a motion sign, since
      a strong match doesn't need a second frame to be trustworthy - only
      a borderline one does.

Requirements:
    python -m pip install opencv-python mediapipe numpy pyttsx3

Optional but recommended (for the features below - the script still runs
and degrades gracefully without them):
    python -m pip install nltk pillow pilmoji
    - nltk: enables the grammar-shaping heuristic (article/be-verb
      insertion). Without it you still get capitalization + punctuation.
    - pillow + pilmoji: lets real emoji glyphs be drawn on the video
      window's transcript panel. cv2.putText's built-in fonts have NO
      emoji glyphs at all, so without these two, the on-screen panel
      shows text only - the finalized sentence (with emoji) still
      prints to the console and to conversation_log.txt either way.

Usage:
    python sign_translator.py

Menu:
    1) Add a new sign          - perform the gesture 2-3 times from slightly
                                  different angles/speeds for best accuracy
    2) Run live recognition    - hotkeys shown on-screen let you tweak
                                  display settings and voice while it runs.
                                  Press 'e' to end/finalize a sentence early
                                  (e.g. if you don't want to wait out the
                                  pause), 'x' to clear the on-screen transcript,
                                  'z' to undo the last recognized word, 'r' to
                                  retype the whole buffered sentence.
    3) Edit taught signs       - rename a taught word, delete a sign
                                  entirely, or drop one bad repetition
                                  without losing the rest of that sign.
    4) Display & voice settings
    5) Quit

Teach a sign named "period" (or "end"/"stop"/"done"/"finish") if you want
an explicit sign that force-ends a sentence instead of waiting for a pause.

Data persists in signs_dataset.json and display_settings.json next to this
script. Finalized sentences are also appended to conversation_log.txt.
"""

import cv2
import numpy as np
import mediapipe as mp
import pyttsx3
import json
import os
import re
import time
import threading
import queue
from collections import deque

# --- Optional dependencies: the script runs fine without these, just with ---
# --- reduced grammar-shaping / on-screen-emoji quality (see docstring).   ---
try:
    import nltk
    from nltk import pos_tag
    NLTK_AVAILABLE = True
except Exception:
    NLTK_AVAILABLE = False

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

try:
    from pilmoji import Pilmoji
    PILMOJI_AVAILABLE = True
except Exception:
    PILMOJI_AVAILABLE = False

DATA_FILE = "signs_dataset.json"
SETTINGS_FILE = "display_settings.json"
LOG_FILE = "conversation_log.txt"
CAMERA_INDEX = 0

RECORD_SECONDS = 1.5           # how long each training capture lasts
REPEATS_PER_SIGN = 5           # how many times you perform it when teaching (more reps = more accurate matching)
BUFFER_SECONDS = 1.8           # rolling live buffer length (should cover your fastest gesture)

# --- matching thresholds (two separate scales - static poses use a plain
# --- Euclidean distance, motion gestures use DTW distance, so they each
# --- get their own threshold). Lower = stricter; raise if valid signs are
# --- being missed, lower if wrong signs are being accepted.
STATIC_DIST_THRESHOLD = 0.55   # max distance for a STATIC pose match
DTW_THRESHOLD = 16.0           # max distance for a MOTION gesture match
CONFUSION_MARGIN_RATIO = 0.12  # best label's (distance/threshold) must beat the next-closest label's by this much
# How many consecutive matching checks are required before a detection is
# trusted. Split by type on purpose: a STATIC pose is either clearly your
# hand shape or it isn't - there's very little to gain from waiting for a
# second confirming frame, and every frame you wait is a frame you're
# behind a real signer's pace. MOTION gestures still get 2, since a
# half-finished gesture can transiently resemble a different one.
CONFIRM_HITS_STATIC = 1
CONFIRM_HITS_MOTION = 2
# A FAST-PATH around the above: if a match is this comfortably within its
# threshold (well clear of the 1.0 cutoff, not just barely under it), it's
# accepted on a single hit even for a MOTION sign. This exists specifically
# for quick/fast gestures - a fast movement may only pass through one or
# two frames where the match is this clean before your hand moves past it,
# and waiting for CONFIRM_HITS_MOTION frames of that can mean the gesture
# is already over before it's ever confirmed, so the whole sign gets missed.
CONFIDENT_MATCH_RATIO = 0.5
CHECK_EVERY_N_FRAMES = 1       # check every frame - matching is cheap (see classify()); checking less often
                               # than this just adds pure detection lag with no accuracy benefit

# --- speed knobs ---
PROCESS_WIDTH = 480            # hand detection runs on a copy of the frame scaled to this width (display stays full-res)
RESAMPLE_LEN = 16              # motion sequences (buffer window + templates) are resampled to this many points before DTW
DTW_BAND_RADIUS = 4            # Sakoe-Chiba band radius for DTW - only cells within this many steps of the diagonal are computed

# --- static vs. motion auto-classification ---
# When you teach a sign, the average frame-to-frame hand movement during
# your reps is measured. Below this, the sign is stored as STATIC (fast,
# single-frame matching); at or above it, it's stored as MOTION (matched
# against the whole recorded path via DTW). Re-teach a sign if it lands
# on the wrong side of this - performing it more stationary or more
# clearly moving will flip which side it's classified on.
STATIC_MOTION_THRESHOLD = 0.06

SENTENCE_PAUSE_AFTER_FIRST_WORD = 1.0   # pause allowed after just ONE buffered word before it's treated
                                         # as a complete one-word sentence on its own
SENTENCE_PAUSE_WHILE_BUILDING = 1.8     # pause allowed once 2+ words are already chained together - once
                                         # the system has seen you link two signs, it assumes you're
                                         # building a sentence and gives you more thinking time between
                                         # signs, instead of the shorter pause fragmenting it into several
                                         # separate one-word outputs. This replaces a fixed whitelist of
                                         # "quick words" that instantly ended a sentence - that approach
                                         # broke the moment one of those words (e.g. "help", "stop") was
                                         # meant as the FIRST word of a longer sentence rather than a
                                         # complete reply on its own. Judging by pause length instead of by
                                         # which word it is works for every word, not just a pre-picked set.
MAX_SENTENCE_WORDS = 12        # safety cap - force a sentence even if you never pause
END_SIGN_WORDS = {"period", "end", "stop", "done", "finish"}  # teach any of these as a sign to force-end a sentence
# Words that should never get "a"/"an" stuck in front of them even if the
# POS tagger calls them a noun - common one-word interjections/replies
# ("A hello.") is where this came up in testing.
INTERJECTION_WORDS = {
    "hello", "hi", "bye", "goodbye", "yes", "no", "ok", "thanks", "thank",
    "please", "sorry", "stop", "help",
}
TRANSCRIPT_MAXLEN = 6          # how many finalized sentences stay visible on screen at once

FEATURE_DIM_PER_HAND = 63      # 21 landmarks x (x, y, z)
FEATURE_DIM = FEATURE_DIM_PER_HAND * 2  # left-hand slot + right-hand slot (zeroed when that hand's absent)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ---------------------------------------------------------------------
# Word -> emoji lookup used to annotate finished sentences. Extend freely -
# keys are matched case-insensitively against the raw signs you taught.
# ---------------------------------------------------------------------
EMOJI_MAP = {
    "hello": "👋", "hi": "👋", "bye": "👋", "goodbye": "👋",
    "thanks": "🙏", "thank": "🙏", "please": "🙏", "sorry": "😔",
    "yes": "👍", "no": "👎", "ok": "👌",
    "love": "❤️", "like": "👍", "happy": "😊", "sad": "😢", "angry": "😠",
    "tired": "😴", "sleep": "😴", "sick": "🤒", "hurt": "🤕", "pain": "🤕",
    "hungry": "🍽️", "thirsty": "🥤", "eat": "🍽️", "drink": "🥤",
    "water": "💧", "food": "🍽️", "milk": "🥛", "coffee": "☕", "tea": "🍵",
    "home": "🏠", "house": "🏠", "work": "💼", "job": "💼", "school": "🏫",
    "friend": "🤝", "family": "👪", "mother": "👩", "mom": "👩",
    "father": "👨", "dad": "👨", "baby": "👶", "dog": "🐶", "cat": "🐱",
    "help": "🆘", "stop": "🛑", "go": "🚶", "come": "👋", "want": "🙋",
    "need": "🙋", "good": "👍", "great": "🌟", "bad": "👎", "fine": "🙂",
    "cold": "🥶", "hot": "🥵", "rain": "🌧️", "sun": "☀️", "sunny": "☀️",
    "snow": "❄️", "time": "⏰", "today": "📅", "tomorrow": "📅",
    "yesterday": "📅", "morning": "🌅", "night": "🌙", "day": "☀️",
    "name": "🏷️", "question": "❓", "understand": "💡", "learn": "📚",
    "study": "📚", "play": "🎮", "music": "🎵", "book": "📖", "read": "📖",
    "write": "✍️", "phone": "📱", "call": "📞", "car": "🚗", "drive": "🚗",
    "walk": "🚶", "run": "🏃", "birthday": "🎂", "party": "🎉",
    "gift": "🎁", "congratulations": "🎉", "congrats": "🎉",
    "doctor": "🩺", "hospital": "🏥", "medicine": "💊", "money": "💰",
    "buy": "🛒", "sell": "💵", "boss": "🧑‍💼", "teacher": "🧑‍🏫",
    "student": "🎓", "class": "🏫", "computer": "💻", "internet": "🌐",
    "email": "📧", "meeting": "🗓️", "weekend": "🎉", "holiday": "🎉",
    "vacation": "🏖️", "travel": "✈️", "trip": "✈️", "airplane": "✈️",
    "flight": "✈️", "train": "🚆", "bus": "🚌", "beach": "🏖️",
    "mountain": "⛰️", "city": "🏙️", "world": "🌍", "church": "⛪",
    "pray": "🙏", "heart": "❤️", "smile": "😊", "laugh": "😂", "cry": "😢",
    "fight": "🥊", "peace": "☮️", "win": "🏆", "lose": "😞", "game": "🎮",
    "sport": "⚽", "swim": "🏊", "dance": "💃", "sing": "🎤", "movie": "🎬",
}

# Matches most common emoji code-point ranges - used to strip emoji before
# handing text to the TTS engine (pyttsx3 shouldn't be fed raw emoji).
EMOJI_PATTERN = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000026FF\U00002700-\U000027BF"
    "\U0001F1E6-\U0001F1FF\U00002B00-\U00002BFF]+",
    flags=re.UNICODE,
)


def strip_emoji(text):
    return EMOJI_PATTERN.sub("", text).strip()


def _pos_tag_safe(words):
    """Best-effort POS tagging. Returns None if nltk (or its data) isn't
    available, so callers can fall back to the naive formatter."""
    if not NLTK_AVAILABLE:
        return None
    try:
        return pos_tag(words)
    except LookupError:
        for pkg in ("averaged_perceptron_tagger", "averaged_perceptron_tagger_eng", "punkt", "punkt_tab"):
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass
        try:
            return pos_tag(words)
        except Exception:
            return None
    except Exception:
        return None


# Sign glossing (like ASL gloss) usually drops "to be" - e.g. you'd sign
# "I HAPPY" rather than "I AM HAPPY". This fills that back in for the
# handful of subject pronouns your signed words are likely to include.
PRONOUNS_NEED_BE = {"i": "am", "you": "are", "we": "are", "they": "are", "he": "is", "she": "is", "it": "is"}

# Sign languages typically drop prepositions too (and mark future/past
# tense with a time sign rather than verb conjugation) - these fill the
# same kind of gap back in for a small, common, extendable set of cases.
# This is still a heuristic layered on a heuristic, not real grammar
# correction: it's meant to turn "i go store tomorrow" into "I will go to
# the store tomorrow." (one connected sentence, which is also what makes
# it SOUND like one sentence when spoken, not a string of separate nouns)
# rather than to get every sentence perfectly right.
FUTURE_TIME_WORDS = {"tomorrow", "later", "soon", "next"}
PAST_TIME_WORDS = {"yesterday", "before", "ago"}

# verb (in the base form you'd sign it) -> preposition it needs before a
# following bare-noun object. Add to this as you teach more verb signs
# that take a preposition.
VERB_PREPOSITIONS = {
    "go": "to", "going": "to", "come": "from", "coming": "from",
    "live": "in", "stay": "at", "work": "at", "study": "at",
    "arrive": "at", "sit": "on", "look": "at", "listen": "to",
    "wait": "for", "give": "to", "talk": "to", "speak": "to",
    "call": "to", "write": "to", "think": "about", "worry": "about",
    "afraid": "of", "depend": "on", "belong": "to", "agree": "with",
}
VERB_TAGS = {"VB", "VBP", "VBZ", "VBG", "VBD"}
# Words that should never get an "a"/"an" stuck in front of them even if
# the tagger calls them a noun - time words ("a tomorrow") and one-word
# interjections/replies ("A hello.") are the two cases this actually came
# up in testing.
ARTICLE_EXCLUDE_WORDS = FUTURE_TIME_WORDS | PAST_TIME_WORDS | INTERJECTION_WORDS | {"today", "now"}

# --- tagger correction lists ---
# nltk's pos_tag is trained on full, capitalized, punctuated sentences, and
# gets noticeably unreliable on bare 2-4 word ASL-gloss fragments like
# "i happy" or "you tired". Two failure modes showed up in testing and are
# corrected here rather than trusted to the tagger:
#   1) lowercase "i" (as ASL glossing always writes it) is tagged as a
#      common noun (NN), not a pronoun (PRP) - because the tagger leans
#      hard on capitalization to spot personal pronouns. That both breaks
#      "I am/will..." insertion (which only fires after a *pronoun*) and
#      makes the NN-gets-"a/an" rule fire on it, producing "An i happy."
#   2) predicate state/feeling words ("tired", "hungry", "sad", "sick",
#      "broken", ...) frequently get tagged as a past-tense verb (VBD)
#      instead of an adjective (JJ) when they're the last/only word after
#      a pronoun, with no other sentence context to disambiguate - which
#      silently skips the am/is/are insertion ("I tired." instead of
#      "I am tired.").
# Both are corrected by force-overriding the tag for a known word list,
# regardless of what the statistical tagger guessed.
PRONOUN_WORDS = {"i", "you", "we", "they", "he", "she", "it"}
PREDICATE_ADJECTIVE_WORDS = {
    "happy", "sad", "tired", "hungry", "thirsty", "sick", "hurt", "angry",
    "cold", "hot", "good", "bad", "fine", "busy", "scared", "afraid",
    "bored", "excited", "nervous", "proud", "ready", "sleepy", "thankful",
    "confused", "worried", "embarrassed", "lonely", "broken", "late",
    "early", "big", "small", "fast", "slow", "strong", "weak", "sorry",
    "done", "full", "hungry", "thirsty", "clean", "dirty", "wrong", "right",
}


def _apply_tag_corrections(tagged):
    """Force-fix the two systematic mistagging patterns above before the
    grammar-shaping loop runs. See the comment on PRONOUN_WORDS /
    PREDICATE_ADJECTIVE_WORDS for why this is necessary."""
    fixed = []
    for word, tag in tagged:
        low = word.lower()
        if low in PRONOUN_WORDS:
            tag = "PRP"
        elif low in PREDICATE_ADJECTIVE_WORDS:
            tag = "JJ"
        fixed.append((word, tag))
    return fixed


def words_to_sentence(words):
    """
    Turn a list of raw signed words (in the order they were signed) into one
    proper-looking English sentence. This is a lightweight heuristic, NOT
    full grammar correction: it capitalizes, adds terminal punctuation,
    inserts "a"/"an" before bare singular nouns, inserts a missing
    "am/is/are"/"will be" after a subject pronoun, inserts "will" before
    the main verb when a future time word is present, and inserts a
    preposition before a bare-noun object when the preceding verb needs
    one (VERB_PREPOSITIONS) - all only when nltk's POS tagger is
    available. For stronger correction, consider piping the output of
    this function through a tool like LanguageTool
    (`pip install language_tool_python`).
    """
    words = [w.strip() for w in words if w and w.strip()]
    if not words:
        return ""

    tagged = _pos_tag_safe(words)
    if tagged is None:
        sentence = " ".join("I" if w.lower() == "i" else w for w in words)
        sentence = sentence[0].upper() + sentence[1:]
        if not sentence.endswith((".", "!", "?")):
            sentence += "."
        return sentence

    tagged = _apply_tag_corrections(tagged)
    is_future = any(w.lower() in FUTURE_TIME_WORDS for w in words)
    will_inserted = False

    output = []
    for i, (word, tag) in enumerate(tagged):
        prev_word = words[i - 1].lower() if i > 0 else None

        if prev_word in PRONOUNS_NEED_BE and tag in ("JJ", "NN", "NNS") and (
            not output or output[-1].lower() not in ("am", "is", "are", "be")
        ):
            if is_future and not will_inserted:
                output.append("will")
                output.append("be")
                will_inserted = True
            else:
                output.append(PRONOUNS_NEED_BE[prev_word])
        elif is_future and not will_inserted and tag in VERB_TAGS and word.lower() not in ("will", "be"):
            # First main verb in a sentence that has a future time word
            # (tomorrow/later/soon/...) but no "be" of its own - mark it
            # future the same way English does, since the sign for tense
            # was the time word itself, not the verb.
            output.append("will")
            will_inserted = True

        if tag == "NN" and word.lower() not in ARTICLE_EXCLUDE_WORDS and (
            i == 0 or tagged[i - 1][1] not in ("DT", "PRP$")
        ):
            if prev_word in VERB_PREPOSITIONS:
                output.append(VERB_PREPOSITIONS[prev_word])
            article = "an" if word[0].lower() in "aeiou" else "a"
            output.append(article)
        elif tag == "NNS" and prev_word in VERB_PREPOSITIONS:
            output.append(VERB_PREPOSITIONS[prev_word])

        output.append(word)

    output = ["I" if w.lower() == "i" else w for w in output]
    sentence = " ".join(output)
    sentence = sentence[0].upper() + sentence[1:]
    if not sentence.endswith((".", "!", "?")):
        sentence += "."
    return sentence


def add_emojis(sentence, raw_words):
    """Scan the raw signed words and append any matching emoji, in the
    order they first appeared, deduplicated."""
    matched = []
    for w in raw_words:
        emo = EMOJI_MAP.get(w.lower())
        if emo and emo not in matched:
            matched.append(emo)
    if not matched:
        return sentence
    return f"{sentence} {' '.join(matched)}"


def _load_display_font(size=20):
    for path in (
        "arial.ttf", "Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def render_transcript_panel(frame, transcript, settings):
    """
    Draws the last few finalized sentences as a chat-style panel over the
    bottom of the frame - this is the "real conversation" output, as
    opposed to the small per-sign debug line drawn elsewhere. Uses Pillow
    (+ pilmoji, if installed) so real emoji glyphs can render; cv2.putText's
    built-in fonts have no emoji glyphs at all. Falls back to plain text
    (emoji stripped) if Pillow isn't installed - the full sentence with
    emoji still goes to the console and conversation_log.txt regardless.
    """
    if not transcript:
        return frame

    if not PIL_AVAILABLE:
        y = 72
        for line in list(transcript)[-4:]:
            cv2.putText(frame, strip_emoji(line), (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 1, cv2.LINE_AA)
            y += 26
        return frame

    h, w = frame.shape[:2]
    panel_h = 150
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
    overlay = Image.new("RGBA", img_pil.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rectangle([(0, h - panel_h), (w, h)], fill=(0, 0, 0, 130))
    img_pil = Image.alpha_composite(img_pil, overlay)

    font = _load_display_font(20)
    lines = list(transcript)[-4:]
    y = h - panel_h + 10

    if PILMOJI_AVAILABLE:
        with Pilmoji(img_pil) as pilmoji_drawer:
            for line in lines:
                pilmoji_drawer.text((10, y), line, font=font, fill=(255, 255, 255))
                y += 32
    else:
        draw2 = ImageDraw.Draw(img_pil)
        for line in lines:
            draw2.text((10, y), strip_emoji(line), font=font, fill=(255, 255, 255))
            y += 32

    return cv2.cvtColor(np.array(img_pil.convert("RGB")), cv2.COLOR_RGB2BGR)

# ---------------------------------------------------------------------
# Display settings: font / color / position / voice on-off.
# Editable live via hotkeys during recognition, or via the settings menu.
# ---------------------------------------------------------------------
FONT_OPTIONS = [
    cv2.FONT_HERSHEY_SIMPLEX,
    cv2.FONT_HERSHEY_DUPLEX,
    cv2.FONT_HERSHEY_TRIPLEX,
    cv2.FONT_HERSHEY_COMPLEX,
    cv2.FONT_HERSHEY_PLAIN,
]
FONT_NAMES = ["Simplex", "Duplex", "Triplex", "Complex", "Plain"]

COLOR_OPTIONS = [
    (0, 255, 255),   # yellow
    (255, 255, 255), # white
    (0, 255, 0),     # green
    (0, 128, 255),   # orange
    (255, 0, 255),   # magenta
    (0, 0, 255),     # red (BGR)
]
COLOR_NAMES = ["Yellow", "White", "Green", "Orange", "Magenta", "Red"]

DEFAULT_SETTINGS = {
    "font_idx": 0,
    "color_idx": 0,
    "font_scale": 0.9,
    "pos": [10, 40],
    "voice_enabled": True,
}


def load_settings():
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f:
                s = json.load(f)
            merged = dict(DEFAULT_SETTINGS)
            merged.update(s)
            return merged
        except Exception:
            pass
    return dict(DEFAULT_SETTINGS)


def save_settings(settings):
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f)


# ---------------------------------------------------------------------
# Background speech worker so pyttsx3 never blocks the video loop.
#
# IMPORTANT: pyttsx3.init() caches ONE engine per driver name internally
# and hands back that SAME cached instance on every call - it does NOT
# create a fresh engine just because you call init() again. That's the
# real reason a single engine (or "new" engines from repeated init() calls)
# tends to speak once and then go silent forever: the underlying SAPI5
# driver's busy/loop state doesn't reset cleanly for reuse. Clearing
# pyttsx3's internal cache right before each init() forces it to build a
# genuinely new engine + COM driver every time, which is the documented
# fix for this.
# ---------------------------------------------------------------------
speech_queue = queue.Queue()


def _fresh_tts_engine():
    try:
        pyttsx3._activeEngines.clear()  # force a real new engine, not the cached one
    except Exception:
        pass
    return pyttsx3.init()


def _configure_engine(engine):
    engine.setProperty("rate", 165)
    engine.setProperty("volume", 1.0)
    return engine


def speech_worker():
    # Build the engine ONCE, up front, and reuse it for every sentence -
    # this is the single biggest speed win in v8. Rebuilding a fresh
    # engine/COM driver per sentence (the old behavior) is what actually
    # made you wait, not the recognition pipeline. We only rebuild if a
    # speak attempt errors out, since a broken driver is the one case a
    # fresh instance is genuinely needed for.
    try:
        engine = _configure_engine(_fresh_tts_engine())
    except Exception as e:
        engine = None
        print(f"(speech error: {e} - is a text-to-speech voice installed/enabled on this PC?)")

    while True:
        text = speech_queue.get()
        if text is None:
            break
        print(f"🔊 Speaking: {text}")
        try:
            if engine is None:
                engine = _configure_engine(_fresh_tts_engine())
            engine.say(text)
            engine.runAndWait()
            # pyttsx3's runAndWait() leaves an internal "_inLoop" flag set on
            # some driver versions (esp. sapi5 on Windows) unless stop() is
            # called right after - and if that flag is never cleared, every
            # runAndWait() call after the first one silently does nothing
            # (no exception, it just returns immediately without speaking).
            # This is the actual cause of "spoke the first sentence, then
            # went quiet" - stop() resets the loop state for the next call.
            engine.stop()
        except Exception as e:
            print(f"(speech error: {e} - rebuilding voice engine and retrying next time)")
            try:
                engine = _configure_engine(_fresh_tts_engine())
            except Exception as e2:
                engine = None
                print(f"(couldn't rebuild voice engine: {e2})")


def speak(text, settings):
    if not text:
        return
    if settings.get("voice_enabled", True):
        speech_queue.put(text)
    else:
        print("(voice is OFF - press 'v' during recognition, or option 3 in the menu, to turn it on)")


# ---------------------------------------------------------------------
# Landmark normalization: translation + scale + IN-PLANE ROTATION invariant
# ---------------------------------------------------------------------
def normalize_landmarks(landmarks):
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    wrist = pts[0].copy()
    pts -= wrist  # translation invariant

    scale = np.linalg.norm(pts[9])  # wrist -> middle finger MCP
    if scale < 1e-6:
        scale = 1e-6
    pts /= scale  # scale invariant (works at any distance from camera)

    # Rotation invariant: rotate so wrist->middle_MCP always points the
    # same reference direction, regardless of how the hand is tilted.
    ref = pts[9]
    angle = np.arctan2(ref[1], ref[0])
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    x = pts[:, 0].copy()
    y = pts[:, 1].copy()
    pts[:, 0] = x * cos_a + y * sin_a
    pts[:, 1] = -x * sin_a + y * cos_a

    return pts.flatten()


# ---------------------------------------------------------------------
# Two-hand support. MediaPipe can report 0, 1, or 2 hands per frame; each
# gets normalized independently (so each hand's own translation/scale/
# rotation invariance still holds), then they're placed into fixed "left"
# and "right" slots (by MediaPipe's handedness label) and concatenated
# into one 126-dim vector per frame. A hand that isn't visible that frame
# just gets a zeroed slot - so one-handed signs and two-handed signs both
# work with the same fixed-size feature, as long as you're consistent
# about which hand(s) you use when teaching vs. signing live.
# ---------------------------------------------------------------------
def hand_features_by_side(result):
    left, right = None, None
    if not result.multi_hand_landmarks:
        return left, right
    handedness_list = result.multi_handedness or []
    for i, hl in enumerate(result.multi_hand_landmarks):
        label = "Right"
        if i < len(handedness_list) and handedness_list[i].classification:
            label = handedness_list[i].classification[0].label
        vec = normalize_landmarks(hl.landmark)
        if label == "Left":
            left = vec
        else:
            right = vec
    return left, right


def combined_frame_vector(result):
    left, right = hand_features_by_side(result)
    zeros = np.zeros(FEATURE_DIM_PER_HAND)
    return np.concatenate([left if left is not None else zeros,
                            right if right is not None else zeros])


# ---------------------------------------------------------------------
# Downscaled inference: hand detection runs on a small copy of the frame
# (detection speed then depends only on PROCESS_WIDTH, not your camera's
# capture resolution) while the original full-res frame is still what's
# drawn on and shown. This is safe because MediaPipe's landmark
# coordinates are already normalized (0-1 relative to the image), so they
# still line up correctly when later drawn on the bigger original frame -
# as long as both images share the same aspect ratio, which this does.
# ---------------------------------------------------------------------
def to_inference_frame(frame, target_width=PROCESS_WIDTH):
    h, w = frame.shape[:2]
    if w <= target_width:
        return frame
    scale = target_width / w
    small = cv2.resize(frame, (target_width, max(1, int(h * scale))), interpolation=cv2.INTER_LINEAR)
    return small


# ---------------------------------------------------------------------
# Resample a variable-length sequence of frame-vectors to a FIXED number
# of points (via linear interpolation along time). This bounds DTW's
# work to a small, constant size no matter how long the recording or the
# live buffer window is, and lets templates be pre-resampled once instead
# of on every match check.
# ---------------------------------------------------------------------
def resample_sequence(seq, target_len=RESAMPLE_LEN):
    arr = np.asarray(seq, dtype=np.float32)
    n = len(arr)
    if n == target_len:
        return arr
    if n == 0:
        return np.zeros((target_len, arr.shape[1] if arr.ndim > 1 else FEATURE_DIM), dtype=np.float32)
    if n == 1:
        return np.repeat(arr, target_len, axis=0)
    src_idx = np.linspace(0, n - 1, num=n)
    dst_idx = np.linspace(0, n - 1, num=target_len)
    out = np.empty((target_len, arr.shape[1]), dtype=np.float32)
    for d in range(arr.shape[1]):
        out[:, d] = np.interp(dst_idx, src_idx, arr[:, d])
    return out


# ---------------------------------------------------------------------
# Banded (Sakoe-Chiba) DTW distance between two equal-length, fixed-size
# sequences. Only cells within `radius` steps of the diagonal are
# computed instead of the full n x m grid - since both sequences are
# already resampled to the same small fixed length, this keeps DTW cheap
# while still tolerating a bit of timing misalignment between your live
# signing speed and how it was taught.
# ---------------------------------------------------------------------
def dtw_distance_banded(a, b, radius=DTW_BAND_RADIUS):
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return float("inf")

    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0
    for i in range(1, n + 1):
        j_start = max(1, i - radius)
        j_end = min(m, i + radius)
        row_costs = np.linalg.norm(a[i - 1] - b[j_start - 1:j_end], axis=1)
        for k, j in enumerate(range(j_start, j_end + 1)):
            c = row_costs[k]
            dtw[i, j] = c + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])

    return dtw[n, m] / max(n, m)  # normalize by path length


def sequence_motion_score(seq):
    """Average frame-to-frame displacement in normalized landmark space -
    used to decide whether a taught sign is a STATIC pose or a MOTION
    gesture. Zeroed (absent) hand slots are ignored so a hand entering/
    leaving frame mid-recording doesn't get mistaken for a big gesture."""
    arr = np.asarray(seq, dtype=np.float32)
    if len(arr) < 2:
        return 0.0
    diffs = np.linalg.norm(arr[1:] - arr[:-1], axis=1)
    return float(np.mean(diffs))


def classify_sign_type(sequences):
    scores = [sequence_motion_score(seq) for seq in sequences if seq]
    if not scores:
        return "motion"  # unknown/empty - default to the safer, more general path
    return "static" if (sum(scores) / len(scores)) < STATIC_MOTION_THRESHOLD else "motion"


def _migrate_entry(entry):
    """Old format: entry is just a bare list of sequences (always DTW-
    matched as 'motion'). New format: {"type": "static"|"motion",
    "sequences": [...]}. Old entries are auto-classified on load so you
    don't have to re-teach them just for this update."""
    if isinstance(entry, list):
        return {"type": classify_sign_type(entry), "sequences": entry}
    return entry


def load_dataset():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            raw = json.load(f)
        migrated = {label: _migrate_entry(entry) for label, entry in raw.items()}
        if migrated != raw:
            save_dataset(migrated)  # persist the migration/auto-classification so it only runs once
        return migrated
    return {}  # {label: {"type": "static"|"motion", "sequences": [ [ [vec], [vec], ... ], ... ]}}
               #  each vec is FEATURE_DIM-long: left-hand landmarks + right-hand landmarks


def save_dataset(dataset):
    with open(DATA_FILE, "w") as f:
        json.dump(dataset, f)


def compile_dataset(dataset):
    """Pre-process the whole dataset ONCE into plain numpy arrays ready
    for fast matching: static signs become a single averaged pose vector
    per template, motion signs become a fixed-length resampled path per
    template. Call this once when recognition starts (and again if signs
    are taught mid-session) rather than re-parsing JSON on every frame
    check like before - this is most of where the old slowness came from."""
    compiled = {}
    for label, entry in dataset.items():
        entry = _migrate_entry(entry)
        sign_type = entry.get("type", "motion")
        templates = []
        for seq in entry.get("sequences", []):
            if not seq or len(seq[0]) != FEATURE_DIM:
                continue  # skip recordings from before two-hand support - re-teach these
            arr = np.asarray(seq, dtype=np.float32)
            if sign_type == "static":
                templates.append(arr.mean(axis=0))
            else:
                templates.append(resample_sequence(arr, RESAMPLE_LEN))
        if templates:
            compiled[label] = {"type": sign_type, "templates": templates}
    return compiled


def make_hands():
    # Always the fast/lite model now (complexity 0) - used for BOTH teaching
    # and recognition, so they stay consistent with each other. Combined
    # with downscaled-frame inference (to_inference_frame) and the
    # static/motion-aware matching below, this is most of where the speed
    # difference comes from; the full-accuracy model (complexity 1) is
    # noticeably heavier per frame and wasn't buying much accuracy here.
    return mp_hands.Hands(
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
        model_complexity=0,
        max_num_hands=2,
    )


def add_sign(cap, hands, dataset):
    label = input("\nType the word/phrase this sign means: ").strip()
    if not label:
        print("No label entered, cancelling.")
        return

    print(f"You'll perform '{label}' {REPEATS_PER_SIGN} times.")
    print("You can use one hand or two, whatever the sign needs - both are")
    print("tracked automatically. Vary the angle/speed slightly each repeat -")
    print("this is what makes recognition robust later. Press 'q' any time to cancel.\n")

    new_sequences = []

    for rep in range(1, REPEATS_PER_SIGN + 1):
        print(f"Repeat {rep}/{REPEATS_PER_SIGN}: get ready...")
        time.sleep(1.5)
        print("GO - perform the sign now!")

        sequence = []
        start = time.time()
        cancelled = False

        while time.time() - start < RECORD_SECONDS:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(to_inference_frame(frame), cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                for hl in result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)
                sequence.append(combined_frame_vector(result).tolist())

            remaining = RECORD_SECONDS - (time.time() - start)
            cv2.putText(frame, f"Recording '{label}' rep {rep}: {remaining:.1f}s",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Add Sign", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                cancelled = True
                break

        if cancelled:
            print("Cancelled.")
            cv2.destroyWindow("Add Sign")
            return

        if len(sequence) < 3:
            print("Barely any hand detected that rep - redoing it.")
            continue

        new_sequences.append(sequence)
        print(f"Captured {len(sequence)} frames for repeat {rep}.")

    cv2.destroyWindow("Add Sign")

    if not new_sequences:
        print("No usable data captured, nothing saved.")
        return

    entry = _migrate_entry(dataset.get(label, {"type": "motion", "sequences": []}))
    entry["sequences"].extend(new_sequences)
    entry["type"] = classify_sign_type(entry["sequences"])  # re-evaluate over ALL reps for this label
    dataset[label] = entry
    save_dataset(dataset)
    print(f"Saved. '{label}' now has {len(entry['sequences'])} example sequence(s), "
          f"classified as {entry['type'].upper()} "
          f"({'fast single-frame match' if entry['type'] == 'static' else 'movement tracked via DTW'}).")
    print("(If that classification looks wrong, just re-teach it - hold still for a static pose, "
          "or move more clearly for a motion gesture.)")


def manage_signs(dataset):
    """Menu for editing the taught vocabulary itself - renaming a sign's
    word (e.g. you taught 'hi' but want it to read as 'hello'), deleting a
    sign entirely, or dropping just one bad repetition without losing the
    rest of that sign's training data. This edits signs_dataset.json
    directly (via save_dataset), separate from editing a live sentence
    buffer during recognition (see the 'z'/'r' hotkeys in run_recognition)."""
    if not dataset:
        print("No signs taught yet - nothing to edit.")
        return

    while True:
        labels = sorted(dataset.keys())
        print("\n--- Edit Taught Signs ---")
        for i, label in enumerate(labels, 1):
            entry = _migrate_entry(dataset[label])
            reps = len(entry.get("sequences", []))
            print(f"{i}) {label}  [{entry.get('type', 'motion').upper()}, {reps} rep(s)]")
        print("r) Rename a sign")
        print("d) Delete a sign entirely")
        print("t) Delete just one repetition from a sign")
        print("b) Back to main menu")
        choice = input("Choose an option: ").strip().lower()

        if choice == "b":
            break

        elif choice == "r":
            old = input("Word to rename (exact, as shown above): ").strip()
            if old not in dataset:
                print(f"'{old}' isn't a taught sign.")
                continue
            new = input(f"New word for '{old}': ").strip()
            if not new:
                print("No new word entered, cancelled.")
                continue
            if new in dataset:
                print(f"'{new}' already exists as a separate sign - delete or merge it first.")
                continue
            dataset[new] = dataset.pop(old)
            save_dataset(dataset)
            print(f"Renamed '{old}' -> '{new}'.")

        elif choice == "d":
            target = input("Word to delete entirely (exact, as shown above): ").strip()
            if target not in dataset:
                print(f"'{target}' isn't a taught sign.")
                continue
            confirm = input(f"Type the word again to confirm deleting ALL of '{target}': ").strip()
            if confirm == target:
                del dataset[target]
                save_dataset(dataset)
                print(f"Deleted '{target}'.")
            else:
                print("Confirmation didn't match - nothing deleted.")

        elif choice == "t":
            target = input("Word to trim a repetition from: ").strip()
            if target not in dataset:
                print(f"'{target}' isn't a taught sign.")
                continue
            entry = _migrate_entry(dataset[target])
            reps = entry.get("sequences", [])
            if not reps:
                print(f"'{target}' has no repetitions to remove.")
                continue
            for i, seq in enumerate(reps, 1):
                print(f"  rep {i}: {len(seq)} frames")
            try:
                idx = int(input(f"Which repetition to delete (1-{len(reps)})? ").strip())
            except ValueError:
                print("Invalid input, cancelled.")
                continue
            if not (1 <= idx <= len(reps)):
                print("Out of range, cancelled.")
                continue
            reps.pop(idx - 1)
            if not reps:
                del dataset[target]
                print(f"That was the last repetition - '{target}' has been removed entirely.")
            else:
                entry["sequences"] = reps
                entry["type"] = classify_sign_type(reps)  # re-evaluate now that a rep is gone
                dataset[target] = entry
                print(f"Removed repetition {idx} from '{target}' ({len(reps)} rep(s) left, "
                      f"now classified as {entry['type'].upper()}).")
            save_dataset(dataset)

        else:
            print("Invalid choice, try again.")


def classify(buffer, compiled):
    """Compare the live rolling buffer against all pre-compiled sign
    templates. Static signs are checked with one cheap distance calc
    against the current (smoothed) hand pose - no DTW. Motion signs are
    resampled to a fixed length and compared with banded DTW.

    Distances are normalized to (distance / that type's threshold) so
    static and motion results - which live on different scales - can be
    ranked against each other on equal footing.

    Two accuracy safeguards on top of the raw distance:
      - CONFUSION_MARGIN_RATIO: the winning label's normalized distance
        must beat the next-best *distinct* label's by a clear margin, or
        we report no match at all rather than guessing between two
        similar-looking signs.
      - Callers additionally require a few consecutive agreeing results
        before treating a detection as real (filters one-off noise/jitter
        frames) - see CONFIRM_HITS_STATIC / CONFIRM_HITS_MOTION. The
        winning label's type is returned so the caller knows which count
        to apply.
    """
    if not buffer:
        return None, None, None

    buf_arr = np.asarray(buffer, dtype=np.float32)
    current_pose = buf_arr[-min(3, len(buf_arr)):].mean(axis=0)  # smoothed current hand shape, for static matches
    resampled_motion = None  # computed lazily, only if a motion sign is actually in the dataset

    label_ratio = {}
    for label, entry in compiled.items():
        sign_type = entry["type"]
        templates = entry["templates"]

        if sign_type == "static":
            best = min(float(np.linalg.norm(current_pose - t)) for t in templates)
            label_ratio[label] = best / STATIC_DIST_THRESHOLD
        else:
            if len(buf_arr) < 3:
                continue
            if resampled_motion is None:
                resampled_motion = resample_sequence(buf_arr, RESAMPLE_LEN)
            best = min(dtw_distance_banded(resampled_motion, t) for t in templates)
            label_ratio[label] = best / DTW_THRESHOLD

    if not label_ratio:
        return None, None, None

    ranked = sorted(label_ratio.items(), key=lambda kv: kv[1])
    best_label, best_ratio = ranked[0]
    second_ratio = ranked[1][1] if len(ranked) > 1 else float("inf")

    if best_ratio <= 1.0 and (second_ratio - best_ratio) >= CONFUSION_MARGIN_RATIO:
        return best_label, best_ratio, compiled[best_label]["type"]
    return None, best_ratio, None


def draw_caption(frame, text, settings):
    font = FONT_OPTIONS[settings["font_idx"] % len(FONT_OPTIONS)]
    color = COLOR_OPTIONS[settings["color_idx"] % len(COLOR_OPTIONS)]
    pos = tuple(settings["pos"])
    scale = settings["font_scale"]
    # A thin black outline behind the text keeps it readable over any background.
    cv2.putText(frame, text, pos, font, scale, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(frame, text, pos, font, scale, color, 2, cv2.LINE_AA)


def get_screen_size():
    """Cross-platform screen resolution lookup (via tkinter, which ships
    with Python) so fullscreen mode can size itself correctly."""
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        w, h = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
        return w, h
    except Exception:
        return 1280, 720


def fit_to_screen(frame, screen_w, screen_h):
    """Scales the frame to fit the screen while preserving its aspect ratio
    (letterboxed with black bars) so fullscreen mode doesn't stretch/distort
    the picture - unlike naively resizing straight to the screen size."""
    h, w = frame.shape[:2]
    scale = min(screen_w / w, screen_h / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
    x_off, y_off = (screen_w - new_w) // 2, (screen_h - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def draw_hud(frame, settings):
    voice_state = "ON" if settings["voice_enabled"] else "OFF"
    hud = (f"[v] voice:{voice_state}  [f] font:{FONT_NAMES[settings['font_idx'] % len(FONT_NAMES)]}  "
           f"[c] color  [+/-] size  [i/k/j/l] move  [m] fullscreen  [e] end sentence  "
           f"[z] undo word  [r] edit words  [x] clear  [q] quit")
    h = frame.shape[0]
    cv2.putText(frame, hud, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)


def handle_hotkey(key, settings, frame_shape):
    """Returns True if 'q' was pressed (quit)."""
    if key == ord('q'):
        return True
    elif key == ord('v'):
        settings["voice_enabled"] = not settings["voice_enabled"]
    elif key == ord('f'):
        settings["font_idx"] = (settings["font_idx"] + 1) % len(FONT_OPTIONS)
    elif key == ord('c'):
        settings["color_idx"] = (settings["color_idx"] + 1) % len(COLOR_OPTIONS)
    elif key in (ord('+'), ord('=')):
        settings["font_scale"] = round(min(3.0, settings["font_scale"] + 0.1), 2)
    elif key == ord('-'):
        settings["font_scale"] = round(max(0.3, settings["font_scale"] - 0.1), 2)
    elif key == ord('i'):
        settings["pos"][1] = max(20, settings["pos"][1] - 10)
    elif key == ord('k'):
        settings["pos"][1] = min(frame_shape[0] - 10, settings["pos"][1] + 10)
    elif key == ord('j'):
        settings["pos"][0] = max(0, settings["pos"][0] - 10)
    elif key == ord('l'):
        settings["pos"][0] = min(frame_shape[1] - 10, settings["pos"][0] + 10)
    elif key == ord('s'):
        save_settings(settings)
        print("Display/voice settings saved.")
    return False


def run_recognition(cap, hands, dataset, settings):
    if not dataset:
        print("No signs taught yet! Use 'Add a new sign' first.")
        return

    print("\nRunning live recognition. Sign naturally - individual signs are")
    print("buffered and combined into one sentence once you pause "
          f"({SENTENCE_PAUSE_AFTER_FIRST_WORD:.1f}s after a single word, "
          f"{SENTENCE_PAUSE_WHILE_BUILDING:.1f}s once you're mid-sentence - "
          "or press 'e' to end it early).")
    print("Only the finished sentence is spoken/shown as a message - not")
    print("each raw sign. Press 'm' to toggle real fullscreen, 'q' to stop.")
    print("Misdetected a word? Press 'z' to undo just the last one, or 'r' to")
    print("retype the whole buffered sentence before it's spoken.\n")

    # Pre-compile the whole dataset ONCE before the loop starts, instead of
    # re-parsing JSON/rebuilding numpy arrays on every match check like
    # before - see compile_dataset()'s docstring.
    compiled = compile_dataset(dataset)
    if not compiled:
        print("No usable signs after compiling (all recordings may be outdated - see the startup warning).")
        return

    window_name = "Sign Recognition"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    screen_w, screen_h = get_screen_size()
    fullscreen = True
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    fps_estimate = 20  # will self-correct below
    buffer_maxlen = int(BUFFER_SECONDS * fps_estimate)
    buffer = deque(maxlen=buffer_maxlen)

    # committed_label is the sign currently being "held". A word is added
    # to the sentence the moment a sign is first confirmed, and NOT again
    # while the same sign keeps being held (however long that is) - only
    # once it changes to a different sign, or the hand leaves frame and
    # comes back, does the same word become eligible again.
    committed_label = None
    confirm_label, confirm_count = None, 0
    frame_count = 0
    prev_t = time.time()
    live_indicator = "watching..."

    pending_words = []
    last_word_time = 0.0
    transcript = deque(maxlen=TRANSCRIPT_MAXLEN)

    def finalize_sentence():
        nonlocal pending_words
        if not pending_words:
            return
        sentence = words_to_sentence(pending_words)
        sentence = add_emojis(sentence, pending_words)
        transcript.append(sentence)
        print(f"You: {sentence}")
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(sentence + "\n")
        except Exception as e:
            print(f"(couldn't write to {LOG_FILE}: {e})")
        speak(strip_emoji(sentence), settings)
        pending_words = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(to_inference_frame(frame), cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hl in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)
            buffer.append(combined_frame_vector(result).tolist())

            frame_count += 1
            if frame_count % CHECK_EVERY_N_FRAMES == 0:
                label, ratio, sign_type = classify(buffer, compiled)
                now = time.time()
                needed_hits = CONFIRM_HITS_STATIC if sign_type == "static" else CONFIRM_HITS_MOTION
                if ratio is not None and ratio <= CONFIDENT_MATCH_RATIO:
                    needed_hits = 1  # fast path - see CONFIDENT_MATCH_RATIO

                # Require the same label back-to-back before trusting it -
                # filters one-off noisy/ambiguous frames. Static poses (and
                # any comfortably confident match) only need 1 hit so a
                # sign registers on essentially the same frame a real
                # signer would expect it to, even mid-gesture.
                if label:
                    if label == confirm_label:
                        confirm_count += 1
                    else:
                        confirm_label, confirm_count = label, 1
                else:
                    confirm_label, confirm_count = None, 0

                if label and confirm_count >= needed_hits:
                    if label != committed_label:
                        # A genuinely new/different sign (not just the same
                        # one still being held) - add it exactly once.
                        committed_label = label
                        live_indicator = f"detecting: {label}"
                        print(f"  (sign detected: {label}, match={ratio:.2f})")
                        if label.lower() in END_SIGN_WORDS:
                            finalize_sentence()
                        else:
                            pending_words.append(label.lower())
                            last_word_time = now
                    else:
                        # Same sign still held - don't add it again, no
                        # matter how long it's been held for.
                        live_indicator = f"holding: {label}"
                elif label:
                    live_indicator = f"confirming: {label}..."
        else:
            buffer.clear()
            committed_label = None  # hand left frame - the same sign is eligible again next time it's shown
            confirm_label, confirm_count = None, 0
            live_indicator = "no hand detected"

        # Pause-based sentence finalization: combine everything buffered
        # since the last pause into one sentence, rather than announcing
        # each sign as it comes in. The allowed pause is SHORTER while
        # only one word is pending (so a genuine one-word sentence still
        # comes back quickly) and LONGER once 2+ words are chained (so an
        # in-progress sentence gets enough thinking time between signs and
        # doesn't get fragmented into several separate one-word outputs).
        now = time.time()
        current_pause_limit = (
            SENTENCE_PAUSE_AFTER_FIRST_WORD if len(pending_words) <= 1
            else SENTENCE_PAUSE_WHILE_BUILDING
        )
        if pending_words and (now - last_word_time > current_pause_limit):
            finalize_sentence()
        elif len(pending_words) >= MAX_SENTENCE_WORDS:
            finalize_sentence()

        # keep a rough live fps estimate to size the buffer sensibly
        now_t = time.time()
        dt = now_t - prev_t
        prev_t = now_t
        if dt > 0:
            live_fps = 1.0 / dt
            target_len = max(5, int(BUFFER_SECONDS * live_fps))
            if abs(target_len - buffer.maxlen) > 5:
                buffer = deque(buffer, maxlen=target_len)

        # Small, muted per-sign debug line - deliberately NOT the main
        # output, so this doesn't read as a word-for-word gloss dump.
        cv2.putText(frame, live_indicator, (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (140, 140, 140), 1, cv2.LINE_AA)
        if pending_words:
            building = "building: " + " ".join(pending_words)
            cv2.putText(frame, building, (10, 44), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (140, 140, 140), 1, cv2.LINE_AA)

        frame = render_transcript_panel(frame, transcript, settings)
        draw_hud(frame, settings)

        display_frame = fit_to_screen(frame, screen_w, screen_h) if fullscreen else frame
        cv2.imshow(window_name, display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key != 255:
            if key == ord('e'):
                finalize_sentence()
            elif key == ord('z'):
                if pending_words:
                    removed = pending_words.pop()
                    print(f"(undo: removed last word '{removed}' - buffer now: "
                          f"{' '.join(pending_words) if pending_words else '(empty)'})")
                    # Let the same sign be picked up again right away if they
                    # redo it, and don't let the pause timer expire mid-fix.
                    committed_label = None
                    confirm_label, confirm_count = None, 0
                    last_word_time = time.time()
            elif key == ord('r'):
                # Free-text edit of the whole buffered sentence before it's
                # spoken - this pauses the camera loop on a console prompt,
                # same as the teaching/settings menus do elsewhere.
                current = " ".join(pending_words) if pending_words else "(none)"
                print(f"\nCurrent buffered words: {current}")
                edited = input("Edit words (space-separated), Enter to keep as-is: ").strip()
                if edited:
                    pending_words = edited.lower().split()
                    print(f"(buffer updated to: {' '.join(pending_words)})")
                last_word_time = time.time()
            elif key == ord('x'):
                transcript.clear()
            elif key == ord('m'):
                fullscreen = not fullscreen
                cv2.setWindowProperty(
                    window_name, cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL,
                )
            elif handle_hotkey(key, settings, frame.shape):
                break

    if pending_words:
        finalize_sentence()

    cv2.destroyWindow(window_name)


def settings_menu(settings):
    while True:
        print("\n--- Display & Voice Settings ---")
        print(f"1) Voice output : {'ON' if settings['voice_enabled'] else 'OFF'}  (toggle)")
        print(f"2) Font         : {FONT_NAMES[settings['font_idx'] % len(FONT_NAMES)]}  (cycle)")
        print(f"3) Color        : {COLOR_NAMES[settings['color_idx'] % len(COLOR_NAMES)]}  (cycle)")
        print(f"4) Position     : {tuple(settings['pos'])}  (set x,y)")
        print(f"5) Font scale   : {settings['font_scale']}  (set)")
        print("6) Save and go back")
        choice = input("Choose an option: ").strip()

        if choice == "1":
            settings["voice_enabled"] = not settings["voice_enabled"]
        elif choice == "2":
            settings["font_idx"] = (settings["font_idx"] + 1) % len(FONT_OPTIONS)
        elif choice == "3":
            settings["color_idx"] = (settings["color_idx"] + 1) % len(COLOR_OPTIONS)
        elif choice == "4":
            try:
                x = int(input("New X position: ").strip())
                y = int(input("New Y position: ").strip())
                settings["pos"] = [x, y]
            except ValueError:
                print("Invalid input, position unchanged.")
        elif choice == "5":
            try:
                settings["font_scale"] = float(input("New font scale (e.g. 0.9): ").strip())
            except ValueError:
                print("Invalid input, scale unchanged.")
        elif choice == "6":
            save_settings(settings)
            print("Saved.")
            break
        else:
            print("Invalid choice, try again.")


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Error: could not open camera.")
        return

    # Ask for higher FPS/resolution from the camera if it supports them -
    # higher resolution also looks much less blurry once scaled up to fill
    # the screen in fullscreen mode. OpenCV/the camera driver will fall back
    # to the nearest supported resolution if 720p isn't available.
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    dataset = load_dataset()  # also auto-migrates/classifies any old-format signs, see load_dataset()
    settings = load_settings()

    outdated = [
        label for label, entry in dataset.items()
        if entry.get("sequences") and entry["sequences"][0]
        and len(entry["sequences"][0][0]) != FEATURE_DIM
    ]
    if outdated:
        print("NOTE: these signs were taught before two-hand tracking was added, so "
              f"they're being ignored until you re-teach them: {', '.join(outdated)}")

    # Quick sanity check: fail fast (with a clear message) if this machine
    # has no usable TTS voice installed, instead of silently never speaking.
    try:
        _probe = pyttsx3.init()
        voices = _probe.getProperty("voices")
        _probe.stop()
        if not voices:
            print("WARNING: no text-to-speech voices were found on this system - "
                  "voice output will not work until one is installed (Windows: "
                  "Settings > Time & Language > Speech > Manage voices).")
    except Exception as e:
        print(f"WARNING: couldn't initialize a text-to-speech engine ({e}). "
              "Voice output will not work until this is resolved.")

    speaker_thread = threading.Thread(target=speech_worker, daemon=True)
    speaker_thread.start()

    # Same fast hand model for BOTH teaching and recognition - consistency
    # between the two matters (see make_hands()).
    with make_hands() as hands:
        while True:
            print("\n--- Sign Language Translator ---")
            print(f"Signs known: {len(dataset)} ({', '.join(dataset.keys()) if dataset else 'none yet'})")
            print(f"Voice: {'ON' if settings['voice_enabled'] else 'OFF'}")
            print("1) Add a new sign")
            print("2) Run live recognition")
            print("3) Edit taught signs (rename/delete)")
            print("4) Display & voice settings")
            print("5) Quit")
            choice = input("Choose an option: ").strip()

            if choice == "1":
                add_sign(cap, hands, dataset)
            elif choice == "2":
                run_recognition(cap, hands, dataset, settings)
            elif choice == "3":
                manage_signs(dataset)
            elif choice == "4":
                settings_menu(settings)
            elif choice == "5":
                break
            else:
                print("Invalid choice, try again.")

    save_settings(settings)
    speech_queue.put(None)  # stop the speech worker
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

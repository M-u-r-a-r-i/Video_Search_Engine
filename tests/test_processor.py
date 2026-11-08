from backend.processor import words_to_sentences


def _words(spec):
    """spec: list of (word, start) with 0.4 s per word."""
    return [{"text": w, "start": t, "end": t + 0.4} for w, t in spec]


def test_splits_on_sentence_punctuation():
    words = _words([("Hello", 0.0), ("world.", 0.5), ("How", 2.0), ("are", 2.5), ("you?", 3.0)])
    out = words_to_sentences(words)
    assert [s["text"] for s in out] == ["Hello world.", "How are you?"]
    assert out[1]["start"] == 2.0          # start = first word's timestamp
    assert out[1]["end"] == 3.4


def test_trailing_words_without_punctuation_are_kept():
    out = words_to_sentences(_words([("so", 0.0), ("yeah", 0.5)]))
    assert [s["text"] for s in out] == ["so yeah"]


def test_unpunctuated_speech_is_force_split_by_time():
    words = _words([(f"w{i}", i * 1.0) for i in range(40)])  # 40 s, no punctuation
    out = words_to_sentences(words, max_seconds=15, max_words=1000)
    assert len(out) >= 3
    assert all(s["end"] - s["start"] <= 15.5 for s in out)


def test_force_split_by_word_count():
    words = _words([(f"w{i}", i * 0.1) for i in range(25)])
    out = words_to_sentences(words, max_seconds=999, max_words=10)
    assert [len(s["text"].split()) for s in out] == [10, 10, 5]


def test_empty():
    assert words_to_sentences([]) == []

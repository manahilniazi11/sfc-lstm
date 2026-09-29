from sonic.dataset.freesound import _is_excluded, load_queries

EXCLUDE = {"melodica", "alarm", "imitation", "music"}


def test_excluded_by_name_word_or_tag():
    assert _is_excluded({"name": "Melodica Car Horn Imitation", "tags": []}, EXCLUDE)
    assert _is_excluded({"name": "Car Alarm Horn, Walking Past.wav", "tags": ["car"]}, EXCLUDE)
    assert _is_excluded({"name": "honk.wav", "tags": ["Music"]}, EXCLUDE)


def test_real_horns_are_kept():
    assert not _is_excluded({"name": "car horn three honks.wav", "tags": ["car", "horn"]}, EXCLUDE)
    assert not _is_excluded({"name": "Horn_Doppler_01.wav", "tags": ["doppler"]}, EXCLUDE)


def test_query_config_uses_known_classes():
    assert "Vehicle Horn" in load_queries()

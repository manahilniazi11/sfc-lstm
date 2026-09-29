"""Tests for the dataset builder, using small synthetic WAV files.

Each fake dataset mimics the real layout just enough for its adapter.
"""

from __future__ import annotations

import csv
import random
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from sonic.dataset.builder import BuildOptions, build
from sonic.dataset.config import ConfigError, load_classes, load_label_map
from sonic.dataset.splitting import assign_splits

SR = 16000


def write_tone(path: Path, seconds: float = 1.0, amplitude: float = 0.5, freq: float = 440.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(SR * seconds)) / SR
    sf.write(path, amplitude * np.sin(2 * np.pi * freq * t), SR, subtype="PCM_16")
    return path


def make_esc50(raw: Path, rows: list[tuple[str, str, str]]) -> None:
    """rows: (filename, category, freesound id)."""
    root = raw / "esc50" / "ESC-50-master"
    (root / "meta").mkdir(parents=True)
    with open(root / "meta" / "esc50.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "fold", "target", "category", "esc10", "src_file", "take"])
        for i, (name, category, fs_id) in enumerate(rows):
            w.writerow([name, 1, 0, category, False, fs_id, "A"])
            write_tone(root / "audio" / name, freq=200 + 10 * i)


def make_urbansound(raw: Path, rows: list[tuple[str, str, str]]) -> None:
    """rows: (slice filename, class, fsID)."""
    root = raw / "urbansound8k" / "UrbanSound8K"
    (root / "metadata").mkdir(parents=True)
    with open(root / "metadata" / "UrbanSound8K.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["slice_file_name", "fsID", "start", "end", "salience", "fold", "classID", "class"])
        for i, (name, cls, fs_id) in enumerate(rows):
            w.writerow([name, fs_id, 0, 1, 1, 1, 0, cls])
            write_tone(root / "audio" / "fold1" / name, freq=900 + 10 * i)


@pytest.fixture
def raw(tmp_path: Path) -> Path:
    return tmp_path / "raw"


def run(tmp_path: Path, raw: Path, **kwargs) -> tuple[list[dict], list[dict], dict]:
    opts = BuildOptions(
        raw_dir=raw, out_dir=tmp_path / "out", metadata_dir=tmp_path / "meta", **kwargs
    )
    result = build(opts)
    with open(opts.metadata_dir / "dataset_metadata.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    with open(opts.metadata_dir / "rejected_files.csv", newline="") as f:
        rejected = list(csv.DictReader(f))
    return rows, rejected, result.stats


# --- splitting -----------------------------------------------------------------

def test_split_ratios_with_equal_groups():
    groups = {f"g{i}": 1 for i in range(100)}
    assignment = assign_splits(groups, random.Random(0))
    assert Counter(assignment.values()) == {"train": 70, "validation": 15, "test": 15}


def test_split_keeps_groups_whole_and_is_deterministic():
    groups = {f"g{i}": (i % 5) + 1 for i in range(40)}
    first = assign_splits(groups, random.Random(7))
    second = assign_splits(groups, random.Random(7))
    assert first == second
    assert set(first) == set(groups)  # every group assigned exactly once


def test_split_rejects_bad_ratios():
    with pytest.raises(ValueError):
        assign_splits({"g": 1}, random.Random(0), (0.5, 0.5, 0.5))


# --- config --------------------------------------------------------------------

def test_default_config_is_consistent():
    classes = load_classes()
    assert len([c for c in classes if c.mandatory]) == 10
    load_label_map(classes)  # raises if any mapped class name is misspelt


def test_label_map_with_unknown_class_fails(tmp_path: Path):
    bad = tmp_path / "map.json"
    bad.write_text('{"esc50": {"dog": "Animal Sounds"}}')
    with pytest.raises(ConfigError, match="Animal Sounds"):
        load_label_map(load_classes(), bad)


# --- end-to-end build ------------------------------------------------------------

def test_build_rejects_silent_short_and_duplicate_files(tmp_path: Path, raw: Path):
    help_dir = raw / "custom" / "help_request"
    for i in range(6):
        write_tone(help_dir / f"take{i}.wav", freq=300 + 50 * i)
    write_tone(help_dir / "silent.wav", amplitude=0.0)
    write_tone(help_dir / "short.wav", seconds=0.1)
    shutil.copy(help_dir / "take0.wav", help_dir / "take0_copy.wav")

    rows, rejected, _ = run(tmp_path, raw)

    assert len(rows) == 6
    reasons = {Path(r["source_path"]).name: r["reason"] for r in rejected}
    assert reasons["silent.wav"].startswith("silent")
    assert reasons["short.wav"].startswith("too short")
    duplicate = next(v for k, v in reasons.items() if k in {"take0.wav", "take0_copy.wav"})
    assert duplicate.startswith("exact duplicate")


def test_build_copies_files_and_assigns_class_coded_ids(tmp_path: Path, raw: Path):
    make_esc50(raw, [(f"1-{i}-A-0.wav", "dog", str(1000 + i)) for i in range(10)])
    rows, _, stats = run(tmp_path, raw)

    assert [r["audio_id"] for r in rows] == [f"ANM-{n:05d}" for n in range(1, 11)]
    for r in rows:
        assert (tmp_path / "out" / r["filename"]).is_file()
        assert r["filename"].startswith(f"{r['split']}/animal_sound/")
        assert r["class_name"] == "Animal Sound"
        assert r["license"].startswith("CC BY-NC")
    assert stats["classes"]["Animal Sound"]["total"] == 10
    assert all(r["quality_grade"] in {"Good", "Acceptable", "Poor"} for r in rows)


def test_slices_of_one_recording_never_cross_splits(tmp_path: Path, raw: Path):
    # 20 Freesound recordings, each cut into 4 slices.
    make_urbansound(raw, [(f"{g}-0-0-{s}.wav", "gun_shot", str(g)) for g in range(20) for s in range(4)])
    rows, _, _ = run(tmp_path, raw, per_class=0)

    splits_per_group: dict[str, set[str]] = {}
    for r in rows:
        splits_per_group.setdefault(r["group_id"], set()).add(r["split"])
    assert all(len(s) == 1 for s in splits_per_group.values())
    assert {r["split"] for r in rows} == {"train", "validation", "test"}


def test_per_class_cap_prefers_distinct_recordings(tmp_path: Path, raw: Path):
    make_urbansound(raw, [(f"{g}-0-0-{s}.wav", "gun_shot", str(g)) for g in range(10) for s in range(5)])
    rows, _, _ = run(tmp_path, raw, per_class=10)
    assert len(rows) == 10
    assert len({r["group_id"] for r in rows}) == 10  # one slice from each recording


def test_same_recording_labelled_differently_is_dropped(tmp_path: Path, raw: Path):
    make_esc50(raw, [("1-1-A-0.wav", "siren", "555"), ("1-2-A-0.wav", "dog", "556")])
    make_urbansound(raw, [("555-0-0-0.wav", "car_horn", "555")])
    rows, rejected, _ = run(tmp_path, raw)

    assert {r["freesound_id"] for r in rows} == {"556"}
    assert sum("label conflict" in r["reason"] for r in rejected) == 2


def test_custom_recordings_metadata_and_groups(tmp_path: Path, raw: Path):
    custom = raw / "custom"
    for i in range(3):
        write_tone(custom / "help_request" / f"ali_{i}.wav", freq=500 + i * 20)
    (custom / "recordings.csv").write_text(
        "path,device,environment,distance_m,speaker,group\n"
        "help_request/ali_0.wav,iPhone 13,indoor room,2,ali,session1\n"
        "help_request/ali_1.wav,iPhone 13,indoor room,2,ali,session1\n"
    )
    rows, _, _ = run(tmp_path, raw)

    by_name = {r["original_filename"]: r for r in rows}
    assert by_name["ali_0.wav"]["device"] == "iPhone 13"
    assert by_name["ali_0.wav"]["distance_m"] == "2"
    assert by_name["ali_0.wav"]["group_id"] == by_name["ali_1.wav"]["group_id"] == "custom:session1"
    assert by_name["ali_0.wav"]["split"] == by_name["ali_1.wav"]["split"]
    assert by_name["ali_2.wav"]["group_id"] == "custom:help_request/ali_2.wav"


def test_fsd50k_skips_excluded_and_ambiguous_clips(tmp_path: Path, raw: Path):
    root = raw / "fsd50k"
    (root / "FSD50K.ground_truth").mkdir(parents=True)
    clips = {
        "1": "Alarm",                                   # kept -> Alarm or Siren
        "2": "Alarm,Telephone",                         # excluded: a phone ring
        "3": "Screaming,Shout,Human_voice",             # ambiguous: two classes
        "4": "Gunshot_and_gunfire,Explosion",           # kept -> Gunshot
    }
    with open(root / "FSD50K.ground_truth" / "eval.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fname", "labels", "mids"])
        for i, (fname, labels) in enumerate(clips.items()):
            w.writerow([fname, labels, ""])
            write_tone(root / "FSD50K.eval_audio" / f"{fname}.wav", freq=300 + 40 * i)

    rows, _, _ = run(tmp_path, raw)
    assert {(r["freesound_id"], r["class_name"]) for r in rows} == {
        ("1", "Alarm or Siren"), ("4", "Gunshot"),
    }


def test_actor_groups_count_each_utterance_as_a_unique_recording(tmp_path: Path, raw: Path):
    # 2 actors x 3 strongly angry utterances: 2 split groups, 6 original recordings.
    for actor in ("01", "02"):
        for statement in ("01", "02", "03"):
            write_tone(raw / "ravdess" / f"Actor_{actor}" / f"03-01-05-02-{statement}-01-{actor}.wav",
                       freq=300 + 20 * int(statement) + int(actor))
    rows, _, stats = run(tmp_path, raw)
    assert len({r["group_id"] for r in rows}) == 2
    assert stats["classes"]["Aggression"]["unique_recordings"] == 6
    for actor in ("01", "02"):
        assert len({r["split"] for r in rows if r["speaker"] == f"ravdess:{actor}"}) == 1


def test_nonspeech7k_and_freesound_sources(tmp_path: Path, raw: Path):
    ns = raw / "nonspeech7k"
    ns.mkdir(parents=True)
    (ns / "metadata of test set.csv").write_text(
        "Filename,File ID,Duration in ms,Class ID,Classname,Augment id,Augmentation type,source\n"
        "111_4_0.wav,111,1000,4,screaming,0,Original,https://freesound.org/\n"
        "a-1_4_0.wav,1,1000,4,screaming,0,Original,https://aigei.com/\n"
        "c_0_0.wav,5,1000,0,breath,0,Original,https://freesound.org/\n"
    )
    for i, name in enumerate(("111_4_0.wav", "a-1_4_0.wav", "c_0_0.wav")):
        write_tone(ns / "test" / name, freq=400 + 50 * i)

    fs = raw / "freesound" / "vehicle_horn"
    write_tone(fs / "999.wav", freq=700)
    (fs / "sounds.json").write_text(
        '{"999": {"file": "999.wav", "class_name": "Vehicle Horn", "query": "car horn",'
        ' "name": "honk", "username": "someone", "license": "Attribution"}}'
    )

    rows, _, _ = run(tmp_path, raw)
    by_source = {(r["source_dataset"], r["class_name"], r["group_id"], r["license"]) for r in rows}
    assert by_source == {
        ("nonspeech7k", "Panic Scream", "freesound:111", "CC BY-NC-SA 4.0 (Nonspeech7k)"),
        ("nonspeech7k", "Panic Scream", "nonspeech7k:aigei.com:screaming:1", "CC BY-NC-SA 4.0 (Nonspeech7k)"),
        ("freesound", "Vehicle Horn", "freesound:999", "CC BY"),
    }


def test_rebuild_requires_force(tmp_path: Path, raw: Path):
    make_esc50(raw, [("1-1-A-0.wav", "dog", "1")])
    run(tmp_path, raw)
    with pytest.raises(FileExistsError):
        run(tmp_path, raw)
    rows, _, _ = run(tmp_path, raw, force=True)
    assert len(rows) == 1


def test_missing_mandatory_classes_are_reported(tmp_path: Path, raw: Path):
    make_esc50(raw, [("1-1-A-0.wav", "dog", "1")])
    _, _, stats = run(tmp_path, raw)
    assert any("Gunshot: no clips at all" in w for w in stats["warnings"])


def test_extend_appends_without_changing_the_built_dataset(tmp_path: Path, raw: Path):
    """New Help clips (and the same voices' ordinary speech) are added; every existing row stays as it was."""
    from sonic.dataset.extend import extend

    make_esc50(raw, [(f"{i}-rain.wav", "rain", str(100 + i)) for i in range(6)])
    before, _, _ = run(tmp_path, raw)
    custom = raw / "custom"
    lines = ["path,speaker,group,license,source"]
    for voice in range(10):
        for take in range(3):
            name = f"help_request/v{voice}_{take}.wav"
            write_tone(custom / name, freq=300 + voice * 40 + take * 7)
            lines.append(f"{name},v{voice},tts-v{voice},synthetic,edge-tts")
        speech = f"background_noise/v{voice}_speech.wav"
        write_tone(custom / speech, freq=2000 + voice * 30)
        lines.append(f"{speech},v{voice},tts-v{voice},synthetic,edge-tts")
    (custom / "recordings.csv").write_text("\n".join(lines) + "\n")
    opts = BuildOptions(raw_dir=raw, out_dir=tmp_path / "out", metadata_dir=tmp_path / "meta")

    added = extend(opts, ["help_request", "background_noise"])
    with open(opts.metadata_dir / "dataset_metadata.csv", newline="") as f:
        after = list(csv.DictReader(f))
    assert after[:len(before)] == before  # nothing already built changed
    new = after[len(before):]
    assert sum(sum(c.values()) for c in added.values()) == len(new) == 40
    help_rows = [r for r in new if r["class_name"] == "Person Asking for Help"]
    assert help_rows[0]["audio_id"] == "HLP-00001" and len({r["origin_id"] for r in help_rows}) == 30
    split_of = {}
    for r in new:  # one voice, one split, across both classes
        assert split_of.setdefault(r["group_id"], r["split"]) == r["split"]
    assert set(split_of.values()) == {"train", "validation", "test"}
    bg_ids = sorted(int(r["audio_id"].split("-")[1]) for r in after if r["class_name"] == "Background Noise")
    assert bg_ids == list(range(1, len(bg_ids) + 1))  # new Background Noise IDs continue the old ones
    assert all((opts.out_dir / r["filename"]).exists() for r in new)
    assert extend(opts, ["help_request"]) == {}  # running it again adds nothing

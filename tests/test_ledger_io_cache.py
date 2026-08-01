"""Tests for read_jsonl_cached caching behavior in ledger_io."""

import json
import sys
import pathlib
from pathlib import Path

# Add hooks directory to path to import ledger_io
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "hooks"))

from ledger_io import read_jsonl, read_jsonl_cached, append_jsonl


def test_empty_nonexistent_file(tmp_path):
    """First call on an empty/nonexistent file returns []."""
    ledger_path = tmp_path / "empty.jsonl"
    assert not ledger_path.exists()

    result = read_jsonl_cached(ledger_path)
    assert result == []


def test_append_and_read_single_record(tmp_path):
    """Appending N lines then calling read_jsonl_cached returns all N records."""
    ledger_path = tmp_path / "ledger.jsonl"

    # Append single record
    record1 = {"event": "TestEvent", "id": 1}
    append_jsonl(ledger_path, record1)

    result = read_jsonl_cached(ledger_path)
    assert len(result) == 1
    assert result[0] == record1


def test_append_multiple_records(tmp_path):
    """Appending multiple lines then calling read_jsonl_cached returns all records."""
    ledger_path = tmp_path / "ledger.jsonl"

    records = [
        {"event": "Event1", "id": 1},
        {"event": "Event2", "id": 2},
        {"event": "Event3", "id": 3},
    ]

    for record in records:
        append_jsonl(ledger_path, record)

    result = read_jsonl_cached(ledger_path)
    assert len(result) == 3
    assert result == records


def test_incremental_append_with_cache(tmp_path):
    """Appending more lines after an initial read returns all records without
    needing to reparse the ones already cached.
    """
    ledger_path = tmp_path / "ledger.jsonl"

    # Initial records
    initial_records = [
        {"event": "Event1", "id": 1},
        {"event": "Event2", "id": 2},
    ]

    for record in initial_records:
        append_jsonl(ledger_path, record)

    # First read - should cache these
    result1 = read_jsonl_cached(ledger_path)
    assert result1 == initial_records

    # Append more records
    additional_records = [
        {"event": "Event3", "id": 3},
        {"event": "Event4", "id": 4},
    ]

    for record in additional_records:
        append_jsonl(ledger_path, record)

    # Second read - should return all records (cached + new tail)
    result2 = read_jsonl_cached(ledger_path)
    expected = initial_records + additional_records
    assert result2 == expected


def test_incremental_append_consistency(tmp_path):
    """Multiple incremental appends are correctly accumulated."""
    ledger_path = tmp_path / "ledger.jsonl"

    all_records = []

    for batch in range(3):
        batch_records = [
            {"event": f"Event{i}", "batch": batch, "id": i}
            for i in range(5)
        ]
        for record in batch_records:
            append_jsonl(ledger_path, record)
        all_records.extend(batch_records)

        result = read_jsonl_cached(ledger_path)
        assert result == all_records, f"Mismatch after batch {batch}"


def test_truncation_invalidates_cache(tmp_path):
    """Truncating the file to empty and appending a fresh line returns only
    the new content.
    """
    ledger_path = tmp_path / "ledger.jsonl"

    # Write initial records
    initial_records = [
        {"event": "Event1", "id": 1},
        {"event": "Event2", "id": 2},
    ]

    for record in initial_records:
        append_jsonl(ledger_path, record)

    # Read (caches it)
    result1 = read_jsonl_cached(ledger_path)
    assert result1 == initial_records

    # Truncate the file
    ledger_path.write_text("")

    # Append new record
    new_record = {"event": "Event3", "id": 3}
    append_jsonl(ledger_path, new_record)

    # Read should return only the new record (cache was invalidated by truncation)
    result2 = read_jsonl_cached(ledger_path)
    assert result2 == [new_record]


def test_mtime_change_invalidates_cache(tmp_path):
    """Cache is invalidated when file mtime changes (e.g., rotation)."""
    ledger_path = tmp_path / "ledger.jsonl"

    # Write initial records
    initial_records = [
        {"event": "Event1", "id": 1},
    ]

    for record in initial_records:
        append_jsonl(ledger_path, record)

    # Read (caches it with mtime)
    result1 = read_jsonl_cached(ledger_path)
    assert result1 == initial_records

    # Simulate rotation: write new content, creating a new inode (effectively)
    ledger_path.unlink()
    new_record = {"event": "Event2", "id": 2}
    ledger_path.write_text(json.dumps(new_record) + "\n")

    # Read should return only the new record (cache was invalidated by mtime change)
    result2 = read_jsonl_cached(ledger_path)
    assert result2 == [new_record]


def test_malformed_line_skipped_like_read_jsonl(tmp_path):
    """A malformed line mixed in with valid ones is skipped, not raised,
    matching read_jsonl's existing behavior exactly.
    """
    ledger_path = tmp_path / "ledger.jsonl"

    # Manually write records including a malformed line
    valid_record_1 = {"event": "Event1", "id": 1}
    malformed_line = "this is not valid json {"
    valid_record_2 = {"event": "Event2", "id": 2}

    ledger_path.write_text(
        json.dumps(valid_record_1) + "\n"
        + malformed_line + "\n"
        + json.dumps(valid_record_2) + "\n",
        encoding="utf-8"
    )

    # Both read_jsonl and read_jsonl_cached should skip the malformed line
    result_read_jsonl = read_jsonl(ledger_path)
    result_cached = read_jsonl_cached(ledger_path)

    # Should be identical and not include the malformed line
    assert result_read_jsonl == result_cached
    assert result_read_jsonl == [valid_record_1, valid_record_2]
    assert len(result_cached) == 2


def test_malformed_line_in_tail(tmp_path):
    """Malformed line in the tail after cache is skipped correctly."""
    ledger_path = tmp_path / "ledger.jsonl"

    # Write initial records
    record1 = {"event": "Event1", "id": 1}
    append_jsonl(ledger_path, record1)

    # Cache it
    result1 = read_jsonl_cached(ledger_path)
    assert result1 == [record1]

    # Manually append a malformed line and then a valid record
    malformed = "bad json line {"
    record2 = {"event": "Event2", "id": 2}

    with ledger_path.open("a", encoding="utf-8") as f:
        f.write(malformed + "\n")
        f.write(json.dumps(record2) + "\n")

    # Read again - should get cached record + new valid record, skip malformed
    result2 = read_jsonl_cached(ledger_path)
    assert result2 == [record1, record2]


def test_empty_lines_in_file_ignored(tmp_path):
    """Empty lines in the file are properly ignored."""
    ledger_path = tmp_path / "ledger.jsonl"

    record1 = {"event": "Event1", "id": 1}
    record2 = {"event": "Event2", "id": 2}

    # Write with empty lines
    ledger_path.write_text(
        json.dumps(record1) + "\n"
        + "\n"
        + json.dumps(record2) + "\n"
        + "\n",
        encoding="utf-8"
    )

    result = read_jsonl_cached(ledger_path)
    assert result == [record1, record2]


def test_cache_file_created_and_updated(tmp_path):
    """Cache file is created and updated as expected."""
    ledger_path = tmp_path / "ledger.jsonl"
    cache_path = tmp_path / "ledger.jsonl.cache.json"

    record1 = {"event": "Event1", "id": 1}
    append_jsonl(ledger_path, record1)

    # First read should create cache
    result1 = read_jsonl_cached(ledger_path)
    assert cache_path.exists()

    cache_data = json.loads(cache_path.read_text(encoding="utf-8"))
    assert "mtime" in cache_data
    assert "size" in cache_data
    assert "offset" in cache_data
    assert "records" in cache_data
    assert cache_data["records"] == [record1]

    # Append another record
    record2 = {"event": "Event2", "id": 2}
    append_jsonl(ledger_path, record2)

    # Second read should update cache
    result2 = read_jsonl_cached(ledger_path)
    assert result2 == [record1, record2]

    cache_data = json.loads(cache_path.read_text(encoding="utf-8"))
    assert cache_data["records"] == [record1, record2]


def test_corrupted_cache_falls_back_to_full_read(tmp_path):
    """Corrupted cache file triggers a full re-read."""
    ledger_path = tmp_path / "ledger.jsonl"
    cache_path = tmp_path / "ledger.jsonl.cache.json"

    record1 = {"event": "Event1", "id": 1}
    append_jsonl(ledger_path, record1)

    # Create a valid cache first
    read_jsonl_cached(ledger_path)
    assert cache_path.exists()

    # Corrupt the cache
    cache_path.write_text("not valid json {", encoding="utf-8")

    # Add more records
    record2 = {"event": "Event2", "id": 2}
    append_jsonl(ledger_path, record2)

    # Should fall back to full re-read and get all records
    result = read_jsonl_cached(ledger_path)
    assert result == [record1, record2]


def test_non_dict_records_skipped(tmp_path):
    """Records that aren't dicts are skipped (matching read_jsonl behavior)."""
    ledger_path = tmp_path / "ledger.jsonl"

    # Write a list (valid JSON but not a dict)
    ledger_path.write_text(
        json.dumps({"event": "Event1", "id": 1}) + "\n"
        + json.dumps([1, 2, 3]) + "\n"
        + json.dumps({"event": "Event2", "id": 2}) + "\n",
        encoding="utf-8"
    )

    result = read_jsonl_cached(ledger_path)
    # Should only have the dict records
    assert result == [{"event": "Event1", "id": 1}, {"event": "Event2", "id": 2}]


def test_read_jsonl_vs_read_jsonl_cached_equivalence(tmp_path):
    """read_jsonl and read_jsonl_cached should return the same results
    on a fresh file (no cache).
    """
    ledger_path = tmp_path / "ledger.jsonl"

    records = [
        {"event": "Event1", "id": 1},
        {"event": "Event2", "id": 2},
        {"event": "Event3", "id": 3},
    ]

    for record in records:
        append_jsonl(ledger_path, record)

    # Remove any cache to ensure fresh read
    cache_path = ledger_path.with_name(ledger_path.name + ".cache.json")
    cache_path.unlink(missing_ok=True)

    result_read_jsonl = read_jsonl(ledger_path)
    result_cached = read_jsonl_cached(ledger_path)

    assert result_read_jsonl == result_cached
    assert result_cached == records

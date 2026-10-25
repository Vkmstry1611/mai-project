from bda.live_pipeline import ensure_cluster, event_snapshot, spark_counts


def test_event_snapshot_changes_for_new_request(tmp_path):
    events = tmp_path / "events"
    events.mkdir()
    first, count, _ = event_snapshot(tmp_path)
    assert count == 0
    (events / "one.json").write_text("{}")
    second, count, _ = event_snapshot(tmp_path)
    assert count == 1 and second != first


def test_spark_counts_reads_job_result():
    log = "INFO start\n{'raw': 2, 'curated': 2, 'matched_pairs': 1}\nINFO stop"
    assert spark_counts(log)["matched_pairs"] == 1


def test_ensure_cluster_starts_missing_controllers(monkeypatch):
    commands = []

    def unavailable(address, timeout):
        raise OSError("not listening")

    monkeypatch.setattr("bda.live_pipeline.socket.create_connection", unavailable)
    monkeypatch.setattr("bda.live_pipeline.subprocess.run",
                        lambda command, **kwargs: commands.append(command))
    ensure_cluster()
    assert commands == [["start-dfs.sh"], ["start-yarn.sh"]]

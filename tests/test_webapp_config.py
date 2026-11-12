import json
from pathlib import Path

from scMM import webapp


def test_packaged_ui_config_has_defaults_and_help():
    path = Path(webapp.__file__).with_name("config") / "defaults.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    assert config["version"] == 1
    assert len(config["parameters"]) >= 100
    for entry in config["parameters"].values():
        assert "default" in entry
        assert entry.get("help")


def test_parameter_loader_reads_packaged_config():
    value, help_text = webapp._parameter("supervised.cv", 99)
    assert value == 5
    assert "交叉验证" in help_text

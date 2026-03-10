from unittest.mock import patch

from cli import commands


def test_read_interactive_input_returns_input():
    """Test that _read_interactive_input returns the user input."""
    with patch("builtins.input", return_value="hello world"):
        result = commands._read_interactive_input()

    assert result == "hello world"


def test_read_interactive_input_handles_eof():
    """Test that EOFError converts to KeyboardInterrupt."""
    with patch("builtins.input", side_effect=EOFError()):
        import pytest
        with pytest.raises(KeyboardInterrupt):
            commands._read_interactive_input()


def test_init_readline_sets_loaded_flag():
    """Test that _init_readline sets the loaded flag."""
    commands._HISTORY_LOADED = False

    with patch("nanobot.cli.commands.get_cli_history_path") as mock_path:
        mock_path.return_value.parent.mkdir = lambda **_: None
        with patch("readline.read_history_file", side_effect=FileNotFoundError):
            with patch("atexit.register"):
                commands._init_readline()

    assert commands._HISTORY_LOADED is True

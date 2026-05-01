"""
synth.py — klattsch CLI runner
Converts an ARPABET string + voice params to a WAV via `npx klattsch`,
then returns the path to the rendered WAV file.
"""
import subprocess
import tempfile
import sys
import os
import shutil


def _build_prefix(params: dict) -> str:
    """
    Build the klattsch directive prefix string from voice params dict.
    Keys: baseF0, rate, scale, vibratoDepth, vibratoRate, aspiration, tilt, effort
    Directive letters: b r s v w h t g
    """
    mapping = {
        "baseF0":       ("b", int),
        "rate":         ("r", int),
        "scale":        ("s", lambda x: f"{float(x):.2f}"),
        "vibratoDepth": ("v", lambda x: f"{float(x):.1f}"),
        "vibratoRate":  ("w", lambda x: f"{float(x):.1f}"),
        "aspiration":   ("h", lambda x: f"{float(x):.2f}"),
        "tilt":         ("t", lambda x: f"{float(x):.2f}"),
        "effort":       ("g", lambda x: f"{float(x):.2f}"),
    }
    parts = []
    for key, (letter, fmt) in mapping.items():
        val = params.get(key)
        if val is not None:
            parts.append(f"{letter}{fmt(val)}")
    return " ".join(parts)


def render_to_wav(arpabet: str, params: dict) -> str | None:
    """
    Render arpabet string to a temporary WAV file using klattsch CLI.
    Returns the path to the WAV, or None on failure.
    Caller is responsible for deleting the file.
    """
    if not arpabet.strip():
        return None

    # Check npx is available
    npx = shutil.which("npx")
    if not npx:
        raise RuntimeError("npx not found — please install Node.js (https://nodejs.org)")

    prefix = _build_prefix(params)
    full_input = f"{prefix} {arpabet}".strip() if prefix else arpabet

    # Create a temp file path for the output WAV
    fd, out_path = tempfile.mkstemp(suffix=".wav", prefix="vsynth_")
    os.close(fd)

    kwargs = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

    try:
        result = subprocess.run(
            [npx, "--yes", "klattsch", full_input, out_path],
            capture_output=True,
            text=True,
            timeout=30,
            **kwargs
        )
        if result.returncode != 0:
            err = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(f"klattsch error: {err}")
        if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
            raise RuntimeError("klattsch produced no output")
        return out_path
    except Exception:
        # Clean up on failure
        try:
            os.unlink(out_path)
        except OSError:
            pass
        raise

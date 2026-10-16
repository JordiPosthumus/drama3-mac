"""Demo: deaf vs reactive vs point-tagged dialogue on the new /v1/dialogue.

Same 10-line script, same voices, same per-line seeds — the only deltas are
(a) threaded audio history (reactive) and (b) inline point tags.
"""

import base64
import json
import urllib.request

HOST = "http://127.0.0.1:8843"


def post(path, payload, timeout=1800):
    req = urllib.request.Request(HOST + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


BASE = [
    ("A", "Welcome back to Signal and Noise, the show about the machines we build and the machines that build us. Today my guest spent thirty years keeping a lighthouse on a rock in the North Atlantic."),
    ("B", "Thirty-one, if we're counting. And thank you for having me. People always ask if it was lonely, and I never quite know how to answer."),
    ("A", "Well, was it?"),
    ("B", "Lonely implies you're missing something. Out there I had the wind, the gulls, the radio, and about four thousand books. What I didn't have was anyone to argue with, and it turns out I miss arguing."),
    ("A", "That's a very romantic way to describe deprivation."),
    ("B", "Ha! Everyone says that. But consider this: the light had to be lit at dusk and doused at dawn, every single day, no exceptions. When your whole life is a routine that matters, you stop needing variety."),
    ("A", "So what changed? Why come back to the mainland now?"),
    ("B", "The automation, of course. They put in a bulb that lasts ten years and a sensor that never sleeps. The sea didn't need me anymore. That was the day the island got smaller."),
    ("A", "Let's talk about the storms. What was the worst night out there?"),
    ("B", "January of my ninth year. Ninety-knot gusts, waves over the lantern room. I clipped myself to the rail and I sang to the light until sunrise. Badly, I might add."),
]

TAGGED = [list(l) for l in BASE]
TAGGED[1][1] = "[laughs] " + TAGGED[1][1]
TAGGED[2][1] = "[sighs] " + TAGGED[2][1]
TAGGED[7][1] = TAGGED[7][1].replace(
    "That was the day the island got smaller.",
    "<whisper>That was the day the island got smaller.</whisper>")
TAGGED[9][1] = TAGGED[9][1].replace("Badly, I might add.", "[laughs] Badly, I might add.")
TAGGED = [tuple(l) for l in TAGGED]


def lines(script):
    return [{"text": t, "voice": f"abvoice-{w.lower()}", "line_id": i, "seed": 1000 + i}
            for i, (w, t) in enumerate(script)]


def run(name, **kw):
    payload = {"lines": lines(BASE), "gap_s": 0.25, **kw}
    env = post("/v1/dialogue", payload)
    ok = env.get("ok")
    finishes = [r.get("finish_reason") for r in env.get("results", [])]
    f0s = [r.get("f0_hz_median") for r in env.get("results", [])]
    print(f"[{name}] ok={ok} finishes={finishes}")
    print(f"[{name}] f0={[(round(f) if f else None) for f in f0s]}")
    if env.get("audio_b64"):
        open(f"outputs/{name}.wav", "wb").write(base64.b64decode(env["audio_b64"]))
        print(f"[{name}] saved outputs/{name}.wav")
    return env


if __name__ == "__main__":
    run("demo-deaf", history_turns=0)
    run("demo-reactive", history_turns=1)
    env = post("/v1/dialogue", {"lines": lines(TAGGED), "gap_s": 0.25, "history_turns": 0})
    finishes = [r.get("finish_reason") for r in env.get("results", [])]
    print(f"[demo-pointtags] ok={env.get('ok')} finishes={finishes}")
    open("outputs/demo-pointtags.wav", "wb").write(base64.b64decode(env["audio_b64"]))
    env = post("/v1/dialogue", {"lines": lines(TAGGED), "gap_s": 0.25, "history_turns": 1})
    finishes = [r.get("finish_reason") for r in env.get("results", [])]
    print(f"[demo-full] ok={env.get('ok')} finishes={finishes}")
    open("outputs/demo-full.wav", "wb").write(base64.b64decode(env["audio_b64"]))
    print("done")

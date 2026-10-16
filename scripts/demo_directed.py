"""Official-style directed dialogue: speaker tags + per-turn voice-actor briefs.

Mirrors the docs.fish.audio Drama 3 multi-speaker recipe: one request, every
turn starts with <|speaker:N|> plus a bracketed direction, paired tags inside
turns for word-level delivery.
"""

import base64
import json
import urllib.request

HOST = "http://127.0.0.1:8843"


def post(path, payload, timeout=1800):
    req = urllib.request.Request(HOST + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


TURNS = [
    (0, "[warm, unhurried radio host voice] Welcome back to Signal and Noise, the show about the machines we build and the machines that build us. Today my guest spent thirty years keeping a lighthouse on a rock in the North Atlantic."),
    (1, "[quietly proud, grounded] Thirty-one, if we're counting. And thank you for having me. People always ask if it was lonely, and I never quite know how to answer."),
    (0, "[gentle, genuinely curious] Well... was it?"),
    (1, "[reflective, slowly opening up] Lonely implies you're missing something. Out there I had the wind, the gulls, the radio, and about four thousand books. What I didn't have was anyone to argue with - and it turns out, I miss arguing."),
    (0, "[wry, teasing] That's a very romantic way to describe deprivation."),
    (1, "[laughing it off, then turning earnest] Everyone says that! But consider this: the light had to be lit at dusk and doused at dawn, every single day, no exceptions. When your whole life is a routine that matters, you stop needing variety."),
    (0, "[leaning in, quieter] So what changed? Why come back to the mainland now?"),
    (1, "[heavy, quiet grief] The automation, of course. They put in a bulb that lasts ten years and a sensor that never sleeps. The sea <emphasis>didn't need me anymore</emphasis>. That was the day the island got smaller."),
    (0, "[sober, respectful] Let's talk about the storms. What was the worst night out there?"),
    (1, "[haunted, then a small smile] January of my ninth year. Ninety-knot gusts, waves over the lantern room. I clipped myself to the rail and I sang to the light until sunrise. [small laugh] Badly, I might add."),
]

text = "".join(f"<|speaker:{speaker}|>{turn} " for speaker, turn in TURNS)

env = post("/v1/tts", {
    "text": text,
    "references": [
        {"audio_b64": base64.b64encode(open("assets/ref-en0.wav", "rb").read()).decode(),
         "text": "<|speaker:0|> We asked over twenty different people, and they all said it was his."},
        {"audio_b64": base64.b64encode(open("assets/brona-ref.wav", "rb").read()).decode(),
         "text": "<|speaker:1|> Now, how does it get that memory? Through something we call the harness. And here's the beautiful part. All you need is a capable harness, and a single text file with instructions."},
    ],
    "seed": 42,
    "response_format": "json",
})
print("ok=", env.get("ok"), "chunks=", env.get("chunks"), "finish=", env.get("finish_reason"))
print("warnings:", env.get("warnings", [])[:3])
open("outputs/demo-directed.wav", "wb").write(base64.b64decode(env["audio_b64"]))
print("saved outputs/demo-directed.wav")

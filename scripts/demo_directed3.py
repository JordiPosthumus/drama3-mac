"""Same interview text as demo_directed.py, ONLY the direction briefs changed.

Variable isolation: identical lines, voices, refs, seed 42, temperature 0.8.
The briefs go from mood labels to over-the-top physical/behavioral direction.
"""

import base64
import json
import sys
import urllib.request

HOST = "http://127.0.0.1:8843"


def post(path, payload, timeout=1800):
    req = urllib.request.Request(HOST + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


TURNS = [
    (0, "[bursting out of the gate, grinning, leaning hard into the mic] Welcome back to Signal and Noise, the show about the machines we build and the machines that build us. Today my guest spent thirty years keeping a lighthouse on a rock in the North Atlantic."),
    (1, "[laughing under his breath, shaking his head] Thirty-one, if we're counting. And thank you for having me. People always ask if it was lonely, and I never quite know how to answer."),
    (0, "[dropping to a hush, eyes narrowing, savoring it] Well... was it?"),
    (1, "[slow exhale, staring out the window, then turning back with real warmth] Lonely implies you're missing something. Out there I had the wind, the gulls, the radio, and about four thousand books. What I didn't have was anyone to argue with - and it turns out, I miss arguing."),
    (0, "[snorting with delight, jabbing a finger at the table] That's a very romantic way to describe deprivation."),
    (1, "[barking a laugh, then leaning forward, deadly earnest] Everyone says that! But consider this: the light had to be lit at dusk and doused at dawn, every single day, no exceptions. When your whole life is a routine that matters, you stop needing variety."),
    (0, "[hooked now, forgetting to be a host, rushing the question] So what changed? Why come back to the mainland now?"),
    (1, "[the smile dying, voice going hollow, like he has said this to himself a thousand times] The automation, of course. They put in a bulb that lasts ten years and a sensor that never sleeps. The sea <emphasis>didn't need me anymore</emphasis>. That was the day the island got smaller."),
    (0, "[sitting back, chastened, almost whispering] Let's talk about the storms. What was the worst night out there?"),
    (1, "[far away for a moment, then steadying himself with a small smile] January of my ninth year. Ninety-knot gusts, waves over the lantern room. I clipped myself to the rail and I sang to the light until sunrise. [chuckles] Badly, I might add."),
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
    "seed": int(sys.argv[1]) if len(sys.argv) > 1 else 42,
    "response_format": "json",
})
print("ok=", env.get("ok"), "chunks=", env.get("chunks"), "finish=", env.get("finish_reason"))
warns = env.get("warnings", [])
print("warnings:", len(warns), warns[:2])
open(f"outputs/demo-directed3-s{sys.argv[1] if len(sys.argv) > 1 else 42}.wav", "wb").write(base64.b64decode(env["audio_b64"]))
print("saved outputs/demo-directed3-s{}.wav".format(sys.argv[1] if len(sys.argv) > 1 else 42))

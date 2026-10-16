"""Limit-test dialogue: maximum animation, every direction mechanism at once.

Stress targets: per-turn actor briefs with physical/behavioral language,
mid-line direction switches (incl. a Chinese brief on English text), all seven
paired delivery tags, inline sound events, em-dash interruptions, shouting vs
whispering dynamic range, turns that react to each other's words.
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
    (0, "[bursting with energy, leaning hard into the mic] Tonight — a story I have been waiting YEARS to tell you. [pause] The night the light went dark."),
    (1, "[groaning, embarrassed but smiling] Oh no. Oh, we are really starting there?"),
    (0, "[relentless, delighted] Oh, we are absolutely starting there. Thirty-one years. One storm. Zero light. <fast>Tell me everything. Everything.</fast>"),
    (1, "[holding up both hands, taking a slow breath] Okay. First — it was <stress>not</stress> my fault. The bulb was new. The generator was new. I had checked them twice."),
    (0, "[cutting in, triumphant, milking it] Twice! He checked twice, folks! And?! And?! What happened at 2:14 in the morning?!"),
    (1, "[the room going cold, voice dropping] [small inhale] The fillament— the filament glowed orange. Then <whisper>nothing</whisper>. No hum. No light. Just the storm laughing at me."),
    (0, ["[shocked, half out of his chair]"][0] + " You're joking. You are joking. [sharp breath] What do you mean, nothing?!"),
    (1, "[suddenly exploding, arms wide] It means I <forceful>climbed the tower</forceful> in ninety-knot winds, that is what it means!"),
    (0, "[delighted, slamming the desk] [laughs] He climbed the tower! You climbed the tower. In ninety-knot winds!"),
    (1, "[突然压低声音，像在自言自语] [suddenly whispering, talking to himself] I clipped myself to the rail. I sang to the glass. Badly. And the light... <slow>came back.</slow>"),
    (0, "[awed, barely above a whisper] [pause] <soft>And the light came back.</soft>"),
    (1, "[warm, the smile finally arriving] [chuckles] At dawn. Just in time for me to make breakfast like nothing happened."),
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
    "temperature": float(sys.argv[2]) if len(sys.argv) > 2 else 0.8,
    "response_format": "json",
})
print("ok=", env.get("ok"), "chunks=", env.get("chunks"), "finish=", env.get("finish_reason"))
warns = env.get("warnings", [])
print("warnings:", len(warns), warns[:2])
open(f"outputs/demo-directed2-s{sys.argv[1] if len(sys.argv) > 1 else 42}.wav", "wb").write(base64.b64decode(env["audio_b64"]))
print("saved outputs/demo-directed2-s{}.wav".format(sys.argv[1] if len(sys.argv) > 1 else 42))

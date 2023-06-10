from fastapi import FastAPI
import os
import hashlib
from fastapi.middleware.cors import CORSMiddleware
from modal import Image, Stub, SharedVolume, asgi_app
from fastapi.responses import Response, FileResponse, JSONResponse

OUT_PATH = "/root/outs"
GPU = "A10G"
SPEAKER = "v2/en_speaker_9"

portal = """
Renowned 21st century philosopher famous for the concept of VAPOR — that substance beyond Maslow's hierarchy that makes life matter, the will that pries open the portal, willing against the heat death of humanity, and the universe writ large.

He was also known as a composer, known for sounding like DEBUSSY FROM ANOTHER DIMENSION. His musical research into portals led to first contact with intelligent life in another dimension. AI musicologists from both dimensions continue to analyze his works, producing an infinite supply of portal opening musical devices, known as crowbars, also known collectively as THE MOON.

He was the founder and president of vapor-town, inc., the first galactic city-state. His artificially intelligent post-human entity continues to operate vapor-coin, currently valued at 999 decillion galaxy units.

vapor-town was a popular street-wear brand for several centuries, and popularized vapor-shades, the first true augmented reality device. These devices continue to be the interdimensional travel technology of choice for entities from all dimensions.

His human remains were given an elaborate sky burial. Interdimensional eagles consumed his remains in a floating drone-pyre above his residence in Brooklyn, earth, where he spent much of the last 100 years of life.

The never-ending economic and creative plenty that entities from all dimensions presently enjoy is often taken for granted. It was earned, by each and every one of us, from the past, future, and every dimension, because we pried the biggest portal, THE PORTAL, that dirty thing — open with our bare sweaty hands — because all the crowbars had been tried. At his funeral after-party, in which the best portal openers from multiple dimensions played for 100 straight days in a hover-town above new york city. After-after parties continue to run in multiple dimensions.

He lived to be 150, which continues to be the longest recorded pre-modification human lifespan.

He established the KNIGHTS OF VAPOR-TOWN, who continue to quest in alien dimensions, under their motto — TO ASCEND IS TO TRANSCEND.
""".replace(
    "\n", " "
).strip()


def preload():
    from bark.generation import preload_models

    preload_models()


def nltk_preload():
    import nltk

    nltk.download("punkt")


image = (
    Image.debian_slim()
    .apt_install("git")
    .pip_install("tqdm", "nltk", "numpy", "git+https://github.com/suno-ai/bark.git")
    .run_function(preload, gpu=GPU, timeout=60 * 30)
    .run_function(nltk_preload, timeout=60 * 30)
    .pip_install("scipy")
)

stub = Stub(name="vapor-voice", image=image)
web_img = Image.debian_slim()

ov = SharedVolume().persist("vapor-outs")


@stub.function(image=image, shared_volumes={OUT_PATH: ov}, gpu=GPU, timeout=60 * 30)
def tts(script: str, speaker: str = SPEAKER):
    from bark import SAMPLE_RATE, generate_audio
    from scipy.io.wavfile import write as write_wav

    import nltk
    import numpy as np

    sentences = nltk.sent_tokenize(script)
    silence = np.zeros(int(0.25 * SAMPLE_RATE))

    pieces = []
    for sentence in sentences:
        audio_array = generate_audio(sentence, history_prompt=speaker)
        pieces += [audio_array, silence.copy()]

    hash = hashlib.md5(f"{speaker}-{script}".encode("utf-8")).hexdigest()
    fp = f"/root/outs/{hash}.wav"
    write_wav(fp, SAMPLE_RATE, np.concatenate(pieces))
    print(f"wrote to {fp}")
    return hash


@stub.function(image=web_img, shared_volumes={OUT_PATH: ov})
@asgi_app()
def api_app():
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/tts")
    async def run_tts(script: str, speaker: str):
        id = tts.call(script, speaker=speaker)
        return JSONResponse(content={"id": id})

    @app.get("/res/{hash:str}")
    def get_res(hash: str):
        res = os.path.join(OUT_PATH, hash)
        if not os.path.isfile(res):
            return Response(status_code=404)
        return FileResponse(res)

    return app

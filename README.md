# glassfix

Turns a customer's car photo into a **transparent-background studio PNG** with clean, slightly grey,
semi-transparent tinted glass and even studio lighting. Everything is done by one **OpenAI image
edit** per photo (`gpt-image-2.5-sunburst`). There are no local AI models and no GPU.

Every call also sends a **style reference** (`assets/style_reference.png`) so all cars get the same look.

## Setup (Python 3.11+)

```bash
uv venv --python 3.11 .venv          # or: python -m venv .venv
uv pip install -r requirements.txt   # or: pip install -r requirements.txt
# optional, for iPhone HEIC photos:
uv pip install -r requirements-heic.txt
```

Set the API key as an environment variable (never commit it or paste it into chats):

```powershell
setx OPENAI_API_KEY "sk-..."        # Windows; then open a new terminal
```
```bash
export OPENAI_API_KEY="sk-..."      # macOS / Linux
```

Use a **restricted key** (Images only) with a **monthly spending cap** in the OpenAI dashboard.

## Use

```bash
python -m glassfix check                                   # free: key, model access, style reference
python -m glassfix render --input test_photos --output results --run-name job1             # medium
python -m glassfix render --input test_photos --output results --run-name job1 --quality high
python -m glassfix render ... --no-reference               # cheaper, less consistent look
```

Input: `<input>/<car_id>/*.jpg|png|webp|heic`, for example 12 photos per car.
Output in `<output>/<run-name>/`:

| File | Content |
|---|---|
| `<car_id>/<photo>.png` | the transparent studio PNG (about 1536×1024) |
| `costs.csv` | per photo: status, seconds, tokens, **cost in USD**, output size, error |
| `sheet.jpg` | labelled overview of all results on white |

A failed photo (after retries) is reported in `costs.csv` and the console, and the rest continue.

## Settings (`config.yaml`)

The settings cover the model, quality, upload size, parallel requests, retries, the prompt, the style
reference and the token prices used for the cost column. Edit the prompt there to change the look.

## Cost and speed (measured 2026-10-17, 4 photos, with style reference)

| Quality | Per photo | Per 12-photo car |
|---|---|---|
| low | about $0.024 | about $0.29 |
| **medium** (default) | **about $0.029** | **about $0.35** |
| high | about $0.054 | about $0.65 |

About 17–36 s per photo. Photos are processed `concurrency` at a time (default 3), so a 12-photo car
takes about 1.5–2 minutes. Without the style reference each photo is about $0.01 cheaper.

## Known limitations

- **The car is redrawn by AI, not edited pixel by pixel.** In tests OpenAI washed off mud and dirt,
  shifted the paint shade slightly, and varied the car's size and position in the frame. Small details
  (wheels, badges, panel lines) can differ between the 12 angles of the same car. Check whether the
  3D stitching tolerates this, and say in the client terms that images are AI-enhanced, since damage
  and dirt may disappear.
- **Output resolution** is about 1536×1024 regardless of the input size.
- **Customer photos are sent to OpenAI.** OpenAI says API data isn't used for training by default.
  Mention it in your privacy policy.
- **OpenAI rate limits** depend on your account tier and cap how many photos run in parallel.

## Tests

```bash
python -m pytest tests -q     # uses a fake OpenAI client: free and offline
```

## History

The earlier open-source pipeline (Grounding DINO, SAM 2, BiRefNet, classical styles) was removed in
favour of OpenAI. It is still in git history (commit `d8f9be7`) if it's ever needed again.

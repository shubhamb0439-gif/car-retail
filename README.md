# glassfix: car glass clean-up (proof of concept)

glassfix takes car photos, finds the window glass and removes reflections (sky, clouds, trees,
glare) from it. Real cracks in the glass are kept, and wipers are never repainted. Every photo of the
same car gets identical style parameters, so the results can be stitched into a 3D model.

## The deliverable: `cutout` (default)

For each photo glassfix writes **`<name>_cutout.png`**: the car on a **transparent background**, with
studio glass in every window. The client's software places it on its own (white) background and
stitches the 12 angles into a 3D model.

- The background (street, sky, other cars) is removed. The car is segmented by SAM 2 from its
  Grounding DINO box, and a guided filter softens the edges.
- **Clear windows** become slightly grey, **30% opaque** glass. On white this reads as the light studio
  glass of the reference photo, and reflections and anything seen through the car are gone.
- **Tinted windows** (detected per window) become dark grey, 85% opaque.
- Wipers and mirrors keep their original pixels, and cracks are drawn on the glass.
- The same glass colour and opacity on every angle, so the 12 photos match.
- `<name>_clean.jpg` is the same cut-out on white, for previewing only.

The interior is **not** shown (`cutout.interior_mode: none`). Keeping the real interior without also
keeping reflections and the street seen through the car was tested (`detect`, `luminance`, `both`
modes) and is not reliable from a single photo. See "Known limitations".

## Run it in a pod (Docker)

```bash
docker build -t glassfix:cpu .                                    # CPU image
docker build -t glassfix:gpu --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu132 .   # NVIDIA GPU

docker run --rm -v /jobs/123/in:/data/input -v /jobs/123/out:/data/output glassfix:cpu \
    run --input /data/input --output /data/output --run-name job123
# GPU pod: add --gpus all. Bigger CPU pod: add -e OMP_NUM_THREADS=16
```

- Input: `<input>/<car_id>/*.jpg` (12 photos per car). Output: `<output>/<run-name>/<car_id>/<name>_cutout.png`,
  plus a JSON per photo, `summary.csv` and `report.html`.
- The model weights (~1.6 GB) are baked into the image at build time, so a pod starts without
  downloading. Use `--build-arg PREFETCH_WEIGHTS=0` to download them on first run instead (mount
  `/app/models` as a volume so this happens once).
- The container runs as user id 10001, so the output volume must be writable by it.
- Memory: plan for about 6 GB of RAM on CPU (4 threads), or a GPU with 4 GB+ of memory.
- The image uses the headless OpenCV build (`requirements-pod.txt`); everything else matches
  `requirements.txt`.
- The image build has not been tested yet, because no Docker host was used during development. The
  same code and pins do run in the local Python 3.11 environment.

There is no HTTP API yet (out of scope for the PoC). The client app calls the container per job with
an input and an output folder. A thin API wrapper is the natural next step if the app needs one.

## All styles

| Style | What happens to the glass | Runs on |
|---|---|---|
| **`cutout`** (default) | Transparent PNG of the car with studio glass (above). | CPU / GPU |
| `cleanglass` | **No reflections at all.** Clean tinted glass; the interior shows through faintly wherever the photo actually captured it. Deterministic and identical across a car's photos. | CPU / GPU |
| `studiowhite` | Like `cutout` but keeps the photo background; light studio glass with interior silhouettes. | CPU / GPU |
| `clear` | The original glass is kept and only the broad glare/haze is removed. Sharp mirrored trees and clouds remain. | CPU / GPU |
| `tint` | The glass is replaced by an opaque dark smoked tint with a sky-to-ground gradient. | CPU / GPU |
| `studio` | `tint` plus one fixed studio-softbox highlight fitted to each window. | CPU / GPU |
| `dereflect` | Learned reflection removal (DSIT). Optional; it barely changes car glass (see below). | CPU (slow) / GPU |
| generative (`tools/`) | **Truly see-through glass:** Qwen-Image-Edit-2509 redraws the glass as clear. Dereflection-Any-Image is included for comparison only. | GPU (Colab/Kaggle) |

This is a PoC for comparing methods and showing results. There is no API, database, auth or billing.

```
photo ─► Grounding DINO / Florence-2 (boxes) ─► SAM 2 (pixel masks) ─► mask cleanup
      ─► LAB L channel: low-frequency layer (reflections) + high-frequency layer (detail, cracks)
      ─► crack detection (ridge filter + thin/long + local-contrast test)
      ─► cleanglass:   tint + (haze-free interior, only where it is not reflection) + preserved cracks
          clear:        subtract the estimated reflection veil (glass, interior and cracks kept)
          tint/studio:  new glass = tint (+ softbox highlight) + 15% original shading + preserved cracks
      ─► feathered composite (pixels outside the glass mask are never changed)
```

## Setup

You need Python **3.11**. The examples use [uv](https://docs.astral.sh/uv/) (fast, and it installs
Python 3.11 for you). Plain `python -m venv` + `pip` also works.

### Windows (PowerShell)

```powershell
uv venv --python 3.11 .venv
.venv\Scripts\activate
uv pip install torch==2.14.1 torchvision==0.29.1 --torch-backend=auto   # picks the CUDA build if you have an NVIDIA GPU
uv pip install -r requirements.txt
python -m glassfix doctor
```

### Linux

```bash
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install torch==2.14.1 torchvision==0.29.1 --torch-backend=auto
uv pip install -r requirements.txt
python -m glassfix doctor
```

With plain pip on an NVIDIA machine, install torch from the CUDA index shown on
<https://pytorch.org/get-started/locally/> before `pip install -r requirements.txt`.

### macOS (Apple Silicon or Intel)

```bash
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -r requirements.txt        # the default torch wheel includes Apple MPS support
python -m glassfix doctor
```

On Apple Silicon the device is `mps`. glassfix sets `PYTORCH_ENABLE_MPS_FALLBACK=1`, so any operation
MPS lacks falls back to the CPU instead of crashing. If MPS gives errors, use `--device cpu`.

### Google Colab / Kaggle (free GPU)

Open `notebooks/colab_demo.ipynb` in Colab and follow the first cell. In short: upload this folder and
your photos to Google Drive, choose a T4 GPU runtime, and run all cells. The report is saved back to
Drive. On Kaggle, run the same commands in a notebook with the GPU enabled, using
`requirements-colab.txt`.

### Low-memory Windows laptops

The CUDA build of torch reserves about 1.5 GB of memory on import, even when running on CPU. If you
see "Unable to allocate", "paging file is too small" or a silent crash, either close other apps
(browsers) or use a CPU-only environment, which needs about 0.7 GB less:

```powershell
uv venv --python 3.11 .venv-cpu
uv pip install --python .venv-cpu\Scripts\python.exe torch==2.14.1 torchvision==0.29.1 --torch-backend=cpu
uv pip install --python .venv-cpu\Scripts\python.exe -r requirements.txt
.venv-cpu\Scripts\python.exe -m glassfix run ...
```

### Optional: HEIC photos (iPhone)

```bash
uv pip install -r requirements-heic.txt
```

This is optional because the pillow-heif wheels bundle LGPL libraries (see LICENSES.md).

### Model weights

The weights (about 1.5 GB in total) download to `./models` on first use. That folder is git-ignored.
To fetch them ahead of time, run `python -m glassfix doctor --download`. `doctor` shows `[x]` for
every model that is already downloaded.

## Usage

```bash
# one method (default style: cutout)
python -m glassfix run --input test_photos --output results --detector groundingdino_sam2 --run-name demo1

# every detector × style × crack-preservation combination in one side-by-side report
# (clear runs once per detector, since it never removes cracks)
python -m glassfix compare --input test_photos --output results --run-name compare1

# environment check
python -m glassfix doctor
```

Generative see-through glass (GPU; easiest in `notebooks/colab_demo.ipynb`):

```bash
python -m glassfix crops   --input test_photos --output results --run-name gen1      # export glass crops
python tools/edit_qwen.py  results/gen1/crops results/gen1/edits_qwen               # own env: tools/requirements-qwen.txt
python -m glassfix compose --run-name gen1 --edits qwen=results/gen1/edits_qwen     # paste back + report
```

The editors run in their own environments, because they need different library versions. `compose`
only changes pixels inside the glass mask, so the body, wipers and background stay original.

Useful flags: `--device cpu|cuda|mps` (default: auto-detect), `--max-side 2048` (processing
resolution), `--no-preserve-cracks`, `--config my_config.yaml`.

All tunable numbers (prompts, thresholds, blur sizes, tint colours, highlight strength, crack filter)
are in **`config.yaml`**, and each one has a comment.

## Adding test photos

Put one folder per car under `test_photos/`:

```
test_photos/
  my_car_001/
    front.jpg
    rear_left.jpg
    side.heic
  my_car_002/
    ...
```

Every photo inside a car folder gets exactly the same style. JPG, PNG and HEIC (optional extra) are
supported, and EXIF rotation is applied. Processing runs at 2048 px on the long side, while outputs
keep the original resolution. The included photos are CC0 (see `test_photos/SOURCES.md`).

## Outputs

For each photo, under `results/<run_name>/<car_id>/` (in `compare` mode, under
`results/<run_name>/<variant>/<car_id>/`):

| File | What it is |
|---|---|
| `<name>_clean.jpg` | final image, original resolution |
| `<name>_glass_mask.png` | where glass was found (white) |
| `<name>_crack_mask.png` | preserved crack pixels (white) |
| `<name>_compare.jpg` | original, mask overlay and result side by side |
| `<name>.json` | detector, glass found, glass area %, crack flag, boxes, per-step timings, device, style fingerprint |

Each run also writes `summary.csv` (one row per photo and variant), `run_config.json` (the exact
config used) and `report.html`.

## Reading the report

`report.html` is one self-contained file with no internet access needed, so you can email it or open
it offline.

- **Summary tiles:** photos processed, the share of photos where glass was found, how many photos were
  flagged as cracked, and average time per photo.
- **Before/after slider:** drag it, or click it and use the arrow keys.
- **Style line per car:** `style <fingerprint>: identical style on all photos` proves the car's photos
  used identical style parameters. A warning appears if they ever differ.
- **Show detected glass:** blue marks the glass mask and red marks preserved cracks. Use this to tell a
  detection problem (wrong area) from a styling problem (right area, wrong look).
- **Comparison report:** a table by variant, then for each photo a button per variant (or click a
  thumbnail) that drives the slider. Detection overlays are shown once per detector.

## Tests

```bash
python -m pytest tests -q
```

The tests run on synthetic windows with no models needed. They check that:
- a gradient reflection is removed (tint/studio: down to the configured 15% shading; clear: at least half);
- clear keeps the interior seen through the glass and the crack detail;
- a drawn thin crack, with a branch, is detected (≥ 80% of its pixels) and visible in the result;
- the crack disappears when preservation is off;
- clean glass and busy interior detail behind the glass are not flagged;
- pixels outside the mask are bit-identical;
- output is deterministic and at full resolution.

## About `cleanglass` (tinted-glass look with the photo background)

If the glass mirrored bright sky, the camera recorded almost nothing of the interior there. No method
can recover it; it can only be guessed. `cleanglass` doesn't guess. Wherever most of a pixel's light
was reflection (or the glass is blown out), it shows clean tinted glass. Where the interior was
actually captured (dark and mid tones), a desaturated share of it shows through. The result never
contains clouds or trees, and it is identical for every photo of a car. The thresholds are under
`style.cleanglass` in `config.yaml`.

For glass you can see through everywhere, the interior has to be redrawn by a generative model (the
Qwen step above). That can differ slightly between photos of the same car, so check the set before
stitching.

## How `clear` works (and what it can't do)

A reflection on glass is extra light *added* on top of what is behind the glass. So even the darkest
pixel in a small neighbourhood still carries that added light. glassfix takes a local lower envelope
of each colour channel (a morphological opening over about 25 px), smooths it into a **veil map**, and
subtracts 90% of it in linear light. Thin details (seat edges, the steering wheel, stickers, cracks)
never survive the envelope, so they are kept exactly. This is the "dark channel" idea from image
dehazing, applied to glare. The settings are under `style.clear` in `config.yaml`.

It removes **broad glare and sky haze** well. It does **not** remove **sharp mirrored objects**
(tree branches, power lines, buildings reflected in the glass). Those look like real detail behind
the glass, and once the haze is gone they can even look slightly stronger. Removing them needs a
learned reflection-separation model. That is the main next step (see limitations).

## About `dereflect` (DSIT)

DSIT is a published reflection-separation network (XReflection model zoo, NTIRE-winning team). It is
built in and works, but on car glass it predicts only a faint reflection. It was trained on photos
through house and shop windows, where reflections are weak, so a windshield mirroring bright sky is
outside what it learned. On CPU it needs about 4 GB of RAM and about 20 s per photo. It is kept for
comparison and is not in the default `compare` set.

## Known limitations

- **No interior in the cut-out (by design).** In a single photo, the seat behind the glass, the tree
  seen through the far window and the reflected cloud are all mixed into the same pixels. Keeping only
  the interior needs a learned or generative model (the Qwen step), or better capture: a polarising
  filter, or app guidance to avoid shooting toward bright sky.
- **The car outline comes from SAM 2.** It is good, but thin parts (antennas, roof rails) and the
  shadow under the car are not perfect. The ground shadow is not included; the client's software
  should add its own.
- **Windows can be missed.** Grounding DINO can miss small quarter windows; a missed window keeps its
  original look in the cut-out.

- **Missed windows.** Grounding DINO sometimes misses small side or rear quarter windows. Those windows
  keep their original look, which breaks consistency between views. Lowering `box_threshold` finds more
  windows but also more false ones.
- **Wrong objects as glass.** Headlights, taillights and the hood can be detected as glass. glassfix
  rejects boxes centred low on the car (`max_glass_center_y_in_car`) and leaves "sunroof" out of the
  prompt, because in eye-level photos it matched hoods and headlights.
- **Crack detection is classic image processing, not a trained model.** It is tuned to stay quiet on
  the busy interior seen through clear glass. As a result it will **miss cracks that run over
  textured areas** (for example, across a visible seat) and faint cracks. Rear-window defroster lines
  and wiper edges can still be flagged. None of the included real photos shows a crack, so real-crack
  recall has only been tested on synthetic images. Test it on your damaged-glass photos.
- **`cleanglass` hides the interior where the glass was mostly reflection.** For example, the
  Matador's windshield becomes almost opaque clean glass. That is by design (nothing behind it was
  captured), but it isn't see-through. Use the generative step for that.
- **The generative step is untested on this PoC's laptop.** It needs a 16 GB+ GPU. The tools follow the
  official documentation, and the crops → edit → compose hand-off was verified locally with a dummy
  editor.
- **`clear` leaves sharp reflections.** Mirrored trees, power lines and buildings stay in the glass, as
  described above. The fix is a learned single-image reflection-removal model, which needs a careful
  license check because many published ones are research-only.
- **`clear` assumes something dark is visible behind the glass** within about 25 px (seats, trim,
  shadow). Over a large, evenly bright area (for example a white headliner seen through a sunroof), it
  cannot tell interior from glare and will darken it.
- **`tint`/`studio` glass is opaque, by design.** The tint hides the interior. Use `clear` to keep it.
- **Mask edges.** SAM 2 edges are good but not perfect around wipers, mirrors and rubber seals. The
  2–3 px inward feather hides most seams.
- **JPEG re-encoding.** The saved `_clean.jpg` is re-encoded at quality 95, so pixels outside the glass
  are unchanged in the image data but go through JPEG compression once more. A lossless PNG output
  would be a small change if the 3D pipeline needs it.
- **Memory on Windows.** With the CUDA build of torch, the process commits about 6 GB of virtual memory
  (mostly the CUDA libraries). On a machine with little free memory, `--device cpu` needs about 4 GB.
- **Speed.** About 5 s per photo on an RTX A500 laptop GPU and about 19 s on CPU (Grounding DINO path,
  2048 px). Florence-2 is slower on CPU.

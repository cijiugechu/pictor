# Anima MLX experiments

Two independently selectable model families were evaluated against pictor's
pinned P3 Turbo GGUF. These measurements preceded public API integration. The
retained P3 BF16 path is now the Apple Silicon default; see the
[production setup/API guide](anima-mlx.md). The comparison executables are native C++.

- Public `xocialize/anima-mlx`: base v1 weights, BF16 or selectively INT4 DiT.
- Local P3: exact existing Turbo GGUF dequantized with the linked ggml Q4_K
  function, rounded to BF16; no re-quantization or replacement Turbo LoRA.

The reference implementation is pinned at
`2338e281ee533568db558963f6e3da064eb60ff0`; weights at
`4cc65eea48a3b57373ccbe1467c13c38c68b1b3f`. See
`benchmarks/anima-mlx-model.json` for hashes and `licenses/anima-mlx.txt` for
attribution. The C++ operator port follows that implementation; prompt parsing,
Philox, ER-SDE and SmoothStep follow pictor's pinned sd.cpp. Consequently this is
not the Python reference's default NumPy RNG/normal-schedule pipeline.

## Preparation

```sh
zig build anima-benchmark-build anima-mlx-build anima-ggml-probe-build
bash scripts/prepare-anima-reference.sh
python3 scripts/download-anima-mlx.py
# Only conversion/reference checks need this isolated Python environment:
test -x build/mlx-venv/bin/python || uv venv --python 3.12.11 build/mlx-venv
uv pip install --python build/mlx-venv/bin/python \
  'mlx==0.32.2' 'mlx-metal==0.32.2' 'numpy==2.5.3'
build/mlx-venv/bin/python scripts/convert-anima-p3.py
```

Conversion accepts only the pinned P3 SHA-256. It streams one tensor at a time,
reconstructs flattened 3D convolution shapes, maps canonical module keys, and
writes a conversion manifest with source/output hashes. The image-only export
omits the VAE encoder. Retained BF16 tensors are exact; Q4_K dequantized floats
are rounded to BF16, which adds rounding error. BF16 storage is approximately
5.14 GiB. FP32 export is available for analysis via `--dtype f32 --output NEW_DIR`;
the current benchmark loader intentionally selects the BF16 filenames.

## Stage checks

```sh
build/mlx-venv/bin/python scripts/anima-reference-fixture.py \
  --model models/anima-p3-mlx-bf16 --output outputs/anima-mlx/new-fixture
PICTOR_ANIMA_WEIGHTS=models/anima-p3-mlx-bf16 \
PICTOR_ANIMA_FIXTURE=outputs/anima-mlx/new-fixture \
  zig-out/bin/pictor_anima_mlx_bench
zig-out/bin/pictor_anima_ggml_probe models/Anima-P3-Turbo-AIO-Q4_K.gguf \
  outputs/anima-mlx/new-fixture
build/mlx-venv/bin/python scripts/check-anima-parity.py outputs/anima-mlx/new-fixture
```

Python and C++ must receive identical IDs, latent and context. Reference-policy
C++ text and adapter were exact on the first 64px fixture; DiT maximum absolute
error was 1.13e-5 and decoded RGB-float error 1.28e-6. Small differences arise from
RoPE table construction and replacing zero-history, single-frame 3D convolutions
with their contributing 2D slice. This says nothing about perceptual quality or
cross-backend exactness. The checker reports ggml differences separately.

## Measurement

```sh
python3 scripts/benchmark-anima.py --output outputs/anima-mlx/new-ggml
python3 scripts/benchmark-anima.py --backend mlx --output outputs/anima-mlx/new-p3
python3 scripts/benchmark-anima.py --backend mlx --bf16-compute \
  --output outputs/anima-mlx/new-p3-bf16
python3 scripts/benchmark-anima.py --backend mlx --int4 \
  --bf16-compute --model models/xocialize-anima-mlx --steps 16 --cfg 4.5 \
  --negative 'worst quality, low quality, blurry, deformed' \
  --output outputs/anima-mlx/new-public
```

Run GPU workloads serially. Each directory must be new. Defaults: 512x768, three
steps, CFG1, seed666, cache none, one warmup and three measured resident calls.
The public base model needs an independently selected recipe; its quality/speed
comparison includes checkpoint, CFG and step-count differences. Compare P3 with
identical requests to assess the runtime change. Also compare the existing
Spectrum preset before recommending a production default change.

Each process checks exact repeated RGB. Reports separate load, text+adapter,
sampling, decode, generation, process wall time, peak RSS, macOS footprint and
MLX allocation peak when available. Reference-policy means BF16 weights with
FP32 promotions in the original reference; `--bf16-compute` keeps major model
activations BF16 while sampler/latent arithmetic stays FP32. The latter is an
additional numerical change and requires its own visual assessment.

The standalone experimental driver has no Spectrum, tiling, batch/FFI integration
or reference-image input. The production public API adds batches and optional
tiled decode. `--count` in these benchmark executables means identical
resident repetitions, not incrementing seeds. Outputs and raw logs remain under
ignored `outputs/anima-mlx/`; the retention decisions and portable results are below.

## Completed 512x768 fast comparison

M4 MacBook Air / 32 GiB, same local P3 Turbo checkpoint, prompt in the manifests,
seed666, three ER-SDE/SmoothStep steps, CFG1, no Spectrum or tiling. Each column is
one warmup followed by three resident calls; all repeated images were RGB-exact.
These sequential runs are not thermally controlled laboratory measurements.

| Median seconds | ggml Q4_K | MLX, reference compute | MLX, BF16 compute |
| --- | ---: | ---: | ---: |
| Text (+adapter on MLX) | 0.070 | 0.466 | 0.355 |
| Sampling | 39.240 | 33.591 | 23.316 |
| Decode | 11.920 | 2.258 | 1.444 |
| Generation | **51.470** | **36.340** | **25.141** |

Reference-policy MLX takes 29.4% less time (1.42x throughput); BF16 takes 51.2%
less time (2.05x throughput). Adapter accounting differs: ggml includes repeated
adapter work in sampling; MLX computes request-constant context once. Thus this
is the complete port's benefit, not an isolated GEMM/backend-only claim.

The native reference-policy peak MLX allocation was 7.326 GB, macOS footprint
12.417 GB; BF16 was 6.423 GB and 8.956 GB respectively. Both fit this host.
The ggml process RSS was 3.514 GB; RSS, allocator peak and physical footprint are
different measures and must not be treated as interchangeable.

Visual inspection: lake, mountain, cottage and paths retain similar composition;
both ports retain the low-step stippled texture of the baseline. Differences in
small objects and textures are visible; no quality improvement is claimed for
P3. Reference-policy versus ggml MAE is 6.89/255 and PSNR 25.63 dB, which measure
difference only. The P3 route meets the user's speed-based retention criterion.
The later public integration retains the same BF16/no-cache generation path.

The default Balanced recipe comparison (8 steps) also favors MLX: ggml with
Spectrum has a resident median of **92.872s** (Spectrum skips 2/8 steps), whereas
P3 MLX BF16 without a cache takes **65.912s**, a **29.0%** time reduction / **1.41x**
throughput. Both used one warmup and three RGB-exact repeats. Sampling medians
were 80.230s vs 63.677s; decoding 12.240s vs 1.554s. MLX footprint peaked at
8.973 GB. The uncached native image shows clearer mountain, tree and cottage
detail than the cached baseline; both include an unwanted person. This quality
difference changes caching as well as runtime/precision and cannot be attributed
solely to dequantization or MLX.

Additional quality controls use the prompts in `benchmarks/anima-cases.json`:

```sh
python3 scripts/anima-quality-samples.py
python3 scripts/anima-comparison-report.py \
  --snapshot benchmarks/anima-results-2026-09-26.json
```

This serial suite produces one image per setting (no warmup). Those times are
single-call observations, not resident medians. It checks the existing P3
16-step Quality preset, a second portrait prompt/seed across both model families,
and weighted Unicode prompts with CFG/negative conditioning in both runtimes.

P3 controls already show why steps/cache must be stated: the original 16-step
Quality landscape is clean and follows `no people`, taking 238.492s in one cold
call. It is a flatter illustration with a prominent cottage; the public v1
16-step image has denser foliage and mountain texture. The latter is not enough
to establish an overall quality ranking across different drawing styles.

The second P3 prompt (portrait, seed42) produces coherent faces, clasped hands,
scarf/coat and snowy streets in both implementations. Native BF16/no-cache has
crisper linework and altered details; no obvious new large artifact was observed.
Its single-call time is 63.238s versus ggml/Spectrum 95.952s. These observations
do not replace the resident medians above or establish pixel equivalence.

Public INT4/reference-compute portraits take 360.686s in the tested 16-step,
CFG4.5 recipe (landscape 373.684s). Snowy village architecture and roof snow are
more detailed than the tested P3 Balanced portrait; the face is not clearly
better. This supports a background-detail/style alternative, not a universal
quality claim. Public and P3 checkpoint, steps, CFG and negative prompts differ.

The 256px weighted Chinese/English CFG2 probe completes in both runtimes
(ggml 15.440s, native BF16 9.501s, single calls), but both images are overexposed
and distorted. It is a functional conditioning check, **not a visual-quality
pass**; P3 speed/retention conclusions use the recommended CFG1 samples above.

Public INT4 with BF16 compute takes **285.659s** for the same landscape/16-step/
CFG4.5 request, versus reference-compute 373.684s. These are single-call
observations, not a warmed benchmark. BF16 preserves the cottage/lake/mountain
composition and fine background detail with small visible differences. Its
MLX peak allocation is 3.892 GB and macOS footprint 6.581 GB. It remains slower
than the tested P3 recipes, including original ggml Quality's 238.492s observation.

## Retention decision

- **Retain P3 dequantized MLX, prefer BF16 compute for this experimental path.**
  It demonstrates 2.05x throughput on the same 3-step/no-cache workload and
  1.41x against the existing 8-step Spectrum default. Regular CFG1 landscape and
  portrait samples show no obvious major new artifacts. Precision and pixels
  differ; dequantization does not recover information already lost to Q4_K.
- **Retain public v1 as an optional detail/style alternative.** The inspected
  landscape has denser foliage/mountain detail, and the portrait has richer
  snowy village backgrounds. This is a limited visual advantage in two samples,
  not evidence of universally better quality; the face is not clearly better.
  It has no demonstrated speed advantage over the tested P3 recipes. BF16 compute
  is the faster public option tested; reference compute remains available.
- **Subsequent integration:** at the user's request, native MLX is now available
  through Anima's public C++/C APIs and is the Apple Silicon CLI default. ggml
  remains explicitly selectable. The benchmark executables remain for comparing
  numerical policies; see [the production guide](anima-mlx.md) for batches,
  tiling, migration and integration validation. Public ABI layouts are unchanged.

All results above are from a 32 GiB M4 host at the stated resolutions. The P3
conversion completed locally; its 5.14 GiB export and approximately 9 GB native
BF16 process footprint fit this host. No claim is made for larger resolutions.

The image gallery is `outputs/anima-mlx/comparison.html`; it links original PNGs,
raw logs and manifests. Portable timing records, model/binary hashes, PNG hashes,
stage comparisons and a source snapshot are retained in
`benchmarks/anima-results-2026-09-26.json`. The measurements are sequential and
not thermally controlled; only the five warmup-plus-three-repeat configurations
support resident-median claims. All other times are explicitly single-call
observations. Quality judgments are visual inspection of these samples, without
a blind panel or aggregate quality score.

Validation: Metal tests/FFI and experimental targets passed (75/75 build steps),
CPU/no-MLX tests/FFI passed (59/59), final experiment rebuild passed (34/34).
Both independent Python/native fixtures pass the numerical checker, and the
checker rejects a missing reference. Help/invalid-dimension/existing-output
preflight, license installation, Python/shell/Zig syntax and whitespace checks
pass. Native executables do not load Python.

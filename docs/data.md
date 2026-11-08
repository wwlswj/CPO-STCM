# Data

## CPO-LED Bench v2.0

The software consumes the public CPO-LED Bench v2.0 data payload. The immutable
data release is separate from this software repository:

`https://github.com/wwlswj/CPO-LED-Bench/releases/tag/v2.0.0`

The Zenodo record for citation version `v2.0.1` is
`https://doi.org/10.5281/zenodo.24031281`.

Pinned properties:

| Property | Value |
|---|---:|
| Archive name | `CPO-LED-Bench-v2.0-full.zip` |
| Size | `395563944` bytes |
| SHA-256 | `a462f4c144b4675aaeb7ccdc819066653fd9500615843996f941226c05a3b19d` |
| Data license | CC-BY-4.0 |

`cpo-stcm data fetch` checks the remote content length, writes a partial file,
validates SHA-256, and only then installs the archive.

## Image-to-array conversion

The temporal workflow expects a memory-mappable array shaped as:

```text
(packed_windows, 10, 100, 300)
```

`cpo-stcm data build-packed` reconstructs that array from the published PNG
difference images. It uses the stable `packed_index` in
`manifests/cpo_stcm_samples.csv`, validates ten frames per base window, and
checks each image's dimensions.

Use `--limit 2` for a fast smoke test against real dataset images.

## Dataset version correction

The data payload remains `v2.0.0`. A separate `v2.0.1` metadata correction adds
the confirmed citation order Wanli Wang, Huafeng Chen, Xian Wei and a persistent
dataset DOI without changing the image payload or its SHA-256. The version DOI
is `10.5281/zenodo.24031281`; the concept DOI is
`10.5281/zenodo.24031280`.

## Original raw frames

Recomputing static YOLO detections from raw full-size LED frames requires an
additional raw-frame dataset that is not part of the public image-only CPO-LED
archive. The public archive supplies frozen static evidence for reproduction.

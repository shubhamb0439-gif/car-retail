# Licenses and services

## Python libraries (`requirements.txt`)

| Package | Version | License |
|---|---|---|
| openai (Python SDK) | 3.31.0 | Apache-2.0 |
| pillow | 12.3.0 | MIT-CMU (HPND, permissive) |
| pyyaml | 6.0.3 | MIT |
| pytest (tests only) | 9.1.1 | MIT |
| pillow-heif (optional, HEIC input) | 1.8.1 | BSD-3-Clause; its wheels bundle libheif/libde265 (**LGPL-3.0**, dynamically linked) |

## External service

- **OpenAI API** (`gpt-image-2.5-sunburst`): a paid, proprietary service under OpenAI's terms of use and
  usage policies. Every photo processed is sent to OpenAI. Under OpenAI's terms, the customer owns the
  outputs. Review OpenAI's current Services Agreement and data-retention settings before production.

## Assets

- `assets/style_reference.png`: style example provided by the project owner. Internal use.

## Test photos

- `test_photos/` and `test_photos_archive/`: CC0 photos from Wikimedia Commons (see the `SOURCES.md`
  files), except `test_photos/ford_ecosport_user/`, which was provided by the project owner for internal
  testing only.

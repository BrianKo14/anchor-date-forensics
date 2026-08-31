"""Image decoding, normalization and compression-parity checks.

`prepare_image` is the load-bearing function in this repo. AI-GenBench's assembly script ships with
`make_jpeg_dataset = False`, which leaves the fake half in native containers (PNG, JPEG, WEBP) while
the real half arrives as web JPEG. Written out as-is, the two classes would be separable on container
format alone -- a "Fake or JPEG?" artifact with nothing to do with synthesis.

Running *both* halves through this one function is what prevents that. It is a port of AI-GenBench's
`prepare_image(convert_to_jpeg=True, jpeg_quality=95)`, i.e. the `make_jpeg_dataset = True` variant.
If you change it, both halves must be rebuilt together, or compression history silently becomes a
class cue.
"""

import io

from PIL import Image

from common import IMAGE_MIN_SIZE, JPEG_QUALITY

# The libjpeg default luminance table, i.e. what a quality-75 re-encode leaves behind. HuggingFace's
# datasets-server hands out images re-encoded this way; they must never reach the sample.
Q75_LUMA = (8, 6, 5, 8, 12, 20, 26, 31)


def decode_and_validate(raw):
    """verify() -> reopen -> load() -> size floor. Returns the image, or None if it doesn't pass."""
    try:
        Image.open(io.BytesIO(raw)).verify()  # verify() leaves the handle unusable...
        image = Image.open(io.BytesIO(raw))   # ...so reopen before actually decoding
        image.load()
    except Exception:
        return None
    if min(image.size) < IMAGE_MIN_SIZE:
        return None
    return image


def prepare_image(image):
    """Port of AI-GenBench prepare_image(convert_to_jpeg=True). Returns encoded JPEG bytes.

    Returns bytes rather than a PIL.Image on purpose: handing back an image would invite the caller
    to save it again, putting a second JPEG generation on every file.
    """
    image.info.pop("xmp", None)
    if image.mode != "RGB":
        if image.mode != "RGBA":
            image = image.convert("RGBA")
        background = Image.new("RGBA", image.size, (255, 255, 255))
        image = Image.alpha_composite(background, image).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=JPEG_QUALITY)
    return buffer.getvalue()


def luma_table(path):
    """The image's luminance quantization table -- its compression fingerprint."""
    with Image.open(path) as image:
        assert image.format == "JPEG", f"{path.name} is {image.format}, not JPEG"
        return tuple(image.quantization[0][:8])


def check_parity(authentic_dir, fakes_dir, verbose=True):
    """Assert both halves carry one and the same quantization table.

    They should, because both were produced by `prepare_image` at the same quality. A divergence
    means something skipped normalization -- a class cue that appears nowhere in the manifest -- so
    this raises rather than warns.
    """
    fake_images = sorted((fakes_dir / "images").glob("*.jpg"))
    authentic_images = sorted((authentic_dir / "images").glob("*.jpg"))
    if not fake_images or not authentic_images:
        raise RuntimeError("both halves must be imported before parity can be checked")

    fake_tables = {luma_table(p) for p in fake_images}
    authentic_tables = {luma_table(p) for p in authentic_images}

    assert len(fake_tables) == 1, f"fakes are not uniformly encoded: {fake_tables}"
    assert len(authentic_tables) == 1, f"authentics are not uniformly encoded: {authentic_tables}"
    assert Q75_LUMA not in fake_tables, "fakes carry the q75 default table -- a re-encode slipped in"
    assert authentic_tables == fake_tables, (
        f"compression history differs between halves: {authentic_tables} vs {fake_tables}"
    )

    table = next(iter(fake_tables))
    if verbose:
        print(f"authentics qtab0[:8] = {list(table)}  ({len(authentic_images)} images)")
        print(f"fakes      qtab0[:8] = {list(table)}  ({len(fake_images)} images)")
        print("both halves share one quantization table -- compression history is not a class cue")
    return table

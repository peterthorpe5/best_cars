"""Conservative, reproducible model identity matching."""

import re
import unicodedata
from dataclasses import dataclass

MAKES = (
    "Alfa Romeo",
    "Aston Martin",
    "DS Automobiles",
    "GM Daewoo",
    "Land Rover",
    "Mercedes-Benz",
    "Volkswagen",
    "Mitsubishi",
    "Chevrolet",
    "Chrysler",
    "SsangYong",
    "Vauxhall",
    "Porsche",
    "Peugeot",
    "Renault",
    "Bentley",
    "Cadillac",
    "Citroen",
    "Citroën",
    "Ferrari",
    "Hyundai",
    "Infiniti",
    "Jaguar",
    "Lancia",
    "Polestar",
    "Suzuki",
    "Toyota",
    "Dacia",
    "Daewoo",
    "Dodge",
    "Honda",
    "Lexus",
    "Mazda",
    "Nissan",
    "Skoda",
    "Maserati",
    "Subaru",
    "Rover",
    "MG",
    "Daihatsu",
    "Isuzu",
    "Škoda",
    "Smart",
    "Tesla",
    "Volvo",
    "Cupra",
    "Audi",
    "BMW",
    "Fiat",
    "Ford",
    "Jeep",
    "Kia",
    "Lada",
    "Mini",
    "Opel",
    "Saab",
    "Seat",
    "SEAT",
    "KIA",
    "VW",
    "DS",
)
MAKE_ALIASES = {
    "vw": "Volkswagen",
    "volkswagen": "Volkswagen",
    "skoda": "Škoda",
    "citroen": "Citroën",
    "mercedes benz": "Mercedes-Benz",
    "mercedes": "Mercedes-Benz",
    "seat": "SEAT",
    "kia": "Kia",
    "bmw": "BMW",
    "mini": "Mini",
    "smart": "Smart",
    "ds automobiles": "DS",
}


@dataclass(frozen=True)
class ModelIdentity:
    """A model series identity, never an engine or generation prediction."""

    make: str
    model: str
    model_id: str
    grouped: bool


def normalise_text(*, text: str) -> str:
    """Return an accent-insensitive text key with harmless punctuation removed.

    Args:
        text: A non-empty source label.

    Returns:
        A lower-case, space-normalised key.

    Raises:
        ValueError: If the label is empty or not a string.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("A non-empty model label is required.")
    decomposed = unicodedata.normalize("NFKD", text)
    plain = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", plain.lower()).strip()


def identify_model(*, label: str, make_hint: str | None = None) -> ModelIdentity:
    """Match explicit model aliases while retaining grouped source models.

    Args:
        label: Full source make/model, or model when make_hint is supplied.
        make_hint: Optional explicit make from a structured source.

    Returns:
        Auditable make/model identity. Opel and Vauxhall remain separate.

    Raises:
        ValueError: If a make or model cannot be identified.
    """
    normalise_text(text=label)
    cleaned = re.sub(r"\s+", " ", label.replace("\n", " ")).strip()
    if make_hint:
        make = MAKE_ALIASES.get(
            normalise_text(text=make_hint), make_hint.strip().title()
        )
        model = cleaned
    else:
        make = ""
        model = ""
        for candidate in MAKES:
            if cleaned.casefold() == candidate.casefold():
                make = MAKE_ALIASES.get(normalise_text(text=candidate), candidate)
                return ModelIdentity(
                    make=make,
                    model="Unspecified model series",
                    model_id=f"{normalise_text(text=make)}:unspecified",
                    grouped=True,
                )
            if cleaned.casefold().startswith(candidate.casefold() + " "):
                make = MAKE_ALIASES.get(normalise_text(text=candidate), candidate)
                model = cleaned[len(candidate) :].strip()
                break
        if not make:
            raise ValueError(f"Unrecognised make in model label: {label!r}")
    if not model:
        raise ValueError(f"Missing model in label: {label!r}")
    model_key = normalise_text(text=model)
    grouped = bool(re.search(r"/|\binkl\b|\btotal\b|,", model, flags=re.I))
    # These are label translations only; different derivatives stay separate.
    if make == "Mercedes-Benz":
        match = re.fullmatch(r"([abces])(?: klasse| class)?", model_key)
        if match:
            model = f"{match.group(1).upper()}-Class"
    if make == "BMW":
        match = re.fullmatch(r"([1-8])(?:er reihe| sarja| series)", model_key)
        if match:
            model = f"{match.group(1)} Series"
    if make == "Volkswagen" and model_key.startswith("nutzfahrzeuge "):
        model = model[len("Nutzfahrzeuge ") :]
    key = normalise_text(text=model)
    # Spacing variants, including a digit, are explicitly identified here.
    spacing_aliases = {
        ("Toyota", "rav 4"): "rav4",
        ("Hyundai", "ioniq5"): "ioniq 5",
        ("Audi", "q4 etron"): "q4 e tron",
    }
    key = spacing_aliases.get((make, key), key)
    return ModelIdentity(
        make=make,
        model=model,
        model_id=f"{normalise_text(text=make)}:{key}",
        grouped=grouped,
    )

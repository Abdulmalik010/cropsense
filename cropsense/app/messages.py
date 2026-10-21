"""Static user-facing strings. Safety paths do not depend on the LLM."""

from __future__ import annotations

from app.config import DISCLAIMER_EN, SUPPORTED_LANGUAGES

DISCLAIMERS = {
    "en": DISCLAIMER_EN,
    "ha": (
        "Wannan shawara ce kawai, ba ta maye gurbin masanin noma ko jami'in "
        "kiwon gona ba."
    ),
    "yo": (
        "Ìmọ̀ràn ìtọ́sọ́nà ni èyí, kì í ṣe dípò amòye iṣẹ́-àgbẹ̀ tàbí òṣìṣẹ́ "
        "ìtẹ̀síwájú iṣẹ́-àgbẹ̀."
    ),
    "ig": (
        "Nke a bụ nkwado mkpebi, ọ bụghị iji dochie ọkà mmụta ọrụ ugbo ma ọ bụ "
        "onye ọrụ ndụmọdụ ọrụ ugbo."
    ),
    "pcm": (
        "This na advice to help you decide. E no replace trained farm officer "
        "or agronomist."
    ),
}

LOW_CONFIDENCE = {
    "en": (
        "We could not identify this with enough confidence. Please take another "
        "photo in good light, of a single leaf, close up. If the plant is still "
        "unwell, contact a local agricultural extension officer."
    ),
    "ha": (
        "Ba mu iya gano cutar da cikakken tabbaci ba. Da fatar ka ɗauki wani "
        "hoton ganye guda a kusa, a haske mai kyau. Idan shuka har yanzu ba ta "
        "lafiya ba, tuntuɓi jami'in kiwon gona na yankin ka."
    ),
    "yo": (
        "A kò ní ìdánilójú tó peye láti sọ àìsàn náà. Jọ̀wọ́ ya fọ́tò mìíràn "
        "ní ìmọ́lẹ̀ dáradára, ewé kan ṣoṣo, ní tòsí. Tí ọ̀gbìn náà kò bá sí ní "
        "ìlera, pè òṣìṣẹ́ ìtẹ̀síwájú iṣẹ́-àgbẹ̀ ní àgbègbè rẹ."
    ),
    "ig": (
        "Anyị enweghị ntụkwasị obi zuru oke ịmata ọrịa a. Biko see foto ọzọ n’ìhè "
        "dị mma, otu akwụkwọ n’ime, nso nso. Ọ bụrụ na ihe ọkụkụ ka na-arịa, "
        "kpọtụrụ onye ọrụ ndụmọdụ ọrụ ugbo n’ogbe gị."
    ),
    "pcm": (
        "We no sure enough wetin dey worry the leaf. Abeg snap another photo "
        "for good light, one leaf only, close up. If the plant still sick, talk "
        "to local agricultural extension officer."
    ),
}

ADVISOR_OFFLINE = {
    "en": (
        "The disease guess is shown, but treatment advice is unavailable because "
        "the language model is offline. Ask a local extension officer before "
        "using any chemical."
    ),
    "ha": (
        "Mun nuna hasashen cutar, amma ba mu iya ba da shawarar magani ba saboda "
        "na'urar harshe ba ta aiki. Tambayi jami'in kiwon gona kafin ka yi amfani "
        "da wani magani."
    ),
    "yo": (
        "A fi àbá àìsàn hàn, ṣùgbọ́n ìmọ̀ràn ìtọ́jú kò sí nítorí ẹ̀rọ èdè kò sí "
        "nílíìnì. Béèrè lọ́wọ́ òṣìṣẹ́ ìtẹ̀síwájú kí o tó lo oògùn kankan."
    ),
    "ig": (
        "Anyị gosiri ihe anyị chere na ọ bụ ọrịa, mana ndụmọdụ ọgwụgwọ adịghị "
        "naka ọrụ asụsụ adịghị ọrụ. Jụọ onye ndụmọdụ ọrụ ugbo tupu i jiri ọgwụ ọ bụla."
    ),
    "pcm": (
        "We fit show the likely disease, but treatment advice no dey because the "
        "language model no dey on. Ask local extension officer before you use any "
        "chemical."
    ),
}

NO_KNOWLEDGE = {
    "en": (
        "No knowledge-base notes were found for this disease. Do not guess a "
        "pesticide. Ask a local agricultural extension officer."
    ),
    "ha": (
        "Babu bayanai a littafin ilimi game da wannan cuta. Kada ka yi hasashe "
        "magani. Tambayi jami'in kiwon gona."
    ),
    "yo": (
        "Kò sí àkọsílẹ̀ nípa àìsàn yìí nínú ìwé ìmọ̀. Má ṣe ṣàbá oògùn. Béèrè "
        "lọ́wọ́ òṣìṣẹ́ ìtẹ̀síwájú iṣẹ́-àgbẹ̀."
    ),
    "ig": (
        "Enweghị ihe odide n’akwụkwọ ihe ọmụma maka ọrịa a. Echela ọgwụ. Jụọ "
        "onye ndụmọdụ ọrụ ugbo."
    ),
    "pcm": (
        "No note dey for this disease for our knowledge base. No guess any "
        "pesticide. Ask local extension officer."
    ),
}

BAD_IMAGE = {
    "en": "Please upload a clear photo of a leaf (JPEG, PNG, or WebP).",
    "ha": "Da fatar ka loda hoton ganye mai kyau (JPEG, PNG, ko WebP).",
    "yo": "Jọ̀wọ́ gbé fọ́tò ewé tó yé (JPEG, PNG, tàbí WebP).",
    "ig": "Biko bulite foto akwụkwọ nri doro anya (JPEG, PNG, ma ọ bụ WebP).",
    "pcm": "Abeg upload clear photo of leaf (JPEG, PNG, or WebP).",
}

MISSING_MODEL = {
    "en": (
        "The disease classifier is not trained yet (missing models/classifier.pt). "
        "Run scripts/train.py after you have images in data/processed."
    ),
    "ha": "Na'urar gane cuta ba a horar da ita ba tukuna.",
    "yo": "Ẹ̀rọ ìdámọ̀ àìsàn kò tíì kẹ́kọ̀ọ́.",
    "ig": "Igwe nchọpụta ọrịa a zụbeghị.",
    "pcm": "The disease classifier never train yet.",
}


def normalize_language(code: str | None) -> str:
    """Return a supported language code, defaulting to English."""
    if not code:
        return "en"
    key = code.strip().lower()
    return key if key in SUPPORTED_LANGUAGES else "en"


def text(table: dict[str, str], language: str) -> str:
    """Pick a localized string, falling back to English."""
    return table.get(language, table["en"])


def disclaimer(language: str) -> str:
    """Return the safety disclaimer in the farmer's language."""
    return text(DISCLAIMERS, language)

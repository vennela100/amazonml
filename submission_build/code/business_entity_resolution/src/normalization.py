"""Conservative, deterministic text views and numeric evidence; no learned rules."""
from __future__ import annotations

import re
import unicodedata

VERSION = '1.0.0'
SPACE = re.compile(r'\s+')
DIGITS = re.compile(r'\d+')
DOTTED_INITIALS = re.compile(r'(?<!\w)(?:[a-z]\.){2,}')
NULL_LIKE = frozenset({'null', 'none', 'nan', 'na', 'n/a', '<na>'})
# Unicode marks (including Indic vowel signs) are deliberately retained.
PUNCTUATION = {i: ' ' for i in range(0x110000)
               if unicodedata.category(chr(i))[0] in ('P', 'S')}
DECIMALS = {i: str(unicodedata.decimal(chr(i))) for i in range(0x110000)
            if unicodedata.category(chr(i)) == 'Nd' and not 48 <= i <= 57}
DASHES = str.maketrans({c: '-' for c in '\u2010\u2011\u2012\u2013\u2014\u2212'})
LEGAL = {'ltd': 'limited', 'limited': 'limited', 'pvt': 'private', 'private': 'private',
         'corp': 'corporation', 'corporation': 'corporation', 'inc': 'incorporated',
         'incorporated': 'incorporated', 'co': 'company', 'company': 'company',
         'llc': 'llc', 'llp': 'llp', 'plc': 'plc', 'lp': 'lp'}

# Preserve composite municipal numbers, fractions, letter suffixes, and zero padding.
NUMBER_BODY = r'(?:[a-z]{1,3}[-/])?\d+[a-z]?(?:\s+\d+/\d+|\s*[-/]\s*\d+[a-z]?)?'
NUMBER = re.compile(r'(?<!\w)' + NUMBER_BODY + r'(?!\w)')
HOUSE = re.compile(r'\b(?:house|door|plot|building|bldg|h\s*\.?\s*no)\s*\.?\s*'
                   r'(?:(?:no\.?|number|#)\s*)?[:#-]?\s*(' + NUMBER_BODY + r')(?!\w)')
LEADING = re.compile(r'^\s*(?:(?:no\.?|number|#)\s*[:#-]?\s*)?(' + NUMBER_BODY + r')(?!\w)')
STREET = re.compile(r'\b(\d+)(?:st|nd|rd|th)\s+(?:street|st|avenue|ave|road|rd)\b'
                    r'|\b(?:street|road)\s+(?:no\.?|number)\s*(\d+)\b')
STREET_WORD = re.compile(r'\b(?:street|st|road|rd|avenue|ave|lane|ln|drive|dr|court|ct|'
                         r'boulevard|blvd|rue|allée|allee|chemin|route|marg|nagar)\b')
POSTAL_BODY = r'(?:\d{5}-\d{4}|\d{6}|\d{5})'
POSTAL = re.compile(r'(?<![\w/\-])(' + POSTAL_BODY + r')(?![\w/\-])')
POSTAL_LABEL = re.compile(r'\b(?:zip(?:\s*code)?|postal\s*code|post\s*code|postcode|'
                          r'pin(?:\s*code)?|pincode|code\s*postal)\s*[:#.-]?\s*'
                          r'(' + POSTAL_BODY + r')(?![\w/\-])')

RAW_COLUMNS = ['entity_id', 'business_name', 'business_address', 'country']
TEXT_COLUMNS = ['name_clean', 'name_normalized', 'address_normalized']
EVIDENCE_COLUMNS = ['address_numbers', 'house_numbers', 'house_number_candidates',
                    'house_number_source', 'street_numbers', 'postal_codes',
                    'postal_candidates', 'postal_code_source']
FLAG_COLUMNS = ['name_missing', 'address_missing', 'name_null_like', 'address_null_like',
                'name_normalized_empty', 'address_normalized_empty', 'house_number_missing',
                'house_number_ambiguous', 'postal_code_missing', 'postal_code_ambiguous']
OUTPUT_COLUMNS = RAW_COLUMNS + TEXT_COLUMNS + EVIDENCE_COLUMNS + FLAG_COLUMNS


def prepare(text):
    return SPACE.sub(' ', unicodedata.normalize('NFKC', text).casefold().translate(DECIMALS)).strip()


def clean_prepared(text):
    text = DOTTED_INITIALS.sub(lambda match: match.group().replace('.', ''), text) if '.' in text else text
    return SPACE.sub(' ', text.replace('&', ' and ').translate(PUNCTUATION)).strip()


def normalize_name(text):
    return normalize_prepared_name(prepare(text))


def normalize_prepared_name(text):
    clean = clean_prepared(text)
    tokens = clean.split()
    # Legal words within the distinctive name are not reinterpreted as suffixes.
    position = len(tokens) - 1
    while position >= 0 and tokens[position] in LEGAL:
        tokens[position] = LEGAL[tokens[position]]
        position -= 1
    return clean, ' '.join(tokens)


def canonical_number(text):
    return re.sub(r'\s*([/-])\s*', r'\1', text.strip())


def numeric_evidence(prepared_address):
    text = prepared_address.translate(DASHES)
    numbers = sorted({canonical_number(match.group()) for match in NUMBER.finditer(text)})
    house_matches = list(HOUSE.finditer(text))
    houses = sorted({canonical_number(match.group(1)) for match in house_matches})
    house_candidates = set(houses)
    house_source = 'explicit_premise_label' if houses else ''
    leading = LEADING.match(text)
    if leading:
        candidate = canonical_number(leading.group(1))
        house_candidates.add(candidate)
        rest = text[leading.end():].strip(' ,.:;-')
        # Bare numeric strings and leading 5/6-digit city codes stay ambiguous.
        if not houses and rest and re.search(r'[^\W\d_]', rest):
            simple_long_number = bool(re.fullmatch(POSTAL_BODY, candidate))
            if not simple_long_number or STREET_WORD.search(rest):
                houses = [candidate]
                house_source = 'leading_number_heuristic'
    house_spans = {match.span(1) for match in house_matches}
    if leading and house_source == 'leading_number_heuristic':
        house_spans.add(leading.span(1))
    postal_matches = [match for match in POSTAL.finditer(text) if match.span(1) not in house_spans]
    postal_candidates = {match.group(1) for match in postal_matches}
    explicit = {match.group(1) for match in POSTAL_LABEL.finditer(text)}
    postal = sorted(explicit)
    postal_source = 'explicit_postal_label' if postal else ''
    if not postal:
        terminal = postal_matches
        if terminal:
            last = terminal[-1]
            before, after = text[:last.start()].strip(), text[last.end():].strip(' ,.;')
            if before and not after and not (leading and leading.start(1) == last.start(1)):
                postal = [last.group(1)]
                postal_source = 'trailing_numeric_heuristic'
    # Do not treat a leading postcode marked explicitly as a house number.
    if house_source == 'leading_number_heuristic' and set(houses) & explicit:
        houses, house_source = [], ''
    streets = sorted({match.group(1) or match.group(2) for match in STREET.finditer(text)})
    return {
        'address_numbers': numbers, 'house_numbers': houses,
        'house_number_candidates': sorted(house_candidates), 'house_number_source': house_source,
        'street_numbers': streets, 'postal_codes': postal,
        'postal_candidates': sorted(postal_candidates), 'postal_code_source': postal_source,
        'house_number_missing': int(not houses),
        'house_number_ambiguous': int(len(houses) > 1 or (not houses and bool(house_candidates))),
        'postal_code_missing': int(not postal),
        'postal_code_ambiguous': int(len(postal_candidates) > 1 or (not postal and bool(postal_candidates))),
    }


def normalize_record(row):
    entity_id, raw_name, raw_address, country = row
    name_prepared, address_prepared = prepare(raw_name), prepare(raw_address)
    name_clean, name_normalized = normalize_prepared_name(name_prepared)
    address_normalized = clean_prepared(address_prepared)
    # Every decimal digit survives in the primary normalized views, in order.
    if DIGITS.findall(name_prepared) != DIGITS.findall(name_normalized):
        raise AssertionError(f'Name digit loss: {entity_id}')
    if DIGITS.findall(address_prepared) != DIGITS.findall(address_normalized):
        raise AssertionError(f'Address digit loss: {entity_id}')
    evidence = numeric_evidence(address_prepared)
    result = dict(zip(RAW_COLUMNS, row))
    result.update(name_clean=name_clean, name_normalized=name_normalized,
                  address_normalized=address_normalized,
                  name_missing=int(not raw_name.strip()), address_missing=int(not raw_address.strip()),
                  name_null_like=int(raw_name.strip().casefold() in NULL_LIKE),
                  address_null_like=int(raw_address.strip().casefold() in NULL_LIKE),
                  name_normalized_empty=int(not name_normalized),
                  address_normalized_empty=int(not address_normalized), **evidence)
    return result

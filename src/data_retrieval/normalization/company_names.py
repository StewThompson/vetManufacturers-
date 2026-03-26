import re
import unicodedata
from typing import Dict

# NFKD / Unicode pre-clean
_UNICODE_QUOTES = re.compile(r'[\u2018\u2019\u201a\u201b\u2032\u2035]')
_UNICODE_DASHES = re.compile(r'[\u2010-\u2015\u2212]')
_CONTROL_CHARS = re.compile(r'[\x00-\x1f\x7f]')
_UNICODE_AMP = re.compile(r'\uff06')

_BRANCH_PREFIX = re.compile(r'^[A-Za-z0-9]{5,}\s*[-–]\s+(?=[A-Za-z])')

_SLASH_SEP = re.compile(r'\s*/\s*')
_AMP_TO_AND = re.compile(r'\s*&\s*')
_AND_SPACES = re.compile(r'\b AND \b', re.I)
_PUNCT_CLEAN = re.compile(r"[''`]")
_TRAILING_PUNCT = re.compile(r'[.,;:\-–/]+$')
_WS = re.compile(r'\s{2,}')

_DBA_CLAUSE = re.compile(r'\s+(?:D/?B/?A|F/?K/?A|A/?K/?A)\s+.+$', re.I)

_LEADING_THE = re.compile(r'^THE\s+', re.I)
_TRAILING_THE = re.compile(r'[,.]?\s*\bTHE\s*$', re.I)

_CORP_SUFFIX = re.compile(
    r'[,.]?\s*\b(?:'
    r'INC\.?|INCORPORATED|'
    r'LLC\.?|L\.?L\.?C\.?|'
    r'LLP\.?|L\.?L\.?P\.?|'
    r'LP\.?|L\.?P\.?|'
    r'CORP\.?|CORPORATION|'
    r'CO\.?|COMPANY|'
    r'LTD\.?|LIMITED(?:\s+(?:PARTNERSHIP|LIABILITY\s+COMPANY))?|'
    r'HOLDINGS?|GROUP|ENTERPRISES?|SOLUTIONS?|INDUSTRIES?|INTERNATIONAL'
    r')\s*$',
    re.I,
)

_FACILITY_NOISE = re.compile(
    r'\s*[-\u2013,]?\s*\b(?:'
    r'PLANT|BLDG|BUILDING|UNIT|LOC|LOCATION|'
    r'STORE|'
    r'WAREHOUSE|WH|'
    r'DC|PDC|FC|IDC|'
    r'YARD|ANNEX|CAMPUS|COMPLEX|SITE|'
    r'DIVISION|DIV'
    r')\s*(?:#?\s*\d+)?\s*$',
    re.I,
)

_TRAILING_HASH_NUM = re.compile(r'\s*#\s*\d+\s*$')
_TRAILING_NUM = re.compile(r'\s+\d{4,}\s*$')
_DANGLING_SEP = re.compile(r'\s*[-\u2013,/]\s*$')

_TOKEN_CANON: Dict[str, str] = {
    'INTL':    'INTERNATIONAL',
    'INTNL':   'INTERNATIONAL',
    'MFG':     'MANUFACTURING',
    'MFGR':    'MANUFACTURING',
    'MFR':     'MANUFACTURING',
    'IND':     'INDUSTRIAL',
    'INDS':    'INDUSTRIES',
    'INDUS':   'INDUSTRIES',
    'SVCS':    'SERVICES',
    'SVC':     'SERVICES',
    'TECH':    'TECHNOLOGY',
    'TECHS':   'TECHNOLOGIES',
    'DIST':    'DISTRIBUTION',
    'DISTR':   'DISTRIBUTION',
    'DISTRIB': 'DISTRIBUTION',
    'DISTRO':  'DISTRIBUTION',
    'PKG':     'PACKAGING',
    'PRODS':   'PRODUCTS',
    'PROD':    'PRODUCTS',
    'EQUIP':   'EQUIPMENT',
    'CORP':    '',
    'ASSOC':   'ASSOCIATES',
    'ASSOCS':  'ASSOCIATES',
    'GRP':     'GROUP',
    'MGMT':    'MANAGEMENT',
    'MGNT':    'MANAGEMENT',
    'NATL':    'NATIONAL',
    'AMER':    'AMERICAN',
    'PWR':     'POWER',
    'SYS':     'SYSTEMS',
    'ENGRG':   'ENGINEERING',
    'ENGR':    'ENGINEERING',
    'ENG':     'ENGINEERING',
    'CHEM':    'CHEMICAL',
    'CHEMS':   'CHEMICALS',
    'ELEC':    'ELECTRIC',
    'ELECS':   'ELECTRIC',
    'AUTH':    'AUTHORITY',
}


def preclean(raw: str) -> str:
    s = unicodedata.normalize('NFKD', raw)
    s = ''.join(c for c in s if not unicodedata.combining(c))
    s = _UNICODE_QUOTES.sub("'", s)
    s = _UNICODE_DASHES.sub('-', s)
    s = _UNICODE_AMP.sub('&', s)
    s = _CONTROL_CHARS.sub(' ', s)
    s = s.upper()
    s = _BRANCH_PREFIX.sub('', s).strip()
    s = _SLASH_SEP.sub(' ', s)
    s = _PUNCT_CLEAN.sub('', s)
    s = _AMP_TO_AND.sub(' AND ', s)
    s = _WS.sub(' ', s).strip()
    return s


def strip_noise(s: str, strip_facility: bool = False) -> str:
    s = _DBA_CLAUSE.sub('', s).strip()
    always_patterns = (_TRAILING_HASH_NUM, _TRAILING_NUM, _DANGLING_SEP)
    facility_patterns = (_FACILITY_NOISE,) if strip_facility else ()

    changed = True
    while changed:
        changed = False
        for pat in always_patterns + facility_patterns:
            new = pat.sub('', s).strip()
            if new != s:
                s = new
                changed = True

    s = _TRAILING_THE.sub('', s).strip()

    for _ in range(5):
        new = _CORP_SUFFIX.sub('', s).strip()
        if new == s:
            break
        s = new

    s = _TRAILING_THE.sub('', s).strip()
    s = _LEADING_THE.sub('', s).strip()
    s = _TRAILING_PUNCT.sub('', s).strip()
    s = _WS.sub(' ', s).strip()
    return s


def canonicalize_tokens(s: str) -> str:
    tokens = s.split()
    result = []
    for tok in tokens:
        expanded = _TOKEN_CANON.get(tok)
        if expanded is None:
            result.append(tok)
        elif expanded:
            result.append(expanded)
    return ' '.join(result)


def normalize_company_name(raw: str) -> str:
    s = preclean(raw)
    s = strip_noise(s, strip_facility=False)
    return s


def company_match_key(raw: str) -> str:
    s = preclean(raw)
    s = strip_noise(s, strip_facility=True)
    s = canonicalize_tokens(s)
    s = _WS.sub(' ', s).strip()
    return s

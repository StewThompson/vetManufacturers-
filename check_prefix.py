import sys, re
sys.path.insert(0, '.')
from src.search.grouped_search import normalize_establishment_name

def parent_prefix(name):
    norm = normalize_establishment_name(name).upper()
    norm = re.sub(r'\s+[-]?\s*[A-Z]{1,5}[0-9]{1,4}\s*$', '', norm).strip()
    norm = re.sub(r'\s*\([A-Z0-9]{2,8}\)\s*$', '', norm).strip()
    norm = re.sub(r'\s+(?:SITE|CAMPUS|FACILITY|WAREHOUSE|DC|FC|IDC|PDC|UNIT|LOC)\s*\d*\s*$', '', norm, flags=re.IGNORECASE).strip()
    norm = re.sub(r'\s*[-/]\s*$', '', norm).strip()
    return norm or normalize_establishment_name(name).upper()

names = [
    'Walmart', 'Walmart, Inc', 'Walmart Warehouse',
    'Walmart Supercenter', 'Walmart Distribution Center',
    'Walmart Stores East', 'Walmart Neighborhood Market',
    'Walmart Stores', 'Walmart Puerto Rico', 'Walmart Associates',
    'Walmart Auto Care Center', 'Walmart Inc.',
]
for n in names:
    print(f'  "{n}" -> "{parent_prefix(n)}"')

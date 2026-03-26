import sys
sys.path.insert(0, '.')
from src.data_retrieval.osha_client import OSHAClient
from src.search.grouped_search import group_establishments, get_or_build_company_key_index

client = OSHAClient()
client.ensure_cache()

company_key_index = get_or_build_company_key_index(client)
_ckey_to_estabs, company_keys = company_key_index

names = client.get_all_company_names()
print(f"Total company names: {len(names)}")
walmart_names = [n for n in names if 'walmart' in n.lower()]
print(f"Walmart-related names in index: {len(walmart_names)}")
for n in sorted(walmart_names):
    print(f"  {n}")

print("\nRunning group_establishments('Walmart')...")
result = group_establishments("Walmart", company_key_index, client)
if result.top_group:
    g = result.top_group
    all_fac = g.all_facilities
    print(f"\nTop group: '{g.parent_name}'")
    print(f"  High confidence:   {len(g.high_confidence)}")
    print(f"  Medium confidence: {len(g.medium_confidence)}")
    print(f"  Low confidence:    {len(g.low_confidence)}")
    print(f"  Total facilities:  {len(all_fac)}")
    print("\n  Facility raw_names:")
    for f in all_fac:
        print(f"    [{f.confidence_label}] {f.raw_name}  ({f.city}, {f.state})")
print(f"\nOther groups: {len(result.other_groups)}")
for g in result.other_groups:
    print(f"  '{g.parent_name}' ({len(g.all_facilities)} facilities)")

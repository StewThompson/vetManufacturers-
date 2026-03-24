import sys
sys.path.insert(0, '.')
from src.data_retrieval.osha_client import OSHAClient as C

cases = [
    # (input, expected_canonical, expected_match_key)
    ("GOODYEAR TIRE & RUBBER",          "GOODYEAR TIRE AND RUBBER",  "GOODYEAR TIRE AND RUBBER"),
    ("GOODYEAR TIRE AND RUBBER COMPANY","GOODYEAR TIRE AND RUBBER",  "GOODYEAR TIRE AND RUBBER"),
    ("GOODYEAR TIRE & RUBBER CO THE",   "GOODYEAR TIRE AND RUBBER",  "GOODYEAR TIRE AND RUBBER"),
    ("Goodyear Tire & Rubber, The",     "GOODYEAR TIRE AND RUBBER",  "GOODYEAR TIRE AND RUBBER"),
    ("WALMART",                         "WALMART",                   "WALMART"),
    ("WALMART, INC",                    "WALMART",                   "WALMART"),
    ("WALMART, INC.",                   "WALMART",                   "WALMART"),
    ("Walmart, Inc.",                   "WALMART",                   "WALMART"),
    ("WALMART SUPERCENTER",             "WALMART SUPERCENTER",       "WALMART SUPERCENTER"),
    ("WALMART STORE #1593",             "WALMART STORE",             "WALMART"),   # match_key strips STORE #1593
    ("WALMART SUPERCENTER STORE #2345", "WALMART SUPERCENTER STORE", "WALMART SUPERCENTER"),
    ("WALMART DISTRIBUTION CENTER",     "WALMART DISTRIBUTION CENTER","WALMART DISTRIBUTION CENTER"),
    ("105891 - WALMART INC",            "WALMART",                   "WALMART"),
    ("WA317974334 - WALMART INC.",      "WALMART",                   "WALMART"),
    ("AMAZON DBA AMAZON FULFILLMENT",   "AMAZON",                    "AMAZON"),
    ("THE KROGER CO",                   "KROGER",                    "KROGER"),
    ("PARKER HANNIFIN CORP",            "PARKER HANNIFIN",           "PARKER HANNIFIN"),
    ("PARKER HANNIFIN MFG",             "PARKER HANNIFIN MFG",       "PARKER HANNIFIN MANUFACTURING"),
    ("FASTENAL INTL SVCS",              "FASTENAL INTL SVCS",        "FASTENAL INTERNATIONAL SERVICES"),
    ("BOEING CO",                       "BOEING",                    "BOEING"),
    ("3M COMPANY",                      "3M",                        "3M"),
    ("TARGET STORE #1189",              "TARGET STORE",              "TARGET"),
    ("TARGET STORE T1063 PLANT 2",      "TARGET STORE T1063 PLANT 2","TARGET STORE T1063"),
]

ok = True
for raw, exp_canon, exp_key in cases:
    canon = C._normalize_company_name(raw)
    key   = C.company_match_key(raw)
    c_ok = canon == exp_canon
    k_ok = key == exp_key
    if not c_ok or not k_ok:
        ok = False
    status = "OK" if (c_ok and k_ok) else "FAIL"
    print(f"[{status}] {raw!r}")
    if not c_ok:
        print(f"       canonical: got {canon!r}, want {exp_canon!r}")
    if not k_ok:
        print(f"       match_key: got {key!r}, want {exp_key!r}")

print("\nAll OK" if ok else "\nSome failures above")

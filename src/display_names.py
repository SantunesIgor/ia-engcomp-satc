from typing import Iterable, List, Tuple

AIRLINE_NAMES = {
    "AS": "Alaska Airlines",
    "AA": "American Airlines",
    "AC": "Air Canada",
    "AM": "Aeromexico",
    "CO": "Continental Airlines",
    "DL": "Delta Airlines",
    "FX": "FedEx",
    "HA": "Hawaiian Airlines",
    "NW": "Northwest Airlines",
    "PO": "Polar Air Cargo",
    "SW": "Southwest Airlines",
    "WN": "Southwest Airlines",
    "UA": "United Airlines",
    "5X": "United Parcel Service",
    "VS": "Virgin Atlantic",
    "VB": "VivaAerobus",
    "WS": "WestJet",
}

AIRPORT_NAMES = {
    "ATL": "Atlanta, Georgia - Hartsfield-Jackson Atlanta International Airport",
    "AUS": "Austin, Texas - Austin-Bergstrom International Airport",
    "BNA": "Nashville, Tennessee - Nashville International Airport",
    "BOS": "Boston, Massachusetts - Boston Logan International Airport",
    "BWI": "Washington - Baltimore-Washington International Thurgood Marshall Airport",
    "CLT": "Charlotte, North Carolina - Charlotte Douglas International Airport",
    "DAL": "Dallas, Texas - Dallas Love Field",
    "DCA": "Arlington, Virginia - Ronald Reagan Washington National Airport",
    "DEN": "Denver, Colorado - Denver International Airport",
    "DFW": "Dallas/Fort Worth, Texas - Dallas/Fort Worth International Airport",
    "DTW": "Detroit, Michigan - Detroit Metropolitan Airport",
    "EWR": "New Jersey - Newark Liberty International Airport",
    "FLL": "Florida - Fort Lauderdale-Hollywood International Airport",
    "HNL": "Honolulu, Hawaii - Daniel K. Inouye International Airport",
    "HOU": "Houston, Texas - William P. Hobby Airport",
    "IAD": "Virginia - Dulles International Airport",
    "IAH": "Houston, Texas - George Bush Intercontinental Airport",
    "JFK": "Queens, New York - John F. Kennedy International Airport",
    "LAS": "Las Vegas, Nevada - McCarran International Airport",
    "LAX": "Los Angeles, California - Los Angeles International Airport",
    "LGA": "Queens, New York - LaGuardia Airport",
    "MCO": "Orlando, Florida - Orlando International Airport",
    "MDW": "Chicago, Illinois - Chicago Midway International Airport",
    "MIA": "Miami, Florida - Miami International Airport",
    "MSP": "Minneapolis-Saint Paul, Minnesota - Minneapolis-Saint Paul International Airport",
    "MSY": "New Orleans, Louisiana - Louis Armstrong New Orleans International Airport",
    "OAK": "Oakland, California - Oakland International Airport",
    "ORD": "Chicago, Illinois - O'Hare International Airport",
    "PDX": "Portland, Oregon - Portland International Airport",
    "PHL": "Philadelphia, Pennsylvania - Philadelphia International Airport",
    "PHX": "Phoenix, Arizona - Phoenix Sky Harbor International Airport",
    "RDU": "Raleigh-Durham, North Carolina - Raleigh-Durham International Airport",
    "SAN": "San Diego, California - San Diego International Airport",
    "SEA": "Washington - Seattle-Tacoma International Airport",
    "SFO": "San Francisco, California - San Francisco International Airport",
    "SJC": "San Jose, California - Norman Y. Mineta San Jose International Airport",
    "SLC": "Salt Lake City, Utah - Salt Lake City International Airport",
    "SMF": "Sacramento, California - Sacramento International Airport",
    "STL": "St. Louis, Missouri - St. Louis Lambert International Airport",
    "TPA": "Tampa, Florida - Tampa International Airport",
}

ALLOWED_AIRLINE_CODES = set(AIRLINE_NAMES.keys())
ALLOWED_AIRPORT_CODES = set(AIRPORT_NAMES.keys())


def normalize_code(code):
    return str(code).strip().upper()


def airline_name(code):
    code = normalize_code(code)
    return AIRLINE_NAMES.get(code, f"Companhia aerea nao identificada: {code}")


def airport_name(code):
    code = normalize_code(code)
    return AIRPORT_NAMES.get(code, f"Aeroporto nao identificado: {code}")


def airline_label(code):
    code = normalize_code(code)
    return f"{airline_name(code)} ({code})"


def airport_label(code):
    code = normalize_code(code)
    return f"{airport_name(code)} ({code})"


def build_options(codes, label_function):
    unique_codes = sorted({normalize_code(code) for code in codes if str(code).strip()})
    labels = sorted([label_function(code) for code in unique_codes])
    label_to_code = {label_function(code): code for code in unique_codes}
    return labels, label_to_code

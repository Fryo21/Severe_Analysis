from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "processed-data"
    / "market-entry"
    / "service-categories"
    / "service_categories_combined_clean.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "processed-data"
    / "market-entry"
    / "service-categories"
    / "taxonomy"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TAXONOMY_OUTPUT = OUTPUT_DIR / "service_taxonomy_final.csv"
COVERAGE_OUTPUT = OUTPUT_DIR / "service_taxonomy_final_coverage.csv"
UNMAPPED_OUTPUT = OUTPUT_DIR / "service_taxonomy_excluded.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "service_taxonomy_final_summary.csv"


# ============================================================
# HELPERS
# ============================================================

def normalise_space(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalise_for_match(value: str) -> str:
    text = normalise_space(value).casefold()
    text = text.replace("&", "and")
    text = re.sub(r"[/_-]+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def match(pattern: str, text: str) -> bool:
    return bool(re.search(pattern, text, flags=re.IGNORECASE))



# ============================================================
# SECOND-PASS EXPANSION
# Added from recurring patterns in service_family_unmapped.csv
# ============================================================

NOISE_PATTERNS = [
    r"^\+?\d[\d\s\-()]{7,}$",      # phone numbers
    r"^(aberdeen)$",               # known location contamination
    r"^(discover|getting started|sitemap)$",
]

EXPANDED_RULES = [

    # --------------------------------------------------------
    # VEHICLE SERVICES
    # --------------------------------------------------------
    (r"\baccident repair\b", "Accident Repair", "Vehicle Services"),
    (r"\bcar body repair|bodywork repair|body repair\b", "Car Body Repair", "Vehicle Services"),
    (r"\balloy wheel repair|alloy wheel refurbishment\b", "Alloy Wheel Repair", "Vehicle Services"),
    (r"\btyre repair|tire repair|puncture repair\b", "Tyre Repair", "Vehicle Services"),
    (r"\btyre fitting|tire fitting\b", "Tyre Fitting", "Vehicle Services"),
    (r"\bmobile mechanic\b", "Mobile Mechanic", "Vehicle Services"),
    (r"\bcar repair|vehicle repair|auto repair|motor repair\b", "General Vehicle Repair", "Vehicle Services"),
    (r"\bcar service|vehicle service|car servicing\b", "Vehicle Servicing", "Vehicle Services"),
    (r"\bcar valeting|car valet\b", "Car Valeting", "Vehicle Services"),
    (r"\bcar detailing|vehicle detailing\b", "Car Detailing", "Vehicle Services"),

    # --------------------------------------------------------
    # DELIVERY / COURIER SERVICES
    # --------------------------------------------------------
    (r"\bfood delivery\b", "Food Delivery", "Delivery Services"),
    (r"\bgrocery delivery\b", "Grocery Delivery", "Delivery Services"),
    (r"\bgift delivery\b", "Gift Delivery", "Delivery Services"),
    (r"\bcake delivery\b", "Cake Delivery", "Delivery Services"),
    (r"\bflower delivery\b", "Flower Delivery", "Delivery Services"),
    (r"\bfurniture delivery\b", "Furniture Delivery", "Delivery Services"),
    (r"\bcourier\b", "Courier Services", "Delivery Services"),
    (r"\bdelivery\b", "General Delivery", "Delivery Services"),

    # --------------------------------------------------------
    # RENEWABLE ENERGY SERVICES
    # --------------------------------------------------------
    (r"\bair source heat pump.*install|install.*air source heat pump\b", "Air Source Heat Pump Installation", "Renewable Energy Services"),
    (r"\bair source heat pump.*repair|repair.*air source heat pump\b", "Air Source Heat Pump Repair", "Renewable Energy Services"),
    (r"\bground source heat pump.*install|install.*ground source heat pump\b", "Ground Source Heat Pump Installation", "Renewable Energy Services"),
    (r"\bground source heat pump.*repair|repair.*ground source heat pump\b", "Ground Source Heat Pump Repair", "Renewable Energy Services"),
    (r"\bheat pump\b", "Heat Pump Services", "Renewable Energy Services"),
    (r"\bsolar panel.*install|install.*solar panel\b", "Solar Panel Installation", "Renewable Energy Services"),
    (r"\bsolar panel.*repair|repair.*solar panel\b", "Solar Panel Repair", "Renewable Energy Services"),
    (r"\bsolar panel|solar pv|photovoltaic\b", "Solar Energy Services", "Renewable Energy Services"),

    # --------------------------------------------------------
    # INSULATION SERVICES
    # --------------------------------------------------------
    (r"\bloft insulation\b", "Loft Insulation", "Insulation Services"),
    (r"\bcavity wall insulation\b", "Cavity Wall Insulation", "Insulation Services"),
    (r"\bwall insulation\b", "Wall Insulation", "Insulation Services"),
    (r"\broof insulation\b", "Roof Insulation", "Insulation Services"),
    (r"\binsulation\b", "General Insulation", "Insulation Services"),

    # --------------------------------------------------------
    # CONSERVATORY SERVICES
    # --------------------------------------------------------
    (r"\bconservatory roof.*repair|repair.*conservatory roof\b", "Conservatory Roof Repair", "Conservatory Services"),
    (r"\bconservatory roof\b", "Conservatory Roofing", "Conservatory Services"),
    (r"\bconservatory.*repair|repair.*conservatory\b", "Conservatory Repair", "Conservatory Services"),
    (r"\bconservatory.*install|install.*conservatory\b", "Conservatory Installation", "Conservatory Services"),
    (r"\bconservatory\b", "General Conservatory Services", "Conservatory Services"),

    # --------------------------------------------------------
    # CHIMNEY SERVICES
    # --------------------------------------------------------
    (r"\bchimney sweep\b", "Chimney Sweeping", "Chimney Services"),
    (r"\bchimney repair\b", "Chimney Repair", "Chimney Services"),
    (r"\bchimney.*install|install.*chimney\b", "Chimney Installation", "Chimney Services"),
    (r"\bchimney\b", "General Chimney Services", "Chimney Services"),

    # --------------------------------------------------------
    # AIR CONDITIONING & VENTILATION
    # --------------------------------------------------------
    (r"\bair conditioning.*install|air conditioner.*install\b", "Air Conditioning Installation", "Air Conditioning & Ventilation"),
    (r"\bair conditioning.*repair|air conditioner.*repair\b", "Air Conditioning Repair", "Air Conditioning & Ventilation"),
    (r"\bair conditioning.*service|air conditioner.*service\b", "Air Conditioning Servicing", "Air Conditioning & Ventilation"),
    (r"\bair conditioning|air conditioner\b", "General Air Conditioning", "Air Conditioning & Ventilation"),
    (r"\bventilation\b", "Ventilation Services", "Air Conditioning & Ventilation"),

    # --------------------------------------------------------
    # PEST CONTROL
    # --------------------------------------------------------
    (r"\bbed bug\b", "Bed Bug Treatment", "Pest Control Services"),
    (r"\brat control|mice control|rodent control\b", "Rodent Control", "Pest Control Services"),
    (r"\bwasp nest|wasp control\b", "Wasp Control", "Pest Control Services"),
    (r"\bpest control|pest removal|exterminator\b", "General Pest Control", "Pest Control Services"),

    # --------------------------------------------------------
    # FIRE & SAFETY
    # --------------------------------------------------------
    (r"\bfire alarm.*install|install.*fire alarm\b", "Fire Alarm Installation", "Fire & Safety Systems"),
    (r"\bfire alarm.*service|fire alarm.*maintenance\b", "Fire Alarm Servicing", "Fire & Safety Systems"),
    (r"\bfire extinguisher\b", "Fire Extinguisher Services", "Fire & Safety Systems"),
    (r"\bfire safety|fire alarm\b", "General Fire Safety", "Fire & Safety Systems"),

    # --------------------------------------------------------
    # BLINDS & SHUTTERS
    # --------------------------------------------------------
    (r"\bblind.*install|install.*blind\b", "Blind Installation", "Blinds & Shutters"),
    (r"\bblind.*repair|repair.*blind\b", "Blind Repair", "Blinds & Shutters"),
    (r"\bshutter.*install|install.*shutter\b", "Shutter Installation", "Blinds & Shutters"),
    (r"\bshutter.*repair|repair.*shutter\b", "Shutter Repair", "Blinds & Shutters"),
    (r"\bblind|shutter\b", "General Blinds & Shutters", "Blinds & Shutters"),

    # --------------------------------------------------------
    # POOLS & HOT TUBS
    # --------------------------------------------------------
    (r"\bswimming pool.*install|install.*swimming pool\b", "Swimming Pool Installation", "Pool & Hot Tub Services"),
    (r"\bswimming pool.*repair|repair.*swimming pool\b", "Swimming Pool Repair", "Pool & Hot Tub Services"),
    (r"\bswimming pool.*clean|pool cleaning\b", "Swimming Pool Cleaning", "Pool & Hot Tub Services"),
    (r"\bswimming pool|pool maintenance\b", "Swimming Pool Services", "Pool & Hot Tub Services"),
    (r"\bhot tub.*install|install.*hot tub\b", "Hot Tub Installation", "Pool & Hot Tub Services"),
    (r"\bhot tub.*repair|repair.*hot tub\b", "Hot Tub Repair", "Pool & Hot Tub Services"),
    (r"\bhot tub\b", "Hot Tub Services", "Pool & Hot Tub Services"),

    # --------------------------------------------------------
    # CEILINGS / DRY LINING
    # --------------------------------------------------------
    (r"\bsuspended ceiling\b", "Suspended Ceilings", "Ceiling & Interior Finishes"),
    (r"\bceiling repair\b", "Ceiling Repair", "Ceiling & Interior Finishes"),
    (r"\bdry lining|drylining\b", "Dry Lining", "Ceiling & Interior Finishes"),
    (r"\bceiling\b", "General Ceiling Services", "Ceiling & Interior Finishes"),

    # --------------------------------------------------------
    # SMART HOME / AUTOMATION
    # --------------------------------------------------------
    (r"\bhome automation|smart home\b", "Home Automation", "Smart Home & Automation"),
    (r"\bhome cinema\b", "Home Cinema Installation", "Smart Home & Automation"),

    # --------------------------------------------------------
    # ACCESSIBILITY / MOBILITY
    # --------------------------------------------------------
    (r"\baccess ramp|wheelchair ramp\b", "Access Ramp Installation", "Accessibility & Mobility Services"),
    (r"\baccessible kitchen\b", "Accessible Kitchen Installation", "Accessibility & Mobility Services"),
    (r"\baccessible bathroom|disabled bathroom\b", "Accessible Bathroom Installation", "Accessibility & Mobility Services"),
    (r"\bstairlift\b", "Stairlift Services", "Accessibility & Mobility Services"),

    # --------------------------------------------------------
    # RESTORATION / SPECIALIST BUILDING
    # --------------------------------------------------------
    (r"\bbuilding restoration\b", "Building Restoration", "Building & Construction"),
    (r"\bproperty restoration\b", "Property Restoration", "Building & Construction"),

    # --------------------------------------------------------
    # EXPAND EXISTING DOOR / WINDOW / FENCING / ROOF / GARDEN
    # --------------------------------------------------------
    (r"\bgarage doors?.*install|install.*garage doors?\b", "Garage Door Installation", "Doors & Door Services"),
    (r"\bgarage doors?.*repair|repair.*garage doors?\b", "Garage Door Repair", "Doors & Door Services"),
    (r"\bdoor replacement\b", "Door Replacement", "Doors & Door Services"),
    (r"\bdoors?.*install|install.*doors?\b", "Door Installation", "Doors & Door Services"),
    (r"\bdoors?.*repair|repair.*doors?\b", "Door Repair", "Doors & Door Services"),
    (r"\bwindows?.*install|install.*windows?\b", "Window Installation", "Windows & Glazing"),
    (r"\bwindows?.*repair|repair.*windows?\b", "Window Repair", "Windows & Glazing"),
    (r"\bwindow frame.*repair\b", "Window Frame Repair", "Windows & Glazing"),
    (r"\bfencing.*install|fence.*install\b", "Fence Installation", "Fencing & Gates"),
    (r"\bfencing.*repair|fence.*repair\b", "Fence Repair", "Fencing & Gates"),
    (r"\bagricultural fencing\b", "Agricultural Fencing", "Fencing & Gates"),
    (r"\bfencing\b", "General Fencing", "Fencing & Gates"),
    (r"\broof repairs?\b", "Roof Repair", "Roofing & Guttering"),
    (r"\broof installations?\b", "Roof Installation", "Roofing & Guttering"),
    (r"\bgarden clearance\b", "Garden Clearance", "Gardening & Landscaping"),
    (r"\bgarden services?\b", "General Gardening", "Gardening & Landscaping"),

    # --------------------------------------------------------
    # EXPAND WASTE / CLEARANCE
    # --------------------------------------------------------
    (r"\bhouse clearance\b", "House Clearance", "Waste Services"),
    (r"\boffice clearance\b", "Office Clearance", "Waste Services"),
    (r"\bwaste collection\b", "Waste Collection", "Waste Services"),
    (r"\bwaste removal\b", "Waste Removal", "Waste Services"),
    (r"\bclearance\b", "General Clearance", "Waste Services"),
]


FAMILY_RENAMES = {
    "Cleaning": "Cleaning Services",
    "Plumbing": "Plumbing Services",
    "Roofing & Guttering": "Roofing Services",
    "Locksmith": "Locksmith Services",
    "Massage": "Massage Services",
}


# ============================================================
# DETAILED TAXONOMY RULES
# IMPORTANT:
# - Put specific services BEFORE broad services.
# - service_family is the broad market category.
# - sub_service is the specific job/service underneath it.
# ============================================================

RULES = [

    # --------------------------------------------------------
    # PLUMBING
    # --------------------------------------------------------
    (r"\bemergency plumber|emergency plumbing\b", "Emergency Plumbing", "Plumbing"),
    (r"\bleak detection|water leak detection\b", "Leak Detection", "Plumbing"),
    (r"\bleak repair|water leak repair|pipe leak\b", "Leak Repair", "Plumbing"),
    (r"\bblocked drain|drain unblock|unblock.*drain|drain clearance\b", "Drain Unblocking", "Plumbing"),
    (r"\bdrain repair|sewer repair\b", "Drain Repair", "Plumbing"),
    (r"\bpipe repair|water pipe repair|burst pipe\b", "Pipe Repair", "Plumbing"),
    (r"\bpipe install|pipework install\b", "Pipe Installation", "Plumbing"),
    (r"\btap repair|faucet repair\b", "Tap Repair", "Plumbing"),
    (r"\btap install|faucet install\b", "Tap Installation", "Plumbing"),
    (r"\btoilet repair|wc repair\b", "Toilet Repair", "Plumbing"),
    (r"\btoilet install|wc install\b", "Toilet Installation", "Plumbing"),
    (r"\bbathroom plumbing\b", "Bathroom Plumbing", "Plumbing"),
    (r"\bshower repair\b", "Shower Repair", "Plumbing"),
    (r"\bshower install\b", "Shower Installation", "Plumbing"),
    (r"\bplumbing repair|plumber repair\b", "Plumbing Repair", "Plumbing"),
    (r"\bplumb|plumber\b", "General Plumbing", "Plumbing"),

    # --------------------------------------------------------
    # HEATING / BOILERS / GAS
    # --------------------------------------------------------
    (r"\bboiler installation|boiler install\b", "Boiler Installation", "Heating & Boiler Services"),
    (r"\bboiler repair\b", "Boiler Repair", "Heating & Boiler Services"),
    (r"\bboiler service|boiler servicing\b", "Boiler Servicing", "Heating & Boiler Services"),
    (r"\bboiler replacement\b", "Boiler Replacement", "Heating & Boiler Services"),
    (r"\bcentral heating install\b", "Central Heating Installation", "Heating & Boiler Services"),
    (r"\bcentral heating repair\b", "Central Heating Repair", "Heating & Boiler Services"),
    (r"\bradiator install\b", "Radiator Installation", "Heating & Boiler Services"),
    (r"\bradiator repair\b", "Radiator Repair", "Heating & Boiler Services"),
    (r"\bunderfloor heating install\b", "Underfloor Heating Installation", "Heating & Boiler Services"),
    (r"\bunderfloor heating repair\b", "Underfloor Heating Repair", "Heating & Boiler Services"),
    (r"\bgas safety certificate|gas safety check\b", "Gas Safety Inspection", "Gas Services"),
    (r"\bgas appliance install\b", "Gas Appliance Installation", "Gas Services"),
    (r"\bgas appliance repair\b", "Gas Appliance Repair", "Gas Services"),
    (r"\bgas engineer\b", "Gas Engineering", "Gas Services"),
    (r"\bcentral heating|heating engineer|heating system\b", "General Heating Services", "Heating & Boiler Services"),

    # --------------------------------------------------------
    # ELECTRICAL
    # --------------------------------------------------------
    (r"\bemergency electrician\b", "Emergency Electrical", "Electrical Services"),
    (r"\brewir|house rewire|full rewire\b", "Rewiring", "Electrical Services"),
    (r"\bsocket install|plug socket install\b", "Socket Installation", "Electrical Services"),
    (r"\bsocket repair|plug socket repair\b", "Socket Repair", "Electrical Services"),
    (r"\blight install|lighting install\b", "Lighting Installation", "Electrical Services"),
    (r"\blight repair|lighting repair\b", "Lighting Repair", "Electrical Services"),
    (r"\bfuse box|consumer unit\b", "Fuse Box & Consumer Unit Services", "Electrical Services"),
    (r"\belectrical inspection|eicr\b", "Electrical Inspection & EICR", "Electrical Services"),
    (r"\belectrical repair\b", "Electrical Repair", "Electrical Services"),
    (r"\belectrician|electrical services?\b", "General Electrical Services", "Electrical Services"),

    # --------------------------------------------------------
    # ROOFING / GUTTERING
    # --------------------------------------------------------
    (r"\broof install|new roof\b", "Roof Installation", "Roofing & Guttering"),
    (r"\broof repair\b", "Roof Repair", "Roofing & Guttering"),
    (r"\broof replacement\b", "Roof Replacement", "Roofing & Guttering"),
    (r"\bflat roof\b", "Flat Roofing", "Roofing & Guttering"),
    (r"\bslate roof\b", "Slate Roofing", "Roofing & Guttering"),
    (r"\btile roof\b", "Tile Roofing", "Roofing & Guttering"),
    (r"\broof clean\b", "Roof Cleaning", "Roofing & Guttering"),
    (r"\bgutter install\b", "Gutter Installation", "Roofing & Guttering"),
    (r"\bgutter repair\b", "Gutter Repair", "Roofing & Guttering"),
    (r"\bgutter clean\b", "Gutter Cleaning", "Roofing & Guttering"),
    (r"\bfascia|soffit\b", "Fascia & Soffit Services", "Roofing & Guttering"),
    (r"\broofer|roofing\b", "General Roofing", "Roofing & Guttering"),

    # --------------------------------------------------------
    # BUILDING / CONSTRUCTION
    # --------------------------------------------------------
    (r"\bhouse extension|home extension|extension builder\b", "Home Extensions", "Building & Construction"),
    (r"\bloft conversion\b", "Loft Conversion", "Building & Construction"),
    (r"\bgarage conversion\b", "Garage Conversion", "Building & Construction"),
    (r"\bbrick repair|brickwork repair\b", "Brickwork Repair", "Building & Construction"),
    (r"\bbricklay|brickwork\b", "Bricklaying", "Building & Construction"),
    (r"\bmasonry\b", "Masonry", "Building & Construction"),
    (r"\bdemolition\b", "Demolition", "Building & Construction"),
    (r"\bstructural repair\b", "Structural Repair", "Building & Construction"),
    (r"\bbuilder|building contractor|construction contractor\b", "General Building", "Building & Construction"),

    # --------------------------------------------------------
    # CARPENTRY / JOINERY
    # --------------------------------------------------------
    (r"\bfitted wardrobe|wardrobe install\b", "Fitted Wardrobes", "Carpentry & Joinery"),
    (r"\bcabinet mak|cabinet install\b", "Cabinet Making & Installation", "Carpentry & Joinery"),
    (r"\bdoor hang|door fitting\b", "Door Fitting", "Carpentry & Joinery"),
    (r"\bstaircase|stairs carpentry\b", "Staircase Services", "Carpentry & Joinery"),
    (r"\bcarpenter|carpentry|joiner|joinery\b", "General Carpentry & Joinery", "Carpentry & Joinery"),

    # --------------------------------------------------------
    # PAINTING / DECORATING / WALLS
    # --------------------------------------------------------
    (r"\bexterior paint\b", "Exterior Painting", "Painting & Decorating"),
    (r"\binterior paint\b", "Interior Painting", "Painting & Decorating"),
    (r"\bwallpaper\b", "Wallpapering", "Painting & Decorating"),
    (r"\bpaint|decorator|decorating\b", "General Painting & Decorating", "Painting & Decorating"),
    (r"\bplaster repair\b", "Plaster Repair", "Plastering & Rendering"),
    (r"\bplastering\b", "Plastering", "Plastering & Rendering"),
    (r"\brender repair\b", "Render Repair", "Plastering & Rendering"),
    (r"\brendering\b", "Rendering", "Plastering & Rendering"),

    # --------------------------------------------------------
    # FLOORING / TILING
    # --------------------------------------------------------
    (r"\bcarpet fitting|carpet install\b", "Carpet Installation", "Flooring"),
    (r"\blaminate floor\b", "Laminate Flooring", "Flooring"),
    (r"\bvinyl floor\b", "Vinyl Flooring", "Flooring"),
    (r"\bwood floor|hardwood floor\b", "Wood Flooring", "Flooring"),
    (r"\bfloor repair\b", "Floor Repair", "Flooring"),
    (r"\bflooring\b", "General Flooring", "Flooring"),
    (r"\bbathroom til|wall til\b", "Wall & Bathroom Tiling", "Tiling"),
    (r"\bfloor til\b", "Floor Tiling", "Tiling"),
    (r"\btiler|tiling\b", "General Tiling", "Tiling"),

    # --------------------------------------------------------
    # WINDOWS / DOORS / GLAZING
    # --------------------------------------------------------
    (r"\bwindow install|window fitting\b", "Window Installation", "Windows & Glazing"),
    (r"\bwindow repair\b", "Window Repair", "Windows & Glazing"),
    (r"\bdouble glazing\b", "Double Glazing", "Windows & Glazing"),
    (r"\bglazier|glass repair\b", "Glazing & Glass Repair", "Windows & Glazing"),
    (r"\bdoor install|door fitting\b", "Door Installation", "Doors & Door Services"),
    (r"\bdoor repair\b", "Door Repair", "Doors & Door Services"),
    (r"\bgarage door install\b", "Garage Door Installation", "Doors & Door Services"),
    (r"\bgarage door repair\b", "Garage Door Repair", "Doors & Door Services"),

    # --------------------------------------------------------
    # KITCHEN / BATHROOM
    # --------------------------------------------------------
    (r"\bkitchen install|kitchen fitting\b", "Kitchen Installation", "Kitchen Services"),
    (r"\bkitchen repair\b", "Kitchen Repair", "Kitchen Services"),
    (r"\bworktop install|countertop install\b", "Worktop Installation", "Kitchen Services"),
    (r"\bworktop repair|countertop repair\b", "Worktop Repair", "Kitchen Services"),
    (r"\bbathroom install|bathroom fitting\b", "Bathroom Installation", "Bathroom Services"),
    (r"\bbathroom repair\b", "Bathroom Repair", "Bathroom Services"),
    (r"\bwet room install\b", "Wet Room Installation", "Bathroom Services"),

    # --------------------------------------------------------
    # GARDEN / LANDSCAPING
    # --------------------------------------------------------
    (r"\blawn mowing|grass cutting\b", "Lawn Mowing", "Gardening & Landscaping"),
    (r"\blawn care\b", "Lawn Care", "Gardening & Landscaping"),
    (r"\bhedge trim|hedge cutting\b", "Hedge Trimming", "Gardening & Landscaping"),
    (r"\btree removal|tree cutting|tree surgeon\b", "Tree Surgery", "Gardening & Landscaping"),
    (r"\blandscaping\b", "Landscaping", "Gardening & Landscaping"),
    (r"\bgarden design\b", "Garden Design", "Gardening & Landscaping"),
    (r"\bgarden maintenance\b", "Garden Maintenance", "Gardening & Landscaping"),
    (r"\bgardener|gardening\b", "General Gardening", "Gardening & Landscaping"),

    # --------------------------------------------------------
    # FENCING / PAVING / OUTDOORS
    # --------------------------------------------------------
    (r"\bfence install\b", "Fence Installation", "Fencing & Gates"),
    (r"\bfence repair\b", "Fence Repair", "Fencing & Gates"),
    (r"\bgate install\b", "Gate Installation", "Fencing & Gates"),
    (r"\bgate repair\b", "Gate Repair", "Fencing & Gates"),
    (r"\bdriveway install|driveway paving\b", "Driveway Installation", "Driveways & Paving"),
    (r"\bdriveway repair\b", "Driveway Repair", "Driveways & Paving"),
    (r"\bpatio install|patio laying\b", "Patio Installation", "Driveways & Paving"),
    (r"\bpaving\b", "Paving", "Driveways & Paving"),

    # --------------------------------------------------------
    # CLEANING
    # --------------------------------------------------------
    (r"\bend of tenancy clean\b", "End of Tenancy Cleaning", "Cleaning"),
    (r"\bdomestic clean|house clean|home clean\b", "Domestic Cleaning", "Cleaning"),
    (r"\boffice clean|commercial clean\b", "Commercial Cleaning", "Cleaning"),
    (r"\bdeep clean\b", "Deep Cleaning", "Cleaning"),
    (r"\bcarpet clean\b", "Carpet Cleaning", "Cleaning"),
    (r"\bupholstery clean\b", "Upholstery Cleaning", "Cleaning"),
    (r"\bwindow clean\b", "Window Cleaning", "Cleaning"),
    (r"\boven clean\b", "Oven Cleaning", "Cleaning"),
    (r"\bpressure wash|jet wash\b", "Pressure Washing", "Cleaning"),
    (r"\bcleaner|cleaning\b", "General Cleaning", "Cleaning"),

    # --------------------------------------------------------
    # REMOVALS / WASTE
    # --------------------------------------------------------
    (r"\bhouse removal|home removal\b", "House Removals", "Removals & Moving"),
    (r"\boffice removal\b", "Office Removals", "Removals & Moving"),
    (r"\bman and van\b", "Man & Van", "Removals & Moving"),
    (r"\bfurniture removal\b", "Furniture Removals", "Removals & Moving"),
    (r"\bremoval|moving service|mover\b", "General Removals", "Removals & Moving"),
    (r"\brubbish removal|junk removal\b", "Rubbish Removal", "Waste Services"),
    (r"\bgarden waste\b", "Garden Waste Removal", "Waste Services"),
    (r"\bskip hire\b", "Skip Hire", "Waste Services"),

    # --------------------------------------------------------
    # REPAIR / ASSEMBLY
    # --------------------------------------------------------
    (r"\bwashing machine repair\b", "Washing Machine Repair", "Appliance Repair"),
    (r"\bdishwasher repair\b", "Dishwasher Repair", "Appliance Repair"),
    (r"\bfridge|refrigerator repair\b", "Fridge Repair", "Appliance Repair"),
    (r"\boven repair\b", "Oven Repair", "Appliance Repair"),
    (r"\bappliance repair\b", "General Appliance Repair", "Appliance Repair"),
    (r"\bfurniture assembly|flat pack\b", "Furniture Assembly", "Furniture Services"),
    (r"\bfurniture repair\b", "Furniture Repair", "Furniture Services"),

    # --------------------------------------------------------
    # SECURITY / LOCKSMITH
    # --------------------------------------------------------
    (r"\bemergency locksmith\b", "Emergency Locksmith", "Locksmith"),
    (r"\block change|lock replacement\b", "Lock Replacement", "Locksmith"),
    (r"\block repair\b", "Lock Repair", "Locksmith"),
    (r"\blocksmith\b", "General Locksmith", "Locksmith"),
    (r"\bcctv install\b", "CCTV Installation", "Security Systems"),
    (r"\balarm install\b", "Alarm Installation", "Security Systems"),
    (r"\balarm repair\b", "Alarm Repair", "Security Systems"),
    (r"\baccess control\b", "Access Control Systems", "Security Systems"),

    # --------------------------------------------------------
    # BEAUTY / WELLNESS
    # --------------------------------------------------------
    (r"\bmen'?s haircut|barber haircut\b", "Men's Haircut", "Hair & Barber Services"),
    (r"\bwomen'?s haircut|ladies haircut\b", "Women's Haircut", "Hair & Barber Services"),
    (r"\bhair colour|hair color|balayage|highlights\b", "Hair Colouring", "Hair & Barber Services"),
    (r"\bhair extension\b", "Hair Extensions", "Hair & Barber Services"),
    (r"\bbarber\b", "Barber Services", "Hair & Barber Services"),
    (r"\bhairdress|hair salon|hair stylist\b", "Hairdressing", "Hair & Barber Services"),
    (r"\bmanicure\b", "Manicure", "Nail Services"),
    (r"\bpedicure\b", "Pedicure", "Nail Services"),
    (r"\bacrylic nail\b", "Acrylic Nails", "Nail Services"),
    (r"\bgel nail\b", "Gel Nails", "Nail Services"),
    (r"\bnail\b", "General Nail Services", "Nail Services"),
    (r"\bdeep tissue massage\b", "Deep Tissue Massage", "Massage"),
    (r"\bsports massage\b", "Sports Massage", "Massage"),
    (r"\bswedish massage\b", "Swedish Massage", "Massage"),
    (r"\bmassage\b", "General Massage", "Massage"),
    (r"\bwaxing\b", "Waxing", "Hair Removal"),
    (r"\blaser hair removal\b", "Laser Hair Removal", "Hair Removal"),
    (r"\bthreading\b", "Threading", "Hair Removal"),
    (r"\bfacial\b", "Facials", "Beauty & Skincare"),
    (r"\beyebrow|brow\b", "Eyebrow Services", "Beauty & Skincare"),
    (r"\beyelash|lash\b", "Eyelash Services", "Beauty & Skincare"),
    (r"\bskin care|skincare\b", "Skincare Treatments", "Beauty & Skincare"),
    (r"\btattoo\b", "Tattoo Services", "Tattoo & Piercing"),
    (r"\bpiercing\b", "Piercing Services", "Tattoo & Piercing"),
    (r"\btanning\b", "Tanning", "Beauty & Wellness"),
    (r"\bmedspa|aesthetic\b", "Aesthetic Treatments", "Beauty & Wellness"),

    # --------------------------------------------------------
    # FITNESS / THERAPY
    # --------------------------------------------------------
    (r"\bpersonal trainer\b", "Personal Training", "Health & Fitness"),
    (r"\byoga\b", "Yoga", "Health & Fitness"),
    (r"\bphysio|physiotherapy|physical therapy\b", "Physiotherapy", "Health & Fitness"),
    (r"\bcounselling|counseling\b", "Counselling", "Therapy & Counselling"),
    (r"\btherapy|therapist\b", "Therapy", "Therapy & Counselling"),

    # --------------------------------------------------------
    # PET SERVICES
    # --------------------------------------------------------
    (r"\bdog grooming|pet grooming\b", "Pet Grooming", "Pet Services"),
    (r"\bdog walking\b", "Dog Walking", "Pet Services"),
    (r"\bpet sitting|dog sitting|cat sitting\b", "Pet Sitting", "Pet Services"),
    (r"\bdog training|pet training\b", "Pet Training", "Pet Services"),

    # --------------------------------------------------------
    # PROFESSIONAL / EVENTS / LESSONS
    # --------------------------------------------------------
    (r"\bwedding photographer\b", "Wedding Photography", "Photography"),
    (r"\bphotographer|photography\b", "General Photography", "Photography"),
    (r"\bvideographer|videography\b", "Videography", "Events & Creative"),
    (r"\bdj\b", "DJ Services", "Events & Creative"),
    (r"\bcatering|caterer\b", "Catering", "Events & Creative"),
    (r"\bwedding planner\b", "Wedding Planning", "Event Planning"),
    (r"\bevent planner|party planner\b", "Event Planning", "Event Planning"),
    (r"\bmath tutor|maths tutor\b", "Maths Tutoring", "Tutoring & Lessons"),
    (r"\benglish tutor\b", "English Tutoring", "Tutoring & Lessons"),
    (r"\bmusic lessons?|music teacher\b", "Music Lessons", "Tutoring & Lessons"),
    (r"\btutor|tuition|lessons\b", "General Tutoring & Lessons", "Tutoring & Lessons"),
    (r"\baccountant\b", "Accounting", "Professional Services"),
    (r"\bbookkeep\b", "Bookkeeping", "Professional Services"),
    (r"\bweb design|website design\b", "Web Design", "Professional Services"),
    (r"\bweb developer|website developer\b", "Web Development", "Professional Services"),
    (r"\bseo\b", "SEO Services", "Professional Services"),
    (r"\bsocial media\b", "Social Media Marketing", "Professional Services"),
    (r"\bmarketing\b", "Marketing Services", "Professional Services"),
]



# ============================================================
# CURATED SEVERSE SCOPE
# Keep categories that fit a local-services marketplace.
# Drop navigation, locations, generic content, and categories
# that are not useful for the current market-entry analysis.
# ============================================================

EXCLUDED_FAMILY_NAMES = {
    "Noise / Excluded",
}

EXCLUDED_RAW_PATTERNS = [
    # navigation / platform content
    r"\babout us\b",
    r"\baffiliate\b",
    r"\bblog\b",
    r"\bcareers?\b",
    r"\bcontact us\b",
    r"\bdownload app\b",
    r"\bfaq\b",
    r"\bfor tradespeople\b",
    r"\bgetting started\b",
    r"\bhow it works\b",
    r"\blog in\b",
    r"\bnews\b",
    r"\boverview\b",
    r"\bprivacy\b",
    r"\breviews?\b",
    r"\bsign up\b",
    r"\bsitemap\b",
    r"\bterms\b",
    r"\btestimonials?\b",
    r"\bwin more work\b",

    # obvious location contamination / geographic-only rows
    r"^(?:[a-z][a-z .'-]+shire|london|england|scotland|wales|northern ireland)$",

    # contact / malformed rows
    r"^\+?\d[\d\s\-()]{7,}$",
    r"^[^\w]+$",
]

# Families we consider useful for SEVERSE's service marketplace universe.
# This is intentionally broad enough for later market analysis while excluding
# clearly irrelevant platform/navigation content.
KEEP_FAMILY_KEYWORDS = [
    "plumbing",
    "heating",
    "boiler",
    "gas",
    "electrical",
    "roof",
    "building",
    "construction",
    "carpentry",
    "joinery",
    "painting",
    "decorating",
    "plaster",
    "render",
    "floor",
    "tiling",
    "window",
    "glazing",
    "door",
    "kitchen",
    "bathroom",
    "garden",
    "landscap",
    "fencing",
    "driveway",
    "paving",
    "cleaning",
    "removal",
    "waste",
    "appliance",
    "furniture",
    "locksmith",
    "security",
    "hair",
    "barber",
    "nail",
    "massage",
    "beauty",
    "wellness",
    "fitness",
    "therapy",
    "pet",
    "photography",
    "event",
    "tutoring",
    "professional",
    "vehicle",
    "delivery",
    "renewable",
    "insulation",
    "conservatory",
    "chimney",
    "air conditioning",
    "ventilation",
    "pest",
    "fire",
    "blind",
    "shutter",
    "pool",
    "hot tub",
    "ceiling",
    "smart home",
    "automation",
    "accessibility",
    "mobility",
    "restoration",
]

def should_exclude_raw(raw_service: str) -> bool:
    text = normalise_for_match(raw_service)

    if not text:
        return True

    for pattern in EXCLUDED_RAW_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return True

    return False


def is_keep_family(service_family: str) -> bool:
    family = normalise_for_match(service_family)

    if service_family in EXCLUDED_FAMILY_NAMES:
        return False

    return any(keyword in family for keyword in KEEP_FAMILY_KEYWORDS)


def classify(raw_service: str) -> tuple[str, str, str]:
    text = normalise_space(raw_service)
    match_text = normalise_for_match(text)

    # 1) Exclude only obvious junk / navigation / location contamination.
    if should_exclude_raw(text):
        return "Excluded", text, "excluded"

    for pattern in NOISE_PATTERNS:
        if re.search(pattern, match_text, flags=re.IGNORECASE):
            return "Excluded", text, "excluded"

    # 2) Keep everything that matches a known service taxonomy rule.
    for pattern, sub_service, service_family in EXPANDED_RULES + RULES:
        if match(pattern, match_text):
            service_family = FAMILY_RENAMES.get(service_family, service_family)
            return service_family, sub_service, "kept_mapped"

    # 3) IMPORTANT:
    # If it looks like a genuine service but is not yet mapped,
    # KEEP it rather than excluding it.
    return "Other Services / Unclassified", text, "kept_unmapped"


# ============================================================
# PIPELINE
# ============================================================

def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE).copy()

    required = {"platform", "raw_service"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    df["platform"] = df["platform"].map(normalise_space)
    df["raw_service"] = df["raw_service"].map(normalise_space)

    classification = df["raw_service"].map(classify)

    df["service_family"] = classification.map(lambda x: x[0])
    df["sub_service"] = classification.map(lambda x: x[1])
    df["taxonomy_status"] = classification.map(lambda x: x[2])

    # Save row-level taxonomy mapping
    preferred_columns = [
        "platform",
        "raw_service_original",
        "raw_service",
        "service_family",
        "sub_service",
        "taxonomy_status",
        "source_version",
        "source_url",
        "source_type",
        "collection_date",
        "coverage_note",
        "geography_scope",
    ]

    output_columns = [
        col
        for col in preferred_columns
        if col in df.columns
    ]

    df[output_columns].to_csv(
        TAXONOMY_OUTPUT,
        index=False,
    )

    # Canonical service coverage across platforms
    coverage = (
        df[df["taxonomy_status"].isin(["kept_mapped", "kept_unmapped"])]
        .groupby(
            ["service_family", "sub_service"],
            as_index=False,
        )
        .agg(
            platform_count=("platform", "nunique"),
            platforms=(
                "platform",
                lambda values:
                    " | ".join(
                        sorted(
                            set(values),
                            key=str.casefold,
                        )
                    ),
            ),
            raw_label_count=("raw_service", "nunique"),
            row_count=("raw_service", "size"),
        )
        .sort_values(
            [
                "platform_count",
                "raw_label_count",
                "service_family",
                "sub_service",
            ],
            ascending=[
                False,
                False,
                True,
                True,
            ],
        )
    )

    coverage.to_csv(
        COVERAGE_OUTPUT,
        index=False,
    )

    # Save excluded rows for audit / traceability.
    excluded = (
        df[
            df["taxonomy_status"] == "excluded"
        ][
            ["platform", "raw_service"]
        ]
        .drop_duplicates()
        .sort_values(
            ["raw_service", "platform"],
            key=lambda col: col.str.casefold(),
        )
    )

    excluded.to_csv(
        UNMAPPED_OUTPUT,
        index=False,
    )

    summary = pd.DataFrame(
        [
            {
                "metric": "input_rows",
                "value": len(df),
            },
            {
                "metric": "unique_raw_services",
                "value": df["raw_service"].nunique(),
            },
            {
                "metric": "kept_mapped_rows",
                "value": int(
                    (df["taxonomy_status"] == "kept_mapped").sum()
                ),
            },
            {
                "metric": "kept_unmapped_rows",
                "value": int(
                    (df["taxonomy_status"] == "kept_unmapped").sum()
                ),
            },
            {
                "metric": "kept_total_rows",
                "value": int(
                    df["taxonomy_status"].isin(["kept_mapped", "kept_unmapped"]).sum()
                ),
            },
            {
                "metric": "excluded_rows",
                "value": int(
                    (df["taxonomy_status"] == "excluded").sum()
                ),
            },
            {
                "metric": "sub_services",
                "value": df["sub_service"].nunique(),
            },
            {
                "metric": "service_families",
                "value": df["service_family"].nunique(),
            },
        ]
    )

    summary.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    kept_mapped_rows = int((df["taxonomy_status"] == "kept_mapped").sum())
    kept_unmapped_rows = int((df["taxonomy_status"] == "kept_unmapped").sum())
    kept_rows = kept_mapped_rows + kept_unmapped_rows
    excluded_rows = int((df["taxonomy_status"] == "excluded").sum())
    kept_rate = (kept_rows / len(df) * 100) if len(df) else 0

    print("=" * 78)
    print("SEVERSE FINAL CURATED SERVICE TAXONOMY")
    print("=" * 78)

    print(f"\nInput rows: {len(df):,}")
    print(
        f"Unique raw service labels: "
        f"{df['raw_service'].nunique():,}"
    )
    print(f"Kept mapped rows: {kept_mapped_rows:,}")
    print(f"Kept unmapped rows: {kept_unmapped_rows:,}")
    print(f"Total kept rows: {kept_rows:,}")
    print(f"Excluded rows: {excluded_rows:,}")
    print(f"Kept rate: {kept_rate:.1f}%")
    print(
        f"Service families kept: "
        f"{df.loc[df['taxonomy_status'].isin(['kept_mapped', 'kept_unmapped']), 'service_family'].nunique():,}"
    )
    print(
        f"Sub-services kept: "
        f"{df.loc[df['taxonomy_status'].isin(['kept_mapped', 'kept_unmapped']), 'sub_service'].nunique():,}"
    )


    print("\nTop service families / sub-services by platform coverage:\n")

    print(
        coverage.head(40).to_string(index=False)
    )

    print("\nFiles written:")
    print(f"  {TAXONOMY_OUTPUT}")
    print(f"  {COVERAGE_OUTPUT}")
    print(f"  {UNMAPPED_OUTPUT}")
    print(f"  {SUMMARY_OUTPUT}")

    print(
        "\nNEXT STEP: use service_taxonomy_final.csv as the working service universe. "
        "Only obvious noise is excluded. Genuine but not-yet-mapped services are retained under Other Services / Unclassified."
    )


if __name__ == "__main__":
    main()

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
    / "taxonomy"
    / "service_taxonomy_final_master_v2.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "processed-data"
    / "market-entry"
    / "service-categories"
    / "taxonomy"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FINAL_OUTPUT = OUTPUT_DIR / "service_taxonomy_final_master_resolved.csv"
COVERAGE_OUTPUT = OUTPUT_DIR / "service_taxonomy_final_master_resolved_coverage.csv"
REMAINING_OUTPUT = OUTPUT_DIR / "service_taxonomy_final_residual_resolved.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "service_taxonomy_final_master_resolved_summary.csv"


# ============================================================
# HELPERS
# ============================================================

def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def norm(value: str) -> str:
    text = clean_text(value).casefold()
    text = text.replace("&", "and")
    text = re.sub(r"[/_-]+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def has(pattern: str, text: str) -> bool:
    return bool(re.search(pattern, text, flags=re.I))


# ============================================================
# NOISE
# Only obvious junk/location/navigation is excluded.
# Genuine services are NEVER excluded simply because they
# do not match a family.
# ============================================================

NOISE_PATTERNS = [
    r"^\+?\d[\d\s\-()]{7,}$",
    r"^(about us|affiliates?|ask a tradesperson|build your reputation|built|careers?|check any tradesperson|checkatrade standard|contact us|cost guides|create a job|discover|download app|download on the app store|eligibility criteria.*|feature your business.*|find a tradesman|find a tradesperson|for tradespeople|get it on google play|getting started|help and faqs|homeowner advice centre|how it works|information centre|inspiration centre|log in|member t and c.*|overview|perks|popular locations|post a job|privacy|rated group|reviewed tradespeople|reviews?|sale|sign up|sitemap|terms of use|terms|testimonials?|trade advice centre|tradesperson sign up|trends report|user agreement|view our prices|view prices|win more work|your profile shows up on google.*)$",
]

LOCATION_NAMES = {
    "aberdeen","angus","antrim","argyll and bute","armagh","ayrshire and arran",
    "banffshire","bedfordshire","berkshire","berwickshire","birkenhead","bolton",
    "bournemouth","buckinghamshire","caithness","cambridgeshire","chelmsford",
    "cheshire","derbyshire","devon","dorset","dumfries","dunbartonshire","durham",
    "dyfed","east lothian","east riding of yorkshire","east sussex","gateshead",
    "glasgow area","gloucester","gloucestershire","ipswich","isle of wight","kent",
    "kincardineshire","lanarkshire","lancashire","moray","nairn","newcastle upon tyne",
    "norfolk","north west","north yorkshire","northampton","northamptonshire",
    "northumberland","norwich","nottinghamshire","orkney","oxford","oxfordshire",
    "reading","renfrewshire","ross and cromarty","roxburgh ettrick and lauderdale",
    "rutland","staffordshire","stirling and falkirk","stockport","stoke on trent",
    "strathclyde","suffolk","sunderland","surrey","sutherland",
    "the stewartry of kirkcudbright","warwickshire","watford","west glamorgan",
    "west lothian","west midlands","west sussex","west yorkshire","western isles",
    "wigtown","wiltshire","cornwall","cumbria","essex","fermanagh","fife","greater manchester","gwent","gwynedd","hampshire","herefordshire","hertfordshire","huddersfield","inverness","leicestershire","lincolnshire","londonderry","lothians","maidstone","merseyside","mid glamorgan","middlesbrough","midlothian","milton keynes","perth and kinross","peterborough","plymouth","poole","powys","shetland","shropshire","slough","solihull","somerset","south glamorgan","south yorkshire","southampton","tweeddale","tyne and wear","tyrone","wolverhampton","worcestershire","york","city of aberdeen","city of dundee","city of edinburgh","city of glasgow","clackmannan","clwyd"
}


# ============================================================
# SUB-SERVICE CANONICALISATION
# These preserve meaningful distinctions, including separate
# hair/barber sub-services instead of collapsing them into beauty.
# ============================================================

SUBSERVICE_RULES = [

    # Software / Development
    (r"\bandroid app development\b", "Android App Development"),
    (r"\bios app development|ios development\b", "iOS App Development"),
    (r"\bapp development\b", "App Development"),
    (r"\bsoftware development\b", "Software Development"),
    (r"\bsql\b", "SQL / Database Services"),
    (r"\bdatabase\b", "Database Services"),
    (r"\bsoftware testing|website and app testing\b", "Software Testing"),

    # Web / Digital
    (r"\bweb design|website design\b", "Web Design"),
    (r"\bweb development|website development\b", "Web Development"),
    (r"\bseo\b", "SEO Services"),
    (r"\bweb hosting\b", "Web Hosting"),
    (r"\bsocial media\b", "Social Media Marketing"),

    # IT Support
    (r"\bapple help|mac support|apple mac repairs?\b", "Apple / Mac Support"),
    (r"\biphone help|iphone repair\b", "iPhone Support & Repair"),
    (r"\blaptop repair\b", "Laptop Repair"),
    (r"\bcomputer repair|pc repair\b", "Computer Repair"),
    (r"\bwifi|wi fi|broadband|router\b", "WiFi & Broadband Support"),
    (r"\bnetwork support|network administration|network troubleshooting\b", "Network Support"),
    (r"\bit support|technical support\b", "General IT Support"),

    # TV / AV / Aerial
    (r"\bapple tv setup\b", "Apple TV Setup"),
    (r"\btv wall mount|tv mounting\b", "TV Wall Mounting"),
    (r"\btv install|television install\b", "TV Installation"),
    (r"\btv repair|television repair\b", "TV Repair"),
    (r"\baerial install|antenna install\b", "Aerial Installation"),
    (r"\baerial repair|antenna repair\b", "Aerial Repair"),
    (r"\bsatellite.*install|satellite dish\b", "Satellite Installation"),
    (r"\baudio visual install\b", "Audio Visual Installation"),

    # Professional
    (r"\baccountant|accounting\b", "Accounting"),
    (r"\bbookkeep\b", "Bookkeeping"),
    (r"\btax advisor|tax advice\b", "Tax Advisory"),
    (r"\barticle writing\b", "Article Writing"),
    (r"\bblog writing\b", "Blog Writing"),
    (r"\bcopywriting\b", "Copywriting"),
    (r"\bproofread\b", "Proofreading"),
    (r"\bresume writing|cv writing\b", "CV / Resume Writing"),
    (r"\btechnical writ", "Technical Writing"),
    (r"\btranslation\b", "Translation"),
    (r"\btranscription\b", "Transcription"),
    (r"\bcareer coach\b", "Career Coaching"),

    # Vehicle
    (r"\bbrake pad replacement\b", "Brake Pad Replacement"),
    (r"\bbrake repair\b", "Brake Repair"),
    (r"\balternator replacement\b", "Alternator Replacement"),
    (r"\balternator repair\b", "Alternator Repair"),
    (r"\balloy wheel", "Alloy Wheel Repair"),
    (r"\btyre|tire\b", "Tyre Services"),
    (r"\bcar service|audi service|bmw service|alfa romeo service\b", "Vehicle Servicing"),

    # Childcare / care
    (r"\bbabysitter\b", "Babysitting"),
    (r"\bchildminder\b", "Childminding"),
    (r"\bnanny\b", "Nanny Services"),
    (r"\bafter school care\b", "After School Care"),
    (r"\baged care\b", "Aged Care"),

    # Hair / Barber / Beauty
    (r"\bafro barber|afro barbers\b", "Afro Barbering"),
    (r"\bmobile barber|mobile barbers\b", "Mobile Barbering"),
    (r"\bbarbing|barbering\b", "General Barbering"),
    (r"\bbarber|barbers\b", "General Barbering"),
    (r"\bcurly hair specialist|curly hair specialists\b", "Curly Hair Services"),
    (r"\bhair braiding|braids|braiding\b", "Hair Braiding"),
    (r"\bhaircut and trimming|haircut.*trim|hair trimming\b", "Haircut & Trimming"),
    (r"\bkids haircut|children'?s haircut|child haircut\b", "Kids Haircuts"),
    (r"\bfade buzz|buzz cut|fade haircut|skin fade\b", "Fade / Buzz Cuts"),
    (r"\bbeard grooming\b", "Beard Grooming"),
    (r"\bblow dry\b", "Blow Dry"),
    (r"\blocs|dreadlocks\b", "Locs / Dreadlocks"),
    (r"\bhair colour|hair color|balayage|highlights\b", "Hair Colouring"),
    (r"\bhair extension|hair extensions\b", "Hair Extensions"),
    (r"\bhairdress|hair stylist|hair salon\b", "Hairdressing"),
    (r"\bbeautician|beauty salon|beauty salons\b", "Beauty Salon Services"),
    (r"^salon$", "General Salon Services"),

    # Building / interiors
    (r"\barchitectural drawings?\b", "Architectural Drawings"),
    (r"\barchitectural technician\b", "Architectural Technician"),
    (r"\barchitect", "Architectural Services"),
    (r"\bbuilding surveyor\b", "Building Surveying"),
    (r"\bquantity surveyor\b", "Quantity Surveying"),
    (r"\bparty wall\b", "Party Wall Surveying"),
    (r"\bstructural engineer\b", "Structural Engineering"),
    (r"\basbestos survey\b", "Asbestos Survey"),
    (r"\basbestos testing\b", "Asbestos Testing"),
    (r"\basbestos\b", "Asbestos Services"),

    # Furniture / woodwork
    (r"\bfurniture restoration|antique furniture restoration|french polishing\b", "Furniture Restoration"),
    (r"\bfurniture repair\b", "Furniture Repair"),
    (r"\bbespoke furniture\b", "Bespoke Furniture"),
    (r"\bcabinet maker|cabinet making\b", "Cabinet Making"),
    (r"\bwoodwork|wood worker|woodworking\b", "Woodworking"),
    (r"\bjoiner|joinery\b", "Joinery"),
    (r"\bcarpenter|carpentry\b", "Carpentry"),

    # Events
    (r"\bwedding planner\b", "Wedding Planning"),
    (r"\bwedding photographer\b", "Wedding Photography"),
    (r"\bwedding band|wedding musicians?|wedding singers?\b", "Wedding Music"),
    (r"\bwedding cake\b", "Wedding Cakes"),
    (r"\bwedding florist\b", "Wedding Floristry"),
    (r"\bbouncy castle\b", "Bouncy Castle Hire"),
    (r"\bballoon artist\b", "Balloon Artist"),
    (r"\bbartend|barista|waiter|waitress\b", "Hospitality Staffing"),
]


# ============================================================
# FAMILY RULES
#
# Key principle:
#   Web design + web development CAN stay together.
#   App/software/database development is separate.
#   IT/device/WiFi support is separate.
#   TV/aerial/audio-visual is separate.
#
# The sub_service remains specific.
# ============================================================

FAMILY_RULES = [

    ("Software & Development Services",
     r"\b(app development|android app|ios development|software development|software engineer|software testing|sql|database|java developer|python developer|ruby developer|programming|developer)\b"),

    ("Web & Digital Services",
     r"\b(web design|website design|web development|website development|seo|web hosting|wordpress|shopify|ecommerce|e commerce|social media|digital marketing|email marketing)\b"),

    ("IT Support & Device Services",
     r"\b(it support|technical support|computer|laptop|pc repair|apple help|apple mac|iphone help|iphone repair|mobile phone repair|tablet repair|wifi|wi fi|broadband|router|network administration|network support|network troubleshooting|data recovery|printer repair)\b"),

    ("TV, Aerial & Audio Visual Services",
     r"\b(tv |television|apple tv|aerial|satellite|antenna|audio visual|home cinema|amplifier repair|speaker repair)\b"),

    ("Accounting & Finance Services",
     r"\b(accountant|accounting|bookkeep|tax advisor|tax advice|budgeting help|mortgage advisor|financial advisor|payroll)\b"),

    ("Writing, Translation & Language Services",
     r"\b(article writing|blog writing|copywriting|proofread|resume writing|cv writing|technical writ|ghostwrit|content writing|translation|translator|transcription|editing)\b"),

    ("Business & Administrative Services",
     r"\b(admin|administrative|business consultant|business consulting|business coach|career coach|project management|receptionist|virtual assistant|research assistant|document filing|business advisory)\b"),

    ("Marketing & Creative Services",
     r"\b(advertising|media buying|branding|graphic design|photoshop designer|autocad designer|cad designer|flyer design|logo design|illustrator|creative design)\b"),

    ("Architecture, Surveying & Planning",
     r"\b(architect|architectural|surveyor|surveying|structural engineer|planning consultant|building designer|quantity surveyor|party wall)\b"),

    ("Childcare & Care Services",
     r"\b(babysitter|babysitting|childcare|childminder|nanny|after school care|aged care|elderly care|mother'?s help|caregiver|carer)\b"),

    ("Vehicle Services",
     r"\b(car repair|vehicle repair|auto repair|motor repair|mechanic|audi service|bmw service|alfa romeo service|alternator|brake|tyre|tire|alloy wheel|battery|exhaust|breakdown assistance|car service|vehicle service|car valeting|car detailing|bike repair|bicycle repair|bicycle service|motorcycle repair)\b"),

    ("Furniture, Carpentry & Woodwork",
     r"\b(furniture|woodwork|woodworking|carpenter|carpentry|joiner|joinery|cabinet|wardrobe|bookcase|bespoke.*wood|bespoke.*door|bespoke.*window|bed frame repair|chair repair|recliner repair)\b"),

    ("Events & Wedding Services",
     r"\b(wedding|event planner|party planner|band hire|bagpiper|balloon artist|bouncy castle|birthday grams|bartending|barista|wait staff|waiter|waitress|florist|catering|caterer|entertainment)\b"),

    ("Photography & Video Services",
     r"\b(photographer|photography|videographer|videography|photo editing|photo booth)\b"),

    ("Hair & Barber Services",
     r"\b(afro barber|afro barbers|barber|barbers|barbing|barbering|mobile barber|mobile barbers|curly hair|hair braiding|braiding|braids|locs|dreadlocks|haircut|hair trimming|kids haircut|children'?s haircut|fade buzz|buzz cut|skin fade|beard grooming|blow dry|hairdress|hair stylist|hair salon|hair colour|hair color|balayage|highlights|hair extension|hair extensions)\b"),

    ("Beauty & Personal Care",
     r"\b(beautician|beauty salon|beauty salons|eyebrow|eyelash|facial|waxing|threading|tanning|makeup artist|make up artist|skin care|skincare|aesthetic|salon)\b"),

    ("Fitness & Sports Coaching",
     r"\b(personal train|fitness|aerobics|boxing instructor|basketball training|bootcamp|kickboxing|martial arts|judo|jiu jitsu|taekwondo|sports coaching|yoga|pilates|weight loss training)\b"),

    ("Health, Therapy & Wellness",
     r"\b(acupuncture|physio|physiotherapy|podiatrist|chiropodist|nutritionist|dietitian|counsell|counsel|therapy|therapist|massage|osteopath|chiropract)\b"),

    ("Pet & Animal Services",
     r"\b(pet |dog |cat |bird sitting|animal |aquarium|aviary|grooming|dog walking|pet sitting|pet boarding|dog boarding|cat boarding|dog training|pet training)\b"),

    ("Agricultural & Rural Services",
     r"\b(agricultural|farm |farming|tractor|borehole|irrigation)\b"),

    ("Accessibility & Mobility Services",
     r"\b(accessible|access ramp|wheelchair|stairlift|adjustable beds?|mobility)\b"),

    ("Asbestos & Hazardous Material Services",
     r"\b(asbestos|hazardous material)\b"),

    ("Security & Alarm Services",
     r"\b(alarm|security system|cctv|biometric security|door entry|access control|alarm response)\b"),

    ("Doors, Windows & Glazing",
     r"\b(window|glazing|glazier|door |doors |double glazing|board up service|awning|canopy)\b"),

    ("Roofing Services",
     r"\b(roof|gutter|guttering|fascia|soffit|leadwork|flashing|ridge tile|thatched)\b"),

    ("Plumbing & Drainage Services",
     r"\b(plumb|plumber|drain|blocked bath|blocked sink|blocked toilet|blocked loo|tap |toilet|shower|water main|pipe repair|pipework)\b"),

    ("Heating, Boiler & Gas Services",
     r"\b(boiler|central heating|radiator|gas engineer|gas fitter|gas cooker|gas hob|gas fire|oil fired|oil tank|thermostat|storage heating)\b"),

    ("Electrical Services",
     r"\b(electrician|electrical|rewir|consumer unit|fuse box|eicr|socket|downlight|electric cooker|ev charger|car charging point)\b"),

    ("Building & Construction",
     r"\b(builder|building |construction|brickwork|bricklay|masonry|extension|loft conversion|garage conversion|barn conversion|basement conversion|cellar conversion|demolition|foundation|repointing|retaining wall|refurbishment|renovation|structural repair|rsj)\b"),

    ("Bathroom Services",
     r"\b(bathroom|bathtub|bath resurfacing|bath crack|bath replacement|wet room)\b"),

    ("Kitchen Services",
     r"\b(kitchen|worktop|countertop)\b"),

    ("Flooring & Tiling Services",
     r"\b(floor|flooring|carpet fitting|carpet repair|laminate|vinyl floor|tiler|tiling|tile fitting|floor sanding|floor screeding)\b"),

    ("Painting, Decorating & Wall Finishes",
     r"\b(paint|decorator|decorating|wallpaper|plaster|rendering|artex|dry lining|ceiling)\b"),

    ("Gardening & Landscaping",
     r"\b(garden|gardener|gardening|landscap|lawn|hedge|tree surgeon|arborist|stump|weed control|artificial grass|astro turf|pond|shed|gazebo|pergola)\b"),

    ("Fencing, Gates & Outdoor Structures",
     r"\b(fence|fencing|gate |gates |balustrad|railing|pergola)\b"),

    ("Driveways, Paving & Surfacing",
     r"\b(driveway|paving|patio|resin bound|resin bond|tarmac|aggregate surfacing)\b"),

    ("Cleaning Services",
     r"\b(cleaner|cleaning|deep clean|window clean|carpet clean|oven clean|pressure wash|jet wash|end of tenancy)\b"),

    ("Removals, Delivery & Courier Services",
     r"\b(removal|moving service|mover|man and van|courier|delivery|transport service|airport transfer|chauffeur|taxi|private hire)\b"),

    ("Waste, Clearance & Recycling",
     r"\b(rubbish|junk removal|waste|skip hire|clearance|recycling)\b"),

    ("Appliance & Equipment Services",
     r"\b(appliance|washing machine|dishwasher|fridge repair|freezer repair|dryer repair|oven repair|cooker repair|generator repair|garden machinery|lawn mower repair|drone repair|bbq repair)\b"),

    ("Renewable Energy & Insulation",
     r"\b(solar|heat pump|biomass|renewable energy|insulation)\b"),

    ("Air Conditioning & Ventilation",
     r"\b(aircon|air conditioning|air conditioner|ventilation)\b"),

    ("Pest & Bird Control",
     r"\b(pest control|bird control|bird restriction|bed bug|rodent|wasp)\b"),

    ("Fire & Safety Services",
     r"\b(fire alarm|fire extinguisher|fire safety)\b"),

    ("Blinds, Shutters & Window Coverings",
     r"\b(blind|shutter|curtain fitting)\b"),

    ("Pool, Hot Tub & Water Feature Services",
     r"\b(swimming pool|hot tub|spa maintenance|water feature|aquarium maintenance)\b"),

    ("Metalwork & Fabrication",
     r"\b(blacksmith|metal worker|metal fabrication|steel fabrication|welding|bespoke metal)\b"),

    ("Damp, Waterproofing & Restoration",
     r"\b(damp|mould|dry rot|waterproof|flood restoration|water damage|restoration)\b"),

    ("Conservatory Services",
     r"\b(conservatory)\b"),

    ("Chimney & Fireplace Services",
     r"\b(chimney|fireplace)\b"),

    ("Tutoring & Lessons",
     r"\b(tutor|tuition|lessons|teacher|music lessons|language lessons)\b"),

    ("Tailoring & Clothing Alterations",
     r"\b(tailor|dressmaker|alteration|seamstress)\b"),

    ("Printing, Signage & Art Services",
     r"\b(printing|signage|sign maker|mural|artist|engraving)\b"),
]



# ============================================================
# FINAL COMPREHENSIVE PASS
# Adds broad but still meaningful families found in the
# remaining-unclassified file. Specific sub-services are kept.
# ============================================================

ADDITIONAL_SUBSERVICE_RULES = [

    # Handyman / assembly / mounting
    (r"\bemergency handyman\b", "Emergency Handyman"),
    (r"\bhandyman|handyperson\b", "General Handyman"),
    (r"\bflat packs?|flat pack assembly\b", "Flat Pack Assembly"),
    (r"\bikea assembly\b", "IKEA Assembly"),
    (r"\bfurniture assembly\b", "Furniture Assembly"),
    (r"\btrampoline assembly\b", "Trampoline Assembly"),
    (r"\bplayground assembly|climbing frame installer\b", "Play Equipment Assembly"),
    (r"\bwall hanging|picture hanging|mirror hanging|sign hanging\b", "Wall Hanging & Mounting"),

    # Property / landlord / estate
    (r"\blandlord inventories.*check in\b", "Tenant Check-In Inventory"),
    (r"\blandlord inventories.*check out\b", "Tenant Check-Out Inventory"),
    (r"\blandlord inventories.*mid tenancy\b", "Mid-Tenancy Inspection"),
    (r"\blandlord inventories\b", "Landlord Inventory"),
    (r"\blandlord reports|landlord safety checks\b", "Landlord Safety Checks"),
    (r"\bproperty maintenance\b", "Property Maintenance"),
    (r"\bblock management\b", "Block Management"),
    (r"\bflat management\b", "Flat Management"),
    (r"\bestate agents?|real estate\b", "Estate Agency"),

    # Interior / design
    (r"\bcommercial interior design\b", "Commercial Interior Design"),
    (r"\bhome interior designer|interior designer\b", "Interior Design"),
    (r"\bbedroom planner|bedroom designer\b", "Bedroom Design"),
    (r"\bspace planners?\b", "Space Planning"),
    (r"\bhome stagers?\b", "Home Staging"),
    (r"\boffice fit out\b", "Office Fit-Out"),
    (r"\bshop fitting\b", "Shop Fitting"),

    # Laundry / domestic
    (r"\bironing service|ironing\b", "Ironing"),
    (r"\blaundry service|commercial laundry|laundry\b", "Laundry Services"),
    (r"\bhousekeeper|housekeepers\b", "Housekeeping"),
    (r"\bdecluttering\b", "Decluttering"),
    (r"\bhome organiser|home organizer\b", "Home Organisation"),

    # Locksmith / keys / safes
    (r"\bkey cutting\b", "Key Cutting"),
    (r"\bcar key repair\b", "Car Key Repair"),
    (r"\bcar key replacement\b", "Car Key Replacement"),
    (r"\bsafe locks?|safes\b", "Safe Services"),

    # Groundworks / excavation / plant
    (r"\bgroundworks?|groundwork foundations\b", "Groundworks"),
    (r"\bearthmoving\b", "Earthmoving"),
    (r"\bbackhoeing\b", "Backhoe Services"),
    (r"\btrench digging\b", "Trench Digging"),
    (r"\bcore drilling\b", "Core Drilling"),
    (r"\bdiamond drilling\b", "Diamond Drilling"),
    (r"\bmini.*digger hire\b", "Mini Digger Hire"),
    (r"\bplant hire\b", "Plant Hire"),
    (r"\bcrane hire\b", "Crane Hire"),
    (r"\bgrab hire\b", "Grab Hire"),

    # Concrete / aggregates
    (r"\baggregate supplier\b", "Aggregate Supply"),
    (r"\bconcrete cutting\b", "Concrete Cutting"),
    (r"\bconcrete repair\b", "Concrete Repair"),
    (r"\bconcrete resurfacing\b", "Concrete Resurfacing"),
    (r"\bconcrete driveway\b", "Concrete Driveways"),
    (r"\bconcreting|concrete contractor\b", "Concrete Work"),

    # Scaffolding
    (r"\bemergency scaffolder\b", "Emergency Scaffolding"),
    (r"\bscaffolding hire\b", "Scaffolding Hire"),
    (r"\bscaffolder|scaffolding\b", "Scaffolding"),

    # Stone / masonry
    (r"\bstonemason|stonemasonry|stoneworker\b", "Stonemasonry"),
    (r"\bstone carving\b", "Stone Carving"),
    (r"\bstone repair\b", "Stone Repair"),
    (r"\bstone cladding\b", "Stone Cladding"),

    # Cladding / exterior
    (r"\bcladding install\b", "Cladding Installation"),
    (r"\bcladding repair\b", "Cladding Repair"),
    (r"\bcladding replacement\b", "Cladding Replacement"),
    (r"\bupvc cladding\b", "uPVC Cladding"),

    # Decking
    (r"\bcomposite decking\b", "Composite Decking"),
    (r"\bwooden decking\b", "Wooden Decking"),
    (r"\bdeck repair\b", "Deck Repair"),
    (r"\bdeck sealing\b", "Deck Sealing"),
    (r"\bdeck staining\b", "Deck Staining"),
    (r"\bdecking\b", "General Decking"),

    # Curtains / blinds / awnings
    (r"\bcurtain supply.*fit\b", "Curtain Supply & Fitting"),
    (r"\bcurtain fitter|curtain installation\b", "Curtain Fitting"),
    (r"\bcurtain pole|curtain track|curtain rod\b", "Curtain Track & Pole Installation"),
    (r"\bawning|canopy\b", "Awnings & Canopies"),

    # Telecoms
    (r"\btelephone engineer\b", "Telephone Engineering"),
    (r"\btelephone faults?\b", "Telephone Fault Repair"),
    (r"\btelephone points?\b", "Telephone Point Installation"),
    (r"\btelecommunications\b", "Telecommunications"),
    (r"\bvoip telephony\b", "VoIP Telephony"),
    (r"\bfixed lines?\b", "Fixed Line Services"),

    # Smart home / cabling
    (r"\bdata cable installation|network cabling\b", "Data & Network Cabling"),
    (r"\bdigital home network|home network setup|home networks\b", "Home Network Setup"),
    (r"\bintercom entry\b", "Intercom Entry Systems"),
    (r"\bsmart heating installation\b", "Smart Heating Installation"),

    # Small electronics
    (r"\bconsole repair\b", "Games Console Repair"),
    (r"\bheadphone repair\b", "Headphone Repair"),
    (r"\bhifi repair\b", "Hi-Fi Repair"),
    (r"\bipod repair\b", "iPod Repair"),
    (r"\blcd screen repair|screen repair\b", "Screen Repair"),
    (r"\bcircuit board repair\b", "Circuit Board Repair"),
    (r"\belectronic repair\b", "General Electronics Repair"),

    # Water treatment / tanks / pumps
    (r"\bwater softener\b", "Water Softener Services"),
    (r"\bwater filters?\b", "Water Filtration"),
    (r"\bwater purifier|purifiers\b", "Water Purification"),
    (r"\bwater tank\b", "Water Tank Services"),
    (r"\bwater pump\b", "Water Pump Services"),
    (r"\bwell pump\b", "Well Pump Repair"),
    (r"\bhousehold water treatment\b", "Household Water Treatment"),
    (r"\brainwater harvesting\b", "Rainwater Harvesting"),

    # Septic / sewage
    (r"\bseptic tank emptying\b", "Septic Tank Emptying"),
    (r"\bseptic tank installation\b", "Septic Tank Installation"),
    (r"\bseptic tank repair\b", "Septic Tank Repair"),
    (r"\bsewage pump installation\b", "Sewage Pump Installation"),
    (r"\bsewage pump maintenance\b", "Sewage Pump Maintenance"),

    # Legal
    (r"\bemployment law\b", "Employment Law"),
    (r"\bsolicitors?\b", "Solicitor Services"),
    (r"\blegal services\b", "General Legal Services"),

    # Training / driving
    (r"\bdriving instructors?\b", "Driving Lessons"),
    (r"\bfirst aid training\b", "First Aid Training"),
    (r"\bcpr training\b", "CPR Training"),
    (r"\bcyber security training\b", "Cyber Security Training"),
    (r"\bcycling training\b", "Cycling Training"),
    (r"\bdrone training\b", "Drone Training"),

    # Food / chef / baking
    (r"\bprivate chef\b", "Private Chef"),
    (r"\bhome chef\b", "Home Chef"),
    (r"\bmeal preparation|meal prep\b", "Meal Preparation"),
    (r"\bartisan baker\b", "Artisan Baking"),
    (r"\bvegan baker\b", "Vegan Baking"),
    (r"\bgluten free baker\b", "Gluten-Free Baking"),
    (r"\bwholesale baker\b", "Wholesale Baking"),
    (r"\bcustom.*cakes?|celebration cakes?\b", "Custom Cakes"),

    # Photography variants
    (r"\bfamily photoshoots?\b", "Family Photography"),
    (r"\bcouple photoshoots?\b", "Couple Photography"),
    (r"\bcorporate filming\b", "Corporate Filming"),
    (r"\bphoto restorations?\b", "Photo Restoration"),
    (r"\bphoto manipulations?\b", "Photo Editing"),

    # Events / entertainment
    (r"\bcomedians?\b", "Comedy Entertainment"),
    (r"\bclown\b", "Clown Entertainment"),
    (r"\bmagician\b", "Magician"),
    (r"\bjugglers?\b", "Juggling Entertainment"),
    (r"\bdancer\b", "Dance Entertainment"),
    (r"\bsinger\b", "Singer"),
    (r"\bmobile discos?\b", "Mobile Disco"),

    # Coaching
    (r"\blife coach\b", "Life Coaching"),
    (r"\bexecutive coach\b", "Executive Coaching"),
    (r"\brelationship coach\b", "Relationship Coaching"),
    (r"\bmotivational speaker\b", "Motivational Speaking"),

    # Survey / reports
    (r"\bhomebuyer report\b", "HomeBuyer Report"),
    (r"\brics condition report\b", "RICS Condition Report"),
    (r"\bland appraisal\b", "Land Appraisal"),

    # Specialist repairs / fabrics
    (r"\bleather repair\b", "Leather Repair"),
    (r"\bzipper repair\b", "Zipper Repair"),
    (r"\bclothing repair\b", "Clothing Repair"),
    (r"\bsewing machine repair\b", "Sewing Machine Repair"),
    (r"\btent repair\b", "Tent Repair"),

    # Surface / coatings
    (r"\banti vandal coatings?\b", "Anti-Vandal Coating"),
    (r"\bweather coatings?\b", "Weather Coating"),
    (r"\bpowder coating\b", "Powder Coating"),
    (r"\bpvc spraying|upvc spraying\b", "uPVC Spraying"),
    (r"\bsoda.*blasting|sand.*blasting|shot blasting\b", "Blasting Services"),

    # Specialist garden / tree
    (r"\bcrown reduction\b", "Crown Reduction"),
    (r"\bcrown thinning\b", "Crown Thinning"),
    (r"\btree felling\b", "Tree Felling"),
    (r"\btree pruning\b", "Tree Pruning"),
    (r"\btree planting\b", "Tree Planting"),
    (r"\bturf laying|turfing\b", "Turf Laying"),
    (r"\bturf repair\b", "Turf Repair"),
    (r"\bgrounds maintenance\b", "Grounds Maintenance"),

    # Signage / vehicle graphics
    (r"\bvehicle signwriting\b", "Vehicle Signwriting"),
    (r"\bvehicle wrapping|vehicle wraps\b", "Vehicle Wrapping"),
    (r"\bsign writers?|sign writing\b", "Sign Writing"),
    (r"\bsign makers?\b", "Sign Making"),

    # Hair / barber catch-all refinements
    (r"\bafro barbers?\b", "Afro Barbering"),
    (r"\bmobile barbers?\b", "Mobile Barbering"),
    (r"\bcurly hair specialists?\b", "Curly Hair Services"),
    (r"\bhair braiding\b", "Hair Braiding"),
    (r"\bhaircut and trimming\b", "Haircut & Trimming"),
    (r"\bkids haircut\b", "Kids Haircuts"),
    (r"\bfade buzz\b", "Fade / Buzz Cuts"),
]


ADDITIONAL_FAMILY_RULES = [

    ("Handyman, Assembly & Mounting",
     r"\b(handyman|handyperson|assembly|flat pack|ikea|mounting|picture hanging|mirror hanging|shelf hanging|wall hanging|climbing frame|playground assembly|trampoline assembly)\b"),

    ("Property, Landlord & Estate Services",
     r"\b(property maintenance|block management|flat management|landlord|estate agent|real estate|homebuyer report|rics condition report|land appraisal)\b"),

    ("Interior Design & Fit-Out",
     r"\b(interior design|interior designer|interior architects?|space planner|home stager|bedroom planner|bedroom designer|office fit out|shop fitting|restaurant designers?|coffee shop designers?)\b"),

    ("Laundry, Housekeeping & Organisation",
     r"\b(laundry|ironing|housekeeper|decluttering|home organiser|home organizer|janitorial)\b"),

    ("Locksmith, Keys & Safes",
     r"\b(locksmith|lock fitting|lock installation|key cutting|car key|safe locks?|safes|snap locks?)\b"),

    ("Groundworks, Excavation & Plant Hire",
     r"\b(groundwork|earthmoving|backhoeing|excavation|trench digging|core drilling|diamond drilling|digger hire|plant hire|crane hire|grab hire|moling)\b"),

    ("Concrete & Aggregate Services",
     r"\b(aggregate supplier|concrete|concreting|polished concrete|patterned imprinted concrete)\b"),

    ("Scaffolding Services",
     r"\b(scaffolder|scaffolding)\b"),

    ("Stone & Masonry Services",
     r"\b(stonemason|stonemasonry|stoneworker|stone carving|stone repair|stone cladding|flint stonework|dry stone wall)\b"),

    ("Cladding & Exterior Finishes",
     r"\b(cladding|weather coating|anti vandal coating|upvc spraying|pvc spraying)\b"),

    ("Decking Services",
     r"\b(deck repair|deck sealing|deck staining|decking|composite decking|wooden decking)\b"),

    ("Curtains, Awnings & Window Dressings",
     r"\b(curtain|awning|canopy|sail shade)\b"),

    ("Telecoms & Telephone Services",
     r"\b(telecommunications|telephone engineer|telephone fault|telephone point|telephone system|voip|fixed lines)\b"),

    ("Smart Home, Network & Cabling",
     r"\b(data cable|network cabling|home network|digital home network|intercom|smart heating|smart thermostat|home theatre)\b"),

    ("Electronics Repair Services",
     r"\b(console repair|headphone repair|hifi repair|ipod repair|screen repair|circuit board repair|electronic repair|lamp repair|clock repair|video audio repair|projector installation|sound system installation)\b"),

    ("Water Treatment, Tanks & Pumps",
     r"\b(water softener|water filter|water purifier|water tank|water pump|well pump|household water treatment|rainwater harvesting|water cooler|water meter)\b"),

    ("Septic, Sewage & Drainage Infrastructure",
     r"\b(septic tank|sewage pump|soakaway)\b"),

    ("Legal Services",
     r"\b(employment law|solicitor|legal services)\b"),

    ("Training & Instruction",
     r"\b(driving instructor|first aid training|cpr training|cyber security training|cycling training|drone training|self defence classes|trailer towing training)\b"),

    ("Food, Baking & Personal Chef Services",
     r"\b(private chef|home chef|meal preparation|meal prep|baker|baking|custom cake|celebration cake|cupcakes?|pastry chef|cooking)\b"),

    ("Coaching & Personal Development",
     r"\b(life coach|executive coach|relationship coach|motivational speaker|coaching)\b"),

    ("Fabric, Clothing & Specialist Repair",
     r"\b(leather repair|zipper repair|clothing repair|sewing machine repair|costume maker|pattern cutter|embroidery services|tent repair)\b"),

    ("Surface Treatment & Specialist Coatings",
     r"\b(powder coating|anti vandal coating|weather coating|sand blasting|shot blasting|soda blasting|surface repair|stain protection)\b"),

    ("Tree & Grounds Services",
     r"\b(crown reduction|crown thinning|tree felling|tree pruning|tree planting|tree lopper|tree surgeons?|tree surgery|grounds maintenance|mulching|wood chipping)\b"),

    ("Vehicle Graphics & Signage",
     r"\b(vehicle signwriting|vehicle wrapping|vehicle wraps|sign writing|sign writers?|sign makers?)\b"),

    ("Hair & Barber Services",
     r"\b(afro barbers?|mobile barbers?|curly hair specialists?|hair braiding|haircut and trimming|kids haircut|fade buzz|barbers?)\b"),

    ("Marine & Boat Services",
     r"\b(boat repairs?|boatbuilders?|boat transport)\b"),

    ("Commercial & Business Property Services",
     r"\b(commercial agents?|commercial interior design|warehouse racking|portable offices?)\b"),

    ("Hire & Access Equipment",
     r"\b(cherry picker|hire services?|tool hire|mini bus hire)\b"),

    ("Landscaping Structures & Outdoor Buildings",
     r"\b(greenhouse|log cabin|summer house|summerhouse|outbuilding|tree house|polytunnel|carport|play areas?)\b"),

    ("Home Improvement & General Maintenance",
     r"\b(home improvements?|home maintenance|home services|property maintenance repair|general labour)\b"),
]



# ============================================================
# LAST LAP
# Final targeted mappings for obvious genuine services that
# were still falling into Other Services / Unclassified.
#
# Rule: keep meaningful distinctions at sub_service level,
# while grouping only genuinely related jobs into families.
# ============================================================

LAST_LAP_SUBSERVICE_RULES = [

    # Accounting / finance
    (r"\baccountants?\b", "Accounting"),
    (r"\bbookkeeping services?\b", "Bookkeeping"),
    (r"\bpayroll services?\b", "Payroll Services"),
    (r"\btax returns?|tax preparation\b", "Tax Preparation"),

    # Transport
    (r"\bairport transfers?\b", "Airport Transfers"),
    (r"\bminibus hire\b", "Minibus Hire"),
    (r"\bchauffeur services?\b", "Chauffeur Services"),

    # Security
    (r"\balarms? / security|alarms? and security\b", "General Alarm & Security Services"),
    (r"\bsecurity alarms?\b", "Security Alarm Services"),

    # Vehicle
    (r"\balloy wheels? repairs?\b", "Alloy Wheel Repair"),
    (r"\bcar diagnostics?\b", "Car Diagnostics"),
    (r"\bcar body ?work\b", "Car Bodywork"),
    (r"\bcar bumper repair\b", "Bumper Repair"),
    (r"\bcar respray|vehicle respray\b", "Vehicle Respraying"),
    (r"\bcar scratch repair\b", "Scratch Repair"),
    (r"\bcar audio installation\b", "Car Audio Installation"),
    (r"\bcar wrapping\b", "Vehicle Wrapping"),
    (r"\bbike repairs?|bicycle repairs?\b", "Bicycle Repair"),
    (r"\bmotorcycle servicing\b", "Motorcycle Servicing"),

    # Architecture / planning
    (r"^architects?$", "Architectural Services"),
    (r"\barchitects?, surveyors? and planners?\b", "Architecture, Surveying & Planning"),
    (r"\bplanning applications?\b", "Planning Applications"),
    (r"\bplanning permission\b", "Planning Permission Support"),

    # Personal training / fitness
    (r"\bat home personal training\b", "At-Home Personal Training"),
    (r"\bmobile personal trainer\b", "Mobile Personal Training"),

    # Awnings / canopies
    (r"\bawnings? / canopies?\b", "Awnings & Canopies"),

    # Food / baking
    (r"^bakers?$", "General Baking"),
    (r"\bcake makers?\b", "Cake Making"),
    (r"\bcupcake makers?\b", "Cupcake Making"),

    # Conversions
    (r"\bbarn conversions?\b", "Barn Conversion"),
    (r"\bbasement / cellar conversions?\b", "Basement / Cellar Conversion"),
    (r"\bcellar conversions?\b", "Cellar Conversion"),
    (r"\battic conversions?\b", "Attic Conversion"),

    # Beauty
    (r"^beauticians?$", "Beautician Services"),
    (r"\bbrows? and lashes\b", "Brows & Lashes"),
    (r"\beyebrow threading\b", "Eyebrow Threading"),
    (r"\blash extensions?\b", "Lash Extensions"),

    # Metalwork
    (r"^blacksmithing$", "Blacksmithing"),

    # Plumbing blockages
    (r"\bblocked baths?\b", "Blocked Bath"),
    (r"\bblocked sinks?\b", "Blocked Sink"),
    (r"\bblocked toilets?\b", "Blocked Toilet"),
    (r"\bblocked showers?\b", "Blocked Shower"),

    # Furniture / bespoke wood
    (r"\bbespoke alcove units?\b", "Bespoke Alcove Units"),
    (r"\bbespoke cupboards?\b", "Bespoke Cupboards"),
    (r"\bbespoke dining tables?\b", "Bespoke Dining Tables"),
    (r"\bbespoke doors?\b", "Bespoke Doors"),
    (r"\bbespoke under stairs storage\b", "Bespoke Under-Stairs Storage"),
    (r"\bbespoke timber windows?\b", "Bespoke Timber Windows"),
    (r"\bbespoke shelving\b", "Bespoke Shelving"),
    (r"\bbespoke storage\b", "Bespoke Storage"),

    # Carpets / flooring
    (r"\bcarpet fitters?\b", "Carpet Fitting"),
    (r"\bcarpet suppliers?\b", "Carpet Supply"),
    (r"\bcarpet cleaning\b", "Carpet Cleaning"),

    # Attic / loft services
    (r"\battic insulation\b", "Attic Insulation"),
    (r"\battic ladders?\b", "Attic Ladder Installation"),
    (r"\battic flooring\b", "Attic Flooring"),

    # Bedrooms / fitted interiors
    (r"\bbedroom fitters?\b", "Bedroom Fitting"),
    (r"\bfitted bedrooms?\b", "Fitted Bedrooms"),

    # General home repair
    (r"\bhome repairs?\b", "General Home Repair"),
    (r"\bproperty repairs?\b", "Property Repair"),

    # Cleaning variants
    (r"\bafter builders clean\b", "After Builders Cleaning"),
    (r"\bbuilders clean\b", "Builders Cleaning"),
    (r"\bmove in cleaning\b", "Move-In Cleaning"),
    (r"\bmove out cleaning\b", "Move-Out Cleaning"),

    # Locksmith variants
    (r"\bdoor lock repair\b", "Door Lock Repair"),
    (r"\bdoor lock replacement\b", "Door Lock Replacement"),

    # Doors / windows
    (r"\bwindow fitters?\b", "Window Fitting"),
    (r"\bdoor fitters?\b", "Door Fitting"),
    (r"\bwindow restoration\b", "Window Restoration"),
    (r"\bwooden window repair\b", "Wooden Window Repair"),

    # Roofing
    (r"\bemergency roofing\b", "Emergency Roofing"),
    (r"\broof leak repair\b", "Roof Leak Repair"),
    (r"\bchimney flashing\b", "Chimney Flashing"),

    # Painting / decorating
    (r"\bcommercial painters?\b", "Commercial Painting"),
    (r"\bhouse painters?\b", "House Painting"),
    (r"\bspray painting\b", "Spray Painting"),

    # Landscaping
    (r"\bgarden clearance\b", "Garden Clearance"),
    (r"\bgarden fencing\b", "Garden Fencing"),
    (r"\bgarden decking\b", "Garden Decking"),

    # Waste
    (r"\bbuilders waste\b", "Builders Waste Removal"),
    (r"\bconstruction waste\b", "Construction Waste Removal"),

    # Appliance
    (r"\bfridge freezer repair\b", "Fridge-Freezer Repair"),
    (r"\btumble dryer repair\b", "Tumble Dryer Repair"),

    # Media / creative
    (r"\bvoice over artists?\b", "Voice-Over Services"),
    (r"\bvideo editing\b", "Video Editing"),
    (r"\bgraphic designers?\b", "Graphic Design"),

    # Language
    (r"\binterpreters?\b", "Interpreting"),
    (r"\blanguage translation\b", "Translation"),

    # Lessons
    (r"\bguitar lessons?\b", "Guitar Lessons"),
    (r"\bpiano lessons?\b", "Piano Lessons"),
    (r"\bsinging lessons?\b", "Singing Lessons"),

    # Pet services
    (r"\bdog groomers?\b", "Dog Grooming"),
    (r"\bcat groomers?\b", "Cat Grooming"),
    (r"\bpet taxi\b", "Pet Taxi"),

    # Misc genuine services
    (r"\bshoe repair\b", "Shoe Repair"),
    (r"\bwatch repair\b", "Watch Repair"),
    (r"\bjewellery repair|jewelry repair\b", "Jewellery Repair"),
    (r"\bclock repair\b", "Clock Repair"),
]


LAST_LAP_FAMILY_RULES = [

    ("Accounting & Finance Services",
     r"\b(accountants?|bookkeeping services?|payroll services?|tax returns?|tax preparation)\b"),

    ("Transport & Transfer Services",
     r"\b(airport transfers?|minibus hire|chauffeur services?)\b"),

    ("Security & Alarm Services",
     r"\b(alarms? / security|alarms? and security|security alarms?)\b"),

    ("Vehicle Services",
     r"\b(alloy wheels?|car diagnostics?|car body ?work|car bumper|car respray|vehicle respray|car scratch|car audio|car wrapping|bike repairs?|bicycle repairs?|motorcycle servicing)\b"),

    ("Architecture, Surveying & Planning",
     r"\b(architects?|architects?, surveyors? and planners?|planning applications?|planning permission)\b"),

    ("Fitness & Sports Coaching",
     r"\b(at home personal training|mobile personal trainer)\b"),

    ("Curtains, Awnings & Window Dressings",
     r"\b(awnings? / canopies?)\b"),

    ("Food, Baking & Personal Chef Services",
     r"\b(bakers?|cake makers?|cupcake makers?)\b"),

    ("Building & Construction",
     r"\b(barn conversions?|basement / cellar conversions?|cellar conversions?|attic conversions?)\b"),

    ("Beauty & Personal Care",
     r"\b(beauticians?|brows? and lashes|eyebrow threading|lash extensions?)\b"),

    ("Metalwork & Fabrication",
     r"\bblacksmithing\b"),

    ("Plumbing & Drainage Services",
     r"\b(blocked baths?|blocked sinks?|blocked toilets?|blocked showers?)\b"),

    ("Furniture, Carpentry & Woodwork",
     r"\b(bespoke alcove|bespoke cupboards?|bespoke dining tables?|bespoke doors?|bespoke under stairs|bespoke timber windows?|bespoke shelving|bespoke storage)\b"),

    ("Flooring & Tiling Services",
     r"\b(carpet fitters?|carpet suppliers?|carpet cleaning)\b"),

    ("Loft, Attic & Storage Services",
     r"\b(attic insulation|attic ladders?|attic flooring)\b"),

    ("Interior Design & Fit-Out",
     r"\b(bedroom fitters?|fitted bedrooms?)\b"),

    ("Home Improvement & General Maintenance",
     r"\b(home repairs?|property repairs?)\b"),

    ("Cleaning Services",
     r"\b(after builders clean|builders clean|move in cleaning|move out cleaning)\b"),

    ("Locksmith, Keys & Safes",
     r"\b(door lock repair|door lock replacement)\b"),

    ("Doors, Windows & Glazing",
     r"\b(window fitters?|door fitters?|window restoration|wooden window repair)\b"),

    ("Roofing Services",
     r"\b(emergency roofing|roof leak repair|chimney flashing)\b"),

    ("Painting, Decorating & Wall Finishes",
     r"\b(commercial painters?|house painters?|spray painting)\b"),

    ("Gardening & Landscaping",
     r"\b(garden clearance|garden fencing|garden decking)\b"),

    ("Waste, Clearance & Recycling",
     r"\b(builders waste|construction waste)\b"),

    ("Appliance & Equipment Services",
     r"\b(fridge freezer repair|tumble dryer repair)\b"),

    ("Marketing & Creative Services",
     r"\b(voice over artists?|video editing|graphic designers?)\b"),

    ("Writing, Translation & Language Services",
     r"\b(interpreters?|language translation)\b"),

    ("Tutoring & Lessons",
     r"\b(guitar lessons?|piano lessons?|singing lessons?)\b"),

    ("Pet & Animal Services",
     r"\b(dog groomers?|cat groomers?|pet taxi)\b"),

    ("Specialist Repair Services",
     r"\b(shoe repair|watch repair|jewellery repair|jewelry repair|clock repair)\b"),
]



# ============================================================
# RESIDUAL RESOLUTION PASS
# Built directly from the latest residual export.
# These rules cover obvious remaining service clusters without
# collapsing technically different services.
# ============================================================

RESIDUAL_SUBSERVICE_RULES = [

    # General platform / service buckets
    (r"^appliances?$", "General Appliance Services"),
    (r"^bathrooms?$", "General Bathroom Services"),
    (r"^bedrooms?$", "General Bedroom Services"),
    (r"^kitchens?$", "General Kitchen Services"),
    (r"^events?$", "General Event Services"),
    (r"^transport$", "General Transport Services"),
    (r"^writing$", "General Writing Services"),
    (r"^storage$", "General Storage Services"),

    # Roofing / gutter / exterior
    (r"\baluminium gutters?\b", "Aluminium Guttering"),
    (r"\bpvc gutters?\b", "PVC Guttering"),
    (r"\bdownpipe installation.*repair\b", "Downpipe Installation & Repair"),
    (r"\bfelt lap vents?\b", "Felt Lap Vents"),
    (r"\bleadworks? installation\b", "Leadwork Installation"),
    (r"\bleadworks? repair\b", "Leadwork Repair"),
    (r"\bdry ridge installation\b", "Dry Ridge Installation"),
    (r"\bdry ridge repair\b", "Dry Ridge Repair"),
    (r"\bdry verge installation\b", "Dry Verge Installation"),
    (r"\bdry verge repair\b", "Dry Verge Repair"),
    (r"\bsingle ply membrane roofs? installation\b", "Single Ply Membrane Roof Installation"),
    (r"\bsingle ply membrane roofs? repair\b", "Single Ply Membrane Roof Repair"),
    (r"\bskylight install|skylights? for flat roofs?\b", "Skylight Installation"),

    # Doors / windows / glazing
    (r"\bbespoke external doors?\b", "Bespoke External Doors"),
    (r"\bbespoke interior doors?\b", "Bespoke Interior Doors"),
    (r"\bbespoke oak doors?\b", "Bespoke Oak Doors"),
    (r"\bbespoke stable doors?\b", "Bespoke Stable Doors"),
    (r"\bbespoke wooden front doors?\b", "Bespoke Wooden Front Doors"),
    (r"\bbespoke wooden windows?\b", "Bespoke Wooden Windows"),
    (r"\bfix aluminium windows?\b", "Aluminium Window Repair"),
    (r"\bfix bay windows?\b", "Bay Window Repair"),
    (r"\bfix loft windows?\b", "Loft Window Repair"),
    (r"\bfix timber.*windows?\b", "Timber Window Repair"),
    (r"\bfix upvc windows?\b", "uPVC Window Repair"),
    (r"\bfix wooden sash windows?\b", "Wooden Sash Window Repair"),
    (r"\bfrosted windows?\b", "Frosted Windows"),
    (r"\bglass cutting\b", "Glass Cutting"),
    (r"\bglass polishing\b", "Glass Polishing"),
    (r"\bglass replacement\b", "Glass Replacement"),
    (r"\bglass splashbacks?\b", "Glass Splashbacks"),
    (r"\bmirror cutting\b", "Mirror Cutting"),
    (r"\bmirror repair\b", "Mirror Repair"),
    (r"\bwindows? supply\b", "Window Supply"),
    (r"\bupvc repairs?\b", "uPVC Repair"),
    (r"\bwindows? and doors?|windows? & doors?\b", "Windows & Doors"),
    (r"\bweather stripping\b", "Weather Stripping"),

    # Carpentry / woodwork / fitted furniture
    (r"\barchitraves?\b", "Architraves"),
    (r"\bbarn doors?\b", "Barn Doors"),
    (r"\bcupboards?\b", "Cupboards"),
    (r"\bcustom shelving\b", "Custom Shelving"),
    (r"\bskirting board installation\b", "Skirting Board Installation"),
    (r"\bskirting boards?\b", "Skirting Boards"),
    (r"\bstairs made to measure\b", "Made-to-Measure Stairs"),
    (r"\bunder stairs storage fitters?\b", "Under-Stairs Storage Fitting"),
    (r"\bwood repair\b", "Wood Repair"),
    (r"\bwood turning\b", "Wood Turning"),
    (r"\bwooden / timber structures?\b", "Timber Structures"),

    # Kitchen / worktops
    (r"\bbespoke kitchens?\b", "Bespoke Kitchens"),
    (r"\bfitted kitchens?\b", "Fitted Kitchens"),
    (r"\bhandmade kitchens?\b", "Handmade Kitchens"),
    (r"\bworktops? - acrylic\b", "Acrylic Worktops"),
    (r"\bworktops? - glass\b", "Glass Worktops"),
    (r"\bworktops? - granite\b", "Granite Worktops"),
    (r"\bworktops? - marble\b", "Marble Worktops"),
    (r"\bworktops? - quartz\b", "Quartz Worktops"),
    (r"\bworktops? - solid wood\b", "Solid Wood Worktops"),
    (r"\bworktops? - stainless steel\b", "Stainless Steel Worktops"),
    (r"^worktops?$", "General Worktop Services"),

    # Plumbing / water / hot water
    (r"\bdripping tap\b", "Dripping Tap Repair"),
    (r"\blead pipe\b", "Lead Pipe Services"),
    (r"\bpipe fitter\b", "Pipe Fitting"),
    (r"\bsink installation\b", "Sink Installation"),
    (r"\bsink crack\b", "Sink Repair"),
    (r"\bsink draining\b", "Sink Drainage"),
    (r"\bsoil stack replacement\b", "Soil Stack Replacement"),
    (r"\bwater leakage repair\b", "Water Leak Repair"),
    (r"\bwater pipe replacement\b", "Water Pipe Replacement"),
    (r"\bwater pressure\b", "Water Pressure Services"),
    (r"\bhot water system installation.*repair\b", "Hot Water System Installation & Repair"),
    (r"\bunvented hot water cylinder installation\b", "Unvented Cylinder Installation"),
    (r"\bunvented hot water cylinder servicing.*repair\b", "Unvented Cylinder Servicing & Repair"),
    (r"\bpower flushing\b", "Power Flushing"),
    (r"\bpower showers? and pump\b", "Power Shower & Pump Services"),

    # Heating / gas / fireplaces
    (r"\belectric heating\b", "Electric Heating"),
    (r"\belectric radiators?\b", "Electric Radiators"),
    (r"\belectric underfloor heating\b", "Electric Underfloor Heating"),
    (r"\bwater underfloor heating\b", "Water Underfloor Heating"),
    (r"\bufh installation\b", "Underfloor Heating Installation"),
    (r"\bufh repair\b", "Underfloor Heating Repair"),
    (r"\bunderfloor heating maintenance\b", "Underfloor Heating Maintenance"),
    (r"\bunderfloor heating servicing\b", "Underfloor Heating Servicing"),
    (r"\bunderfloor heating suppliers?\b", "Underfloor Heating Supply"),
    (r"\blpg engineer\b", "LPG Engineer"),
    (r"\bgas safety checks? - cp12\b", "Gas Safety Check / CP12"),
    (r"\bgas stove installation.*repair\b", "Gas Stove Installation & Repair"),
    (r"\bflue specialist\b", "Flue Services"),
    (r"\bfireplaces? / stoves?\b", "Fireplace & Stove Services"),
    (r"\bstoves? installer\b", "Stove Installation"),
    (r"\bwood / log burning stoves?\b", "Wood / Log Burning Stove Services"),
    (r"\bwood burner flue installers?\b", "Wood Burner Flue Installation"),

    # Electrical
    (r"\bcar / auto electrics\b", "Auto Electrical Services"),
    (r"\bchandelier installation\b", "Chandelier Installation"),
    (r"\bdownlights? installation\b", "Downlight Installation"),
    (r"\belectric cookers?\b", "Electric Cooker Services"),
    (r"\belectric oven / hob - installation\b", "Electric Oven & Hob Installation"),
    (r"\belectric showers?\b", "Electric Shower Services"),
    (r"\belectric smart thermostats?\b", "Smart Thermostat Installation"),
    (r"\belectric sockets?\b", "Socket Services"),
    (r"\belectric stoves?\b", "Electric Stove Services"),
    (r"\belectric vehicle charger installation\b", "EV Charger Installation"),
    (r"\bexternal lighting\b", "External Lighting"),
    (r"\bextractor fan installation\b", "Extractor Fan Installation"),
    (r"\bextractor fans?\b", "Extractor Fan Services"),
    (r"\bfault finding\b", "Electrical Fault Finding"),
    (r"\binternal lighting\b", "Internal Lighting"),
    (r"\bled lighting|led lights? installation\b", "LED Lighting Installation"),
    (r"\blighting design\b", "Lighting Design"),
    (r"\bpower point installation\b", "Power Point Installation"),
    (r"\bsockets? and switches? installation\b", "Socket & Switch Installation"),

    # Building / plaster / render / drywall
    (r"\bartexing\b", "Artexing"),
    (r"\bcoving\b", "Coving"),
    (r"\bcoving installers?\b", "Coving Installation"),
    (r"\bcornice repair\b", "Cornice Repair"),
    (r"\bdecorative cornicing / plasterwork\b", "Decorative Cornicing & Plasterwork"),
    (r"\bdrywall installers?\b", "Drywall Installation"),
    (r"\bdrywall repair\b", "Drywall Repair"),
    (r"\bdrywall replacement\b", "Drywall Replacement"),
    (r"\bdrywall taping\b", "Drywall Taping"),
    (r"\bexternal plasterers?\b", "External Plastering"),
    (r"\bflat ceilings?\b", "Flat Ceilings"),
    (r"\bk rend installers?\b", "K Rend Installation"),
    (r"\bk rend specialists?\b", "K Rend Services"),
    (r"\blime plasterer\b", "Lime Plastering"),
    (r"\blime render specialist\b", "Lime Rendering"),
    (r"\bmetal stud\b", "Metal Stud Partitioning"),
    (r"\bmonocouche render\b", "Monocouche Rendering"),
    (r"\bpartition walls?\b", "Partition Walls"),
    (r"\bpebble dash repair\b", "Pebble Dash Repair"),
    (r"\bpebble dashing\b", "Pebble Dashing"),
    (r"\bplasterboard installation\b", "Plasterboard Installation"),
    (r"\bplasterer / renderer\b", "Plastering & Rendering"),
    (r"\bskim coating contractor\b", "Skim Coating"),
    (r"\bskimming\b", "Skimming"),
    (r"\bstud partition\b", "Stud Partition"),
    (r"\bstud wall\b", "Stud Wall"),
    (r"\bsuspended ceilings?\b", "Suspended Ceilings"),
    (r"\btape and jointing\b", "Tape & Jointing"),

    # Floors / tiling
    (r"\bcarpet laying.*installation\b", "Carpet Installation"),
    (r"\bcarpet supply.*fit\b", "Carpet Supply & Fit"),
    (r"\bcarpet tiles?\b", "Carpet Tiles"),
    (r"\bliquid screeding\b", "Liquid Screeding"),
    (r"\bmosaic tiles?\b", "Mosaic Tiling"),
    (r"\bnatural stone tiles?\b", "Natural Stone Tiling"),
    (r"\bporcelain tiles?\b", "Porcelain Tiling"),
    (r"\bregrouting\b", "Regrouting"),
    (r"\bslate tiles?\b", "Slate Tiling"),
    (r"\btile.*repair\b", "Tile Repair"),
    (r"\btile supplier\b", "Tile Supply"),
    (r"\bvictorian tiles?\b", "Victorian Tiling"),

    # Garden / landscaping
    (r"\baviaries?\b", "Aviary Installation"),
    (r"\bchicken runs?\b", "Chicken Runs"),
    (r"\bequestrian stable\b", "Equestrian Stables"),
    (r"\bgolf / bowls greens?\b", "Golf & Bowls Green Services"),
    (r"\bgritting\b", "Gritting"),
    (r"\bhedging\b", "Hedging"),
    (r"\bland clearing\b", "Land Clearing"),
    (r"\blandscape contractor\b", "Landscape Contracting"),
    (r"\blandscape designers?\b", "Landscape Design"),
    (r"\blandscape lighting\b", "Landscape Lighting"),
    (r"\bplanting\b", "Planting"),
    (r"\bpruning\b", "Pruning"),
    (r"\bsensory gardens?\b", "Sensory Garden Services"),
    (r"\bshrub.*trimming\b", "Shrub & Bush Trimming"),
    (r"\bwater gardens?\b", "Water Garden Services"),
    (r"\byard work\b", "Yard Work"),

    # Vehicle / transport
    (r"\bcambelt change|cambelts?\b", "Cambelt Replacement"),
    (r"\bcar alarms?\b", "Car Alarm Services"),
    (r"\bcar cd player installation\b", "Car CD Player Installation"),
    (r"\bcar inspection\b", "Car Inspection"),
    (r"\bcar keyless entry system installation\b", "Keyless Entry Installation"),
    (r"\bcar seat repair\b", "Car Seat Repair"),
    (r"\bcar speaker installation\b", "Car Speaker Installation"),
    (r"\bcar stereo installation\b", "Car Stereo Installation"),
    (r"\bcar upholstery repair\b", "Car Upholstery Repair"),
    (r"\bcar wash\b", "Car Wash"),
    (r"\bcaravan transport\b", "Caravan Transport"),
    (r"\bcaravan valeting\b", "Caravan Valeting"),
    (r"\bcarburetor tuning\b", "Carburetor Tuning"),
    (r"\bcentral locking installation\b", "Central Locking Installation"),
    (r"\bclutch replacement|clutches?\b", "Clutch Replacement"),
    (r"\bdash cam installation\b", "Dash Cam Installation"),
    (r"\bdent repair\b", "Dent Repair"),
    (r"\bdiagnostic testing\b", "Vehicle Diagnostic Testing"),
    (r"\bengine rebuild\b", "Engine Rebuild"),
    (r"\bengine replacement\b", "Engine Replacement"),
    (r"\bfan belt replacement\b", "Fan Belt Replacement"),
    (r"\bfuel injection repair\b", "Fuel Injection Repair"),
    (r"\bfuel pump replacement\b", "Fuel Pump Replacement"),
    (r"\bgearbox repair|gearbox\b", "Gearbox Repair"),
    (r"\bglow plug replacement\b", "Glow Plug Replacement"),
    (r"\bhead gasket repair\b", "Head Gasket Repair"),
    (r"\bheadlight bulb replacement\b", "Headlight Bulb Replacement"),
    (r"\bignition systems? repair\b", "Ignition System Repair"),
    (r"\bmobile car airconditioning service\b", "Mobile Car Air Conditioning"),
    (r"\bmobile mechanics?\b", "Mobile Mechanic"),
    (r"\bmoped repair\b", "Moped Repair"),
    (r"\bmot testing|vehicle mot|motorcycle mots?|scooter mots?|van mot\b", "MOT Testing"),
    (r"\bo2 sensor replacement\b", "O2 Sensor Replacement"),
    (r"\boil & filter change\b", "Oil & Filter Change"),
    (r"\bpanel beater\b", "Panel Beating"),
    (r"\bparking sensor replacement\b", "Parking Sensor Replacement"),
    (r"\bperformance tuning\b", "Performance Tuning"),
    (r"\bpre purchase car inspection\b", "Pre-Purchase Car Inspection"),
    (r"\breversing camera installation\b", "Reversing Camera Installation"),
    (r"\bscratch repair\b", "Scratch Repair"),
    (r"\bsmart repairs?\b", "SMART Repair"),
    (r"\bsteering wheel repair\b", "Steering Wheel Repair"),
    (r"\bsuspension repair\b", "Suspension Repair"),
    (r"\btiming belt replacement\b", "Timing Belt Replacement"),
    (r"\btow bar installation\b", "Tow Bar Installation"),
    (r"\btowing services?|towing / transportation\b", "Towing Services"),
    (r"\bvehicle recovery\b", "Vehicle Recovery"),
    (r"\bvehicle transport\b", "Vehicle Transport"),
    (r"\bwheel alignment\b", "Wheel Alignment"),
    (r"\bwheel refurbishments?\b", "Wheel Refurbishment"),
    (r"\bwheel tracking\b", "Wheel Tracking"),
    (r"\bwindscreen repair\b", "Windscreen Repair"),
    (r"\bwindscreen replacement\b", "Windscreen Replacement"),

    # IT / digital / software
    (r"\bcomputers? & it\b", "General IT Services"),
    (r"\bemail setup\b", "Email Setup"),
    (r"\bicloud setup\b", "iCloud Setup"),
    (r"\binternet help\b", "Internet Help"),
    (r"\binternet services?\b", "Internet Services"),
    (r"\bit and consultancy\b", "IT Consultancy"),
    (r"\bmacbook air repair|macbook repairs?\b", "MacBook Repair"),
    (r"\bmicrosoft excel help\b", "Microsoft Excel Help"),
    (r"\bmicrosoft help\b", "Microsoft Help"),
    (r"\bmicrosoft powerpoint help\b", "Microsoft PowerPoint Help"),
    (r"\bmicrosoft windows help\b", "Microsoft Windows Help"),
    (r"\bmicrosoft word help\b", "Microsoft Word Help"),
    (r"\bmobile app\b", "Mobile App Development"),
    (r"\bmobile phone repairs?\b", "Mobile Phone Repair"),
    (r"\bnetflix setup\b", "Netflix Setup"),
    (r"\bnetwork installation\b", "Network Installation"),
    (r"\bpc repairs?\b", "PC Repair"),
    (r"\bprinter help\b", "Printer Help"),
    (r"\bprinter repairs?\b", "Printer Repair"),
    (r"\bscanner setup\b", "Scanner Setup"),
    (r"\bsoftware help\b", "Software Help"),
    (r"\btroubleshooting\b", "Technical Troubleshooting"),
    (r"\bwebsite.*app testing\b", "Website & App Testing"),
    (r"\bwebsite designers?\b", "Website Design"),
    (r"\bwix help\b", "Wix Help"),
    (r"\bwoocommerce help\b", "WooCommerce Help"),

    # Writing / admin / business
    (r"\bcase study writing\b", "Case Study Writing"),
    (r"\bcontent creation\b", "Content Creation"),
    (r"\bcreative writer\b", "Creative Writing"),
    (r"\bdata entry\b", "Data Entry"),
    (r"\bdrafting\b", "Drafting"),
    (r"\bdraftsman\b", "Drafting Services"),
    (r"\bfinancial modelling\b", "Financial Modelling"),
    (r"\bfinancial planning\b", "Financial Planning"),
    (r"\bfinancial reporting\b", "Financial Reporting"),
    (r"\bghostwriting\b", "Ghostwriting"),
    (r"\bhr services?\b", "HR Services"),
    (r"\bjournalist\b", "Journalism"),
    (r"\blinkedin profile writing\b", "LinkedIn Profile Writing"),
    (r"\bmarket research\b", "Market Research"),
    (r"\bmortgages?\b", "Mortgage Services"),
    (r"\bmystery shopping\b", "Mystery Shopping"),
    (r"\bpension advisor\b", "Pension Advice"),
    (r"\bpersonal assistant\b", "Personal Assistant"),
    (r"\bpersonal concierge\b", "Personal Concierge"),
    (r"\bpersonal shopper\b", "Personal Shopper"),
    (r"\bpersonal stylist\b", "Personal Styling"),
    (r"\bpoetry writing\b", "Poetry Writing"),
    (r"\bpresentations? design\b", "Presentation Design"),
    (r"\bpress release writing\b", "Press Release Writing"),
    (r"\bproposal writing\b", "Proposal Writing"),
    (r"\breport writing\b", "Report Writing"),
    (r"\bsales assistant\b", "Sales Assistance"),
    (r"\bspeech writing\b", "Speech Writing"),
    (r"\btechnical drawing\b", "Technical Drawing"),
    (r"\btravel writer\b", "Travel Writing"),
    (r"\btypist\b", "Typing Services"),
    (r"\bwhite paper writer\b", "White Paper Writing"),

    # Health / care / wellness
    (r"\bchiropodists? and podiatrists?\b", "Podiatry"),
    (r"\bday nurseries?\b", "Day Nursery"),
    (r"\bhome.*domiciliary care\b", "Home & Domiciliary Care"),
    (r"\bhome care\b", "Home Care"),
    (r"\bhomeopathy\b", "Homeopathy"),
    (r"\bmaternity nurse\b", "Maternity Nurse"),
    (r"\bmental health services?\b", "Mental Health Services"),
    (r"\bprivate doctors?\b", "Private Doctor"),
    (r"\bpsychologist\b", "Psychology Services"),
    (r"\bwellness.*day spa\b", "Wellness & Day Spa"),

    # Fitness / sports
    (r"\bgym buddy\b", "Gym Buddy"),
    (r"\bhiit training\b", "HIIT Training"),
    (r"\bmma training\b", "MMA Training"),
    (r"\bsports coaches?\b", "Sports Coaching"),
    (r"\btriathlon training\b", "Triathlon Training"),
    (r"\bzumba classes?\b", "Zumba Classes"),

    # Pets / animals
    (r"\baquariums?\b", "Aquarium Services"),
    (r"\bfish tank repair\b", "Fish Tank Repair"),
    (r"\bpuppy training\b", "Puppy Training"),
    (r"\brabbit boarding\b", "Rabbit Boarding"),
    (r"\breptile boarding\b", "Reptile Boarding"),
    (r"\bvets?\b", "Veterinary Services"),
    (r"\bwildlife management\b", "Wildlife Management"),

    # Food / hospitality
    (r"\bcafes? & coffee shops?\b", "Cafe & Coffee Shop Services"),
    (r"\bchef\b", "Chef Services"),
    (r"\bflorists?\b", "Floristry"),
    (r"\bflower arrangements?\b", "Flower Arrangements"),
    (r"\bhotels?\b", "Hotel Services"),
    (r"\bpubs?\b", "Pub Services"),
    (r"\bsushi chef\b", "Sushi Chef"),
    (r"\btakeaway\b", "Takeaway Food Services"),

    # Security / fire
    (r"\bburglar repairs?\b", "Burglar Alarm Repair"),
    (r"\bcar alarms?\b", "Car Alarm Services"),
    (r"\bcarbon monoxide alarms?.*installation\b", "Carbon Monoxide Alarm Installation"),
    (r"\bevent security\b", "Event Security"),
    (r"\bfire extinguishers?\b", "Fire Extinguisher Services"),
    (r"\bfire risk assessment\b", "Fire Risk Assessment"),
    (r"\bintruder alarms?\b", "Intruder Alarm Services"),
    (r"\bscaffold alarms?\b", "Scaffold Alarm Systems"),
    (r"\bsecurity assessment\b", "Security Assessment"),
    (r"\bsecurity barriers?\b", "Security Barriers"),
    (r"\bsecurity guards? services?\b", "Security Guard Services"),
    (r"\bsecurity shutters? / grilles?\b", "Security Shutters & Grilles"),
    (r"\bsmoke alarms?\b", "Smoke Alarm Services"),

    # Misc repair / specialist
    (r"\bchainsaw service.*repair\b", "Chainsaw Service & Repair"),
    (r"\bcoffee machine repairs?\b", "Coffee Machine Repair"),
    (r"\bghd repair\b", "GHD Repair"),
    (r"\bice machine repair.*installation\b", "Ice Machine Repair & Installation"),
    (r"\blamp repairs?\b", "Lamp Repair"),
    (r"\bmicrowave repair\b", "Microwave Repair"),
    (r"\boven fan repair\b", "Oven Fan Repair"),
    (r"\bpool table recovering\b", "Pool Table Recovering"),
    (r"\brange cooker servicing.*repair\b", "Range Cooker Servicing & Repair"),
    (r"\brayburn servicing.*repair\b", "Rayburn Servicing & Repair"),
    (r"\bsawblade sharpening\b", "Sawblade Sharpening"),
    (r"\bsewing\b", "Sewing Services"),
    (r"\bsharpening\b", "Sharpening Services"),
    (r"\btoaster repair\b", "Toaster Repair"),
    (r"\btool repair\b", "Tool Repair"),
    (r"\btool sharpening\b", "Tool Sharpening"),
    (r"\btreadmill repair\b", "Treadmill Repair"),
    (r"\btrampoline repair\b", "Trampoline Repair"),
    (r"\bvacuum service.*repair\b", "Vacuum Service & Repair"),

    # Specialist building / structural
    (r"\benergy performance certificate\b", "Energy Performance Certificate"),
    (r"\bfoundations?\b", "Foundation Work"),
    (r"\bgarage conversions?\b", "Garage Conversion"),
    (r"\bgeneral building\b", "General Building"),
    (r"\bland surveyors?\b", "Land Surveying"),
    (r"\blintel installation\b", "Lintel Installation"),
    (r"\blintel repair\b", "Lintel Repair"),
    (r"\blintel replacement\b", "Lintel Replacement"),
    (r"\bloft conversions?\b", "Loft Conversion"),
    (r"\bloft extensions?\b", "Loft Extension"),
    (r"\bnew builds?\b", "New Build Construction"),
    (r"\bplanning consultants?\b", "Planning Consultancy"),
    (r"\bproperty extensions?\b", "Property Extensions"),
    (r"\brefurbishments?\b", "Property Refurbishment"),
    (r"\bretaining walls?\b", "Retaining Walls"),
    (r"\brolled steel joist installation\b", "RSJ Installation"),
    (r"\bstructural design\b", "Structural Design"),
    (r"\bstructural engineering services?\b", "Structural Engineering"),
    (r"\bstructural steels?\b", "Structural Steel Work"),
    (r"\bsubsidence repair\b", "Subsidence Repair"),
    (r"\bsubsidence surveys?\b", "Subsidence Survey"),
    (r"\bunder pinning\b", "Underpinning"),
    (r"\bunderpinning / piling / foundations\b", "Underpinning, Piling & Foundations"),
    (r"\bwall tie replacement\b", "Wall Tie Replacement"),
    (r"\bwall tie survey\b", "Wall Tie Survey"),
]


RESIDUAL_FAMILY_RULES = [

    ("Roofing Services",
     r"\b(gutter|downpipe|felt lap vent|leadwork|dry ridge|dry verge|single ply membrane roof|skylight|ridged tiles?)\b"),

    ("Doors, Windows & Glazing",
     r"\b(window|windows|glass|glazing|mirror|stable doors?|barn doors?|bespoke.*doors?|weather stripping)\b"),

    ("Furniture, Carpentry & Woodwork",
     r"\b(architrave|cupboard|shelving|skirting|wood turning|wood repair|timber structures?|made to measure stairs?|under stairs storage)\b"),

    ("Kitchen Services",
     r"\b(kitchen|worktop)\b"),

    ("Plumbing & Drainage Services",
     r"\b(dripping tap|lead pipe|pipe fitter|sink |soil stack|water leakage|water pipe|water pressure|power flushing|power showers?)\b"),

    ("Heating, Boiler & Gas Services",
     r"\b(hot water system|underfloor heating|ufh |electric heating|electric radiator|lpg engineer|gas safety|gas stove|flue specialist|fireplaces?|stoves?|wood burner)\b"),

    ("Electrical Services",
     r"\b(electric |electrical|chandelier|downlight|extractor fan|fault finding|lighting|power point|sockets? and switches?)\b"),

    ("Painting, Decorating & Wall Finishes",
     r"\b(artex|coving|cornice|drywall|plaster|render|k rend|lime plaster|lime render|pebble dash|plasterboard|skim|skimming|suspended ceiling|tape and jointing|metal stud|stud wall|stud partition)\b"),

    ("Flooring & Tiling Services",
     r"\b(carpet|screeding|tiles?|tiling|regrouting)\b"),

    ("Gardening & Landscaping",
     r"\b(aviary|chicken run|equestrian stable|golf / bowls green|gritting|hedging|land clearing|landscape|planting|pruning|sensory garden|shrub|water garden|yard work)\b"),

    ("Vehicle Services",
     r"\b(car |vehicle|motorcycle|moped|scooter|van mot|mot testing|cambelt|clutch|dent repair|engine|fan belt|fuel injection|fuel pump|gearbox|glow plug|head gasket|headlight|ignition|mobile mechanic|oil and filter|panel beater|parking sensor|performance tuning|pre purchase car inspection|reversing camera|scratch repair|smart repair|steering wheel|suspension|timing belt|tow bar|towing|wheel |windscreen)\b"),

    ("IT Support & Device Services",
     r"\b(computers? and it|email setup|icloud|internet help|internet services|it and consultancy|macbook|microsoft |mobile phone repair|netflix setup|network installation|pc repair|printer|scanner setup|software help|troubleshooting|wix help|woocommerce help)\b"),

    ("Software & Development Services",
     r"\b(mobile app|website and app testing)\b"),

    ("Web & Digital Services",
     r"\b(website designers?|web$|wix help|woocommerce help)\b"),

    ("Writing, Translation & Language Services",
     r"\b(case study writing|creative writer|ghostwriting|journalist|linkedin profile writing|poetry writing|press release writing|proposal writing|report writing|speech writing|technical drawing|travel writer|white paper writer)\b"),

    ("Business & Administrative Services",
     r"\b(data entry|drafting|draftsman|hr services|market research|personal assistant|personal concierge|personal shopper|sales assistant|typist|mystery shopping)\b"),

    ("Accounting & Finance Services",
     r"\b(financial modelling|financial planning|financial reporting|mortgages?|pension advisor)\b"),

    ("Marketing & Creative Services",
     r"\b(content creation|presentation design|poster design|print design|illustrators?|digital design)\b"),

    ("Health, Therapy & Wellness",
     r"\b(chiropod|podiatrist|homeopathy|mental health|psychologist|wellness|day spa)\b"),

    ("Childcare & Care Services",
     r"\b(day nurseries?|home care|domiciliary care|maternity nurse)\b"),

    ("Fitness & Sports Coaching",
     r"\b(gym buddy|hiit|mma training|sports coaches?|triathlon|zumba)\b"),

    ("Pet & Animal Services",
     r"\b(aquarium|fish tank|puppy training|rabbit boarding|reptile boarding|vets?|wildlife management)\b"),

    ("Events & Wedding Services",
     r"\b(clown|comedians?|corporate filming|couple photoshoot|family photoshoot|florist|flower arrangements?|jugglers?|magician|mobile discos?|singer)\b"),

    ("Food, Baking & Personal Chef Services",
     r"\b(chef|takeaway|cafes?|coffee shops?|custom cakes?|celebration cakes?|pizza oven|sushi chef)\b"),

    ("Security & Alarm Services",
     r"\b(burglar|alarm|security|fire risk assessment|fire extinguishers?|smoke alarms?)\b"),

    ("Appliance & Equipment Services",
     r"\b(chainsaw|coffee machine|ghd|ice machine|microwave|oven fan|range cooker|rayburn|toaster|tool repair|treadmill|vacuum)\b"),

    ("Specialist Repair Services",
     r"\b(lamp repair|pool table recovering|sawblade sharpening|sharpening|tool sharpening|trampoline repair)\b"),

    ("Building & Construction",
     r"\b(foundations?|garage conversions?|general building|loft conversion|loft extension|new builds?|property extensions?|refurbishments?|retaining walls?|rsj|structural |subsidence|under pinning|underpinning|wall tie)\b"),

    ("Architecture, Surveying & Planning",
     r"\b(land surveyor|planning consultant|structural design)\b"),

    ("Water Treatment, Tanks & Pumps",
     r"\b(water coolers?|water filters?|water softeners?|water pumps?|water tanks?|purifiers?)\b"),

    ("Transport & Transfer Services",
     r"\b(transport|transport services|refrigerated transportation|haulage|return loads?)\b"),

    ("Tailoring & Clothing Alterations",
     r"\b(alterations?|dressmakers?|tailors?|women's tailoring|men's tailoring|coat repair)\b"),

    ("Printing, Signage & Art Services",
     r"\b(muralists?|wall murals?|poster design|print design|sign hanging)\b"),

    ("Commercial & Business Property Services",
     r"\b(hotels?|estate agents?|commercial sales|car dealerships?)\b"),
]


def canonical_subservice(raw_service: str) -> str:
    text = clean_text(raw_service)
    n = norm(text)

    for pattern, canonical in RESIDUAL_SUBSERVICE_RULES + LAST_LAP_SUBSERVICE_RULES + ADDITIONAL_SUBSERVICE_RULES + SUBSERVICE_RULES:
        if has(pattern, n):
            return canonical

    # Preserve the original service distinction when no canonical rule is needed.
    return text


def family_for(raw_service: str) -> str | None:
    n = norm(raw_service)

    for family, pattern in RESIDUAL_FAMILY_RULES + LAST_LAP_FAMILY_RULES + ADDITIONAL_FAMILY_RULES + FAMILY_RULES:
        if has(pattern, n):
            return family

    return None


def is_noise(raw_service: str) -> bool:
    n = norm(raw_service)

    if not n:
        return True

    if n in LOCATION_NAMES:
        return True

    return any(has(pattern, n) for pattern in NOISE_PATTERNS)


# ============================================================
# PIPELINE
# ============================================================

def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE).copy()

    required = {
        "platform",
        "raw_service",
        "service_family",
        "sub_service",
        "taxonomy_status",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Keep existing mapped work. Only revisit kept_unmapped and obvious exclusions.
    for idx, row in df.iterrows():
        raw = clean_text(row["raw_service"])
        status = clean_text(row["taxonomy_status"])

        if status in {"kept_mapped", "kept_mapped_final"}:
            continue

        if is_noise(raw):
            df.at[idx, "service_family"] = "Excluded"
            df.at[idx, "sub_service"] = raw
            df.at[idx, "taxonomy_status"] = "excluded"
            continue

        family = family_for(raw)

        if family:
            df.at[idx, "service_family"] = family
            df.at[idx, "sub_service"] = canonical_subservice(raw)
            df.at[idx, "taxonomy_status"] = "kept_mapped_final"
        else:
            # Genuine service remains in the dataset.
            df.at[idx, "service_family"] = "Other Services / Unclassified"
            df.at[idx, "sub_service"] = raw
            df.at[idx, "taxonomy_status"] = "kept_unclassified"

    df.to_csv(FINAL_OUTPUT, index=False)

    kept_mask = df["taxonomy_status"].isin(
        ["kept_mapped", "kept_mapped_final", "kept_unclassified"]
    )

    coverage = (
        df[kept_mask]
        .groupby(["service_family", "sub_service"], as_index=False)
        .agg(
            platform_count=("platform", "nunique"),
            platforms=("platform", lambda x: " | ".join(sorted(set(x), key=str.casefold))),
            raw_label_count=("raw_service", "nunique"),
            row_count=("raw_service", "size"),
        )
        .sort_values(
            ["platform_count", "raw_label_count", "service_family", "sub_service"],
            ascending=[False, False, True, True],
        )
    )

    coverage.to_csv(COVERAGE_OUTPUT, index=False)

    remaining = (
        df[df["taxonomy_status"] == "kept_unclassified"]
        [["platform", "raw_service"]]
        .drop_duplicates()
        .sort_values(["raw_service", "platform"], key=lambda c: c.str.casefold())
    )

    remaining.to_csv(REMAINING_OUTPUT, index=False)

    summary = pd.DataFrame([
        {"metric": "input_rows", "value": len(df)},
        {"metric": "existing_mapped_rows", "value": int((df["taxonomy_status"] == "kept_mapped").sum())},
        {"metric": "newly_mapped_rows", "value": int((df["taxonomy_status"] == "kept_mapped_final").sum())},
        {"metric": "remaining_unclassified_rows", "value": int((df["taxonomy_status"] == "kept_unclassified").sum())},
        {"metric": "excluded_rows", "value": int((df["taxonomy_status"] == "excluded").sum())},
        {"metric": "service_families", "value": int(df.loc[kept_mask, "service_family"].nunique())},
        {"metric": "sub_services", "value": int(df.loc[kept_mask, "sub_service"].nunique())},
    ])

    summary.to_csv(SUMMARY_OUTPUT, index=False)

    print("=" * 78)
    print("SEVERSE FINAL TAXONOMY PASS")
    print("=" * 78)
    print(f"\nInput rows: {len(df):,}")
    print(f"Existing mapped rows: {(df['taxonomy_status'] == 'kept_mapped').sum():,}")
    print(f"Newly mapped rows: {(df['taxonomy_status'] == 'kept_mapped_final').sum():,}")
    print(f"Remaining genuine but unclassified: {(df['taxonomy_status'] == 'kept_unclassified').sum():,}")
    print(f"Excluded noise/location rows: {(df['taxonomy_status'] == 'excluded').sum():,}")
    print(f"Service families: {df.loc[kept_mask, 'service_family'].nunique():,}")
    print(f"Sub-services: {df.loc[kept_mask, 'sub_service'].nunique():,}")

    print("\nFiles written:")
    print(f"  {FINAL_OUTPUT}")
    print(f"  {COVERAGE_OUTPUT}")
    print(f"  {REMAINING_OUTPUT}")
    print(f"  {SUMMARY_OUTPUT}")

    print(
        "\nUse service_taxonomy_finished.csv as the final taxonomy dataset. "
        "The remaining_unclassified file is only for optional later refinement."
    )


if __name__ == "__main__":
    main()

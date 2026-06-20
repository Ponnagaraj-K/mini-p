"""
Submarine Knowledge Base
Based on publicly available naval recognition data (Jane's Fighting Ships, public domain)
All country inferences are visual-feature-based only — NOT confirmed intelligence.
"""

SUBMARINE_DATABASE = {
    "kilo_class": {
        "display_name": "Kilo-class (Project 877/636)",
        "type": "Conventional Attack Submarine (SSK)",
        "category": "attack",
        "countries": ["Russia", "India", "Algeria", "Vietnam", "China", "Iran", "Poland"],
        "length_m": (73, 74),
        "beam_m": 9.9,
        "visual_features": [
            "teardrop hull shape",
            "single large propeller",
            "small streamlined conning tower",
            "smooth hull surface",
            "forward hydroplanes on bow"
        ],
        "threat_level": "HIGH",
        "propulsion": "Diesel-electric (AIP in 636 variant)",
        "notes": "Most widely exported conventional submarine. Indian Navy operates Sindhughosh-class (Kilo variant).",
        "friendly_operators": ["India"]
    },
    "akula_class": {
        "display_name": "Akula-class (Project 971)",
        "type": "Nuclear Attack Submarine (SSN)",
        "category": "nuclear_attack",
        "countries": ["Russia", "India (leased as INS Chakra)"],
        "length_m": (110, 113),
        "beam_m": 13.6,
        "visual_features": [
            "large elongated hull",
            "prominent fin/sail",
            "teardrop body shape",
            "pump-jet or 7-blade propeller",
            "retractable bow planes"
        ],
        "threat_level": "CRITICAL",
        "propulsion": "Nuclear",
        "notes": "Nuclear capable. India leased Akula-II as INS Chakra.",
        "friendly_operators": ["India (leased)"]
    },
    "virginia_class": {
        "display_name": "Virginia-class (SSN-774)",
        "type": "Nuclear Attack Submarine (SSN)",
        "category": "nuclear_attack",
        "countries": ["USA"],
        "length_m": (114, 115),
        "beam_m": 10.4,
        "visual_features": [
            "sail-mounted horizontal planes",
            "no traditional periscopes (photonics masts)",
            "smooth blunt bow",
            "pump-jet propulsor",
            "X-shaped stern control surfaces"
        ],
        "threat_level": "HIGH",
        "propulsion": "Nuclear",
        "notes": "US Navy primary attack submarine. AUKUS partner — Australia acquiring.",
        "friendly_operators": []
    },
    "yuan_class": {
        "display_name": "Yuan-class (Type 039A/B)",
        "type": "Conventional Attack Submarine (SSK/AIP)",
        "category": "attack",
        "countries": ["China"],
        "length_m": (77, 78),
        "beam_m": 8.4,
        "visual_features": [
            "AIP system hull bulge amidships",
            "distinctive albacore teardrop hull",
            "angled bow",
            "mid-sized conning tower",
            "7-blade skewed propeller"
        ],
        "threat_level": "HIGH",
        "propulsion": "Diesel-electric + AIP",
        "notes": "PLAN primary conventional submarine. Significant numbers in South China Sea / Indian Ocean.",
        "friendly_operators": []
    },
    "shang_class": {
        "display_name": "Shang-class (Type 093)",
        "type": "Nuclear Attack Submarine (SSN)",
        "category": "nuclear_attack",
        "countries": ["China"],
        "length_m": (107, 110),
        "beam_m": 11.0,
        "visual_features": [
            "large hull diameter",
            "prominent sail structure",
            "elongated body",
            "rounded bow",
            "cruciform stern planes"
        ],
        "threat_level": "CRITICAL",
        "propulsion": "Nuclear",
        "notes": "PLAN nuclear attack submarine. Active in Indian Ocean region.",
        "friendly_operators": []
    },
    "scorpene_class": {
        "display_name": "Scorpène-class (CM-2000)",
        "type": "Conventional Attack Submarine (SSK)",
        "category": "attack",
        "countries": ["India", "France", "Brazil", "Chile", "Malaysia", "Morocco"],
        "length_m": (66, 67.5),
        "beam_m": 6.2,
        "visual_features": [
            "slim teardrop hull",
            "compact conning tower",
            "X-form stern planes",
            "no bow planes",
            "smooth hydrodynamic profile"
        ],
        "threat_level": "MEDIUM",
        "propulsion": "Diesel-electric (AIP optional)",
        "notes": "Indian Navy Kalvari-class are Scorpène variant. 6 in service.",
        "friendly_operators": ["India", "France"]
    },
    "midget_submarine": {
        "display_name": "Midget Submarine (various)",
        "type": "Midget Submarine (SSM)",
        "category": "midget",
        "countries": ["North Korea", "Iran", "Non-state actors", "Unknown"],
        "length_m": (10, 30),
        "beam_m": (2, 4),
        "visual_features": [
            "very small hull",
            "minimal or no conning tower",
            "simple cylindrical body",
            "small propeller",
            "limited hydrodynamic features"
        ],
        "threat_level": "CRITICAL",
        "propulsion": "Battery electric / small diesel",
        "notes": "Used for covert infiltration, mine laying, special operations. High asymmetric threat.",
        "friendly_operators": []
    },
    "arihant_class": {
        "display_name": "Arihant-class (SSBN)",
        "type": "Nuclear Ballistic Missile Submarine (SSBN)",
        "category": "nuclear_ballistic",
        "countries": ["India"],
        "length_m": (111, 112),
        "beam_m": 11.0,
        "visual_features": [
            "large hull with missile hump behind sail",
            "distinctive bulge on upper aft hull",
            "streamlined conning tower",
            "pump-jet propulsor"
        ],
        "threat_level": "FRIENDLY",
        "propulsion": "Nuclear",
        "notes": "India's indigenous SSBN. Nuclear deterrent. FRIENDLY asset.",
        "friendly_operators": ["India"]
    },
    "unknown_submarine": {
        "display_name": "Unknown Submarine",
        "type": "Unidentified Submarine",
        "category": "unknown",
        "countries": ["Unknown"],
        "length_m": (0, 0),
        "beam_m": 0,
        "visual_features": [],
        "threat_level": "HIGH",
        "propulsion": "Unknown",
        "notes": "Unidentified submarine. Treat as potential threat until confirmed.",
        "friendly_operators": []
    }
}

# Non-submarine underwater objects
UNDERWATER_OBJECTS = {
    "mine": {
        "display_name": "Underwater Mine",
        "threat_level": "CRITICAL",
        "description": "Explosive naval mine. Immediate danger to vessels.",
        "action": "IMMEDIATE ALERT — Do not approach — Report to naval command"
    },
    "diver": {
        "display_name": "Unauthorized Diver / Frogman",
        "threat_level": "HIGH",
        "description": "Human diver detected in restricted waters.",
        "action": "Alert security personnel — Verify authorization — Monitor movement"
    },
    "uuv": {
        "display_name": "Unmanned Underwater Vehicle (UUV/Torpedo-like)",
        "threat_level": "HIGH",
        "description": "Autonomous underwater vehicle or torpedo-like object detected.",
        "action": "Track trajectory — Alert naval command — Possible ISR or weapon"
    },
    "drone_underwater": {
        "display_name": "Underwater Drone",
        "threat_level": "MEDIUM",
        "description": "Small underwater drone detected. Possible surveillance device.",
        "action": "Track and log — Possible intelligence gathering device"
    },
    "fish_school": {
        "display_name": "Fish School",
        "threat_level": "NONE",
        "description": "Group of fish detected. No security threat.",
        "action": "Log for marine biology record"
    },
    "marine_mammal": {
        "display_name": "Marine Mammal (Dolphin/Whale)",
        "threat_level": "NONE",
        "description": "Large marine mammal detected.",
        "action": "Log for marine wildlife record — Avoid disturbance"
    },
    "coral_structure": {
        "display_name": "Coral / Rock Structure",
        "threat_level": "NONE",
        "description": "Natural underwater structure detected.",
        "action": "Environmental log"
    },
    "debris": {
        "display_name": "Underwater Debris / Wreckage",
        "threat_level": "LOW",
        "description": "Unidentified debris or wreckage on seafloor.",
        "action": "Log and monitor — Check for hazardous materials"
    },
    "pipeline": {
        "display_name": "Underwater Pipeline / Cable",
        "threat_level": "LOW",
        "description": "Infrastructure cable or pipeline detected.",
        "action": "Monitor for damage or tampering"
    },
    "invasive_species": {
        "display_name": "Invasive Marine Species",
        "threat_level": "NONE",
        "description": "Potentially invasive species detected.",
        "action": "Report to marine biology team"
    }
}

THREAT_COLORS = {
    "CRITICAL": "#FF0000",
    "HIGH": "#FF6600",
    "MEDIUM": "#FFAA00",
    "LOW": "#FFFF00",
    "NONE": "#00FF00",
    "FRIENDLY": "#0080FF"
}

THREAT_ACTIONS = {
    "CRITICAL": "IMMEDIATE ACTION REQUIRED — Alert naval command now",
    "HIGH": "HIGH PRIORITY — Alert operator — Do not act autonomously",
    "MEDIUM": "FLAG FOR REVIEW — Monitor closely — Await confirmation",
    "LOW": "LOG AND MONITOR — No immediate action required",
    "NONE": "No threat — Logged for record",
    "FRIENDLY": "FRIENDLY ASSET — Verify IFF — Do not engage"
}


def get_submarine_info(class_key: str) -> dict:
    return SUBMARINE_DATABASE.get(class_key, SUBMARINE_DATABASE["unknown_submarine"])


def get_object_info(class_key: str) -> dict:
    return UNDERWATER_OBJECTS.get(class_key, {
        "display_name": "Unidentified Object",
        "threat_level": "MEDIUM",
        "description": "Object could not be classified with available data.",
        "action": "Manual operator review required"
    })


def find_submarines_by_country(country: str) -> list:
    return [
        key for key, val in SUBMARINE_DATABASE.items()
        if any(country.lower() in c.lower() for c in val["countries"])
    ]


def infer_countries_from_features(detected_features: list, sub_type: str) -> dict:
    """
    Infers possible countries based on visual features.
    Returns possibilities with reasoning — never a definitive answer.
    """
    matches = {}
    for key, sub in SUBMARINE_DATABASE.items():
        if sub["category"] == "unknown":
            continue
        feature_overlap = sum(
            1 for f in detected_features
            if any(f.lower() in sf.lower() for sf in sub["visual_features"])
        )
        if feature_overlap > 0:
            score = feature_overlap / max(len(sub["visual_features"]), 1)
            matches[sub["display_name"]] = {
                "score": round(score * 100, 1),
                "countries": sub["countries"],
                "threat_level": sub["threat_level"],
                "matched_features": feature_overlap,
                "friendly_operators": sub["friendly_operators"]
            }
    return dict(sorted(matches.items(), key=lambda x: x[1]["score"], reverse=True)[:3])

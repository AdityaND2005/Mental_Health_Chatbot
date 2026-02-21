"""
Indian-specific crisis resources and helplines for mental health emergencies
"""

INDIAN_CRISIS_RESOURCES = {
    "national_helplines": [
        {
            "name": "KIRAN Mental Health Rehabilitation Helpline",
            "number": "1800-599-0019",
            "availability": "24/7",
            "description": "National helpline by Ministry of Social Justice and Empowerment for mental health support",
            "languages": ["Hindi", "English", "Regional languages"]
        },
        {
            "name": "Vandrevala Foundation Helpline",
            "number": "1860-266-2345 / +91-8657-666-555",
            "availability": "24/7",
            "description": "Free mental health helpline and counseling support",
            "languages": ["Hindi", "English"]
        },
        {
            "name": "iCall - TISS Helpline",
            "number": "022-2556-3291 / 9152-987-821",
            "availability": "Mon-Sat, 8 AM to 10 PM",
            "description": "Psychosocial helpline by Tata Institute of Social Sciences",
            "languages": ["Hindi", "English", "Marathi", "Tamil", "Telugu"]
        },
        {
            "name": "Snehi - National Suicide Prevention Helpline",
            "number": "+91-9899-889-235",
            "availability": "24/7",
            "description": "Delhi-based suicide prevention and crisis intervention",
            "languages": ["Hindi", "English"]
        }
    ],
    
    "state_specific": {
        "Karnataka": [
            {
                "name": "SAHAI - Bangalore",
                "number": "+91-8022-960-000 / +91-8022-960-001",
                "availability": "24/7",
                "description": "24-hour helpline for people with mental health issues",
                "languages": ["Kannada", "English", "Hindi"]
            }
        ],
        "Maharashtra": [
            {
                "name": "Connecting Trust - Mumbai",
                "number": "+91-9922-001-122 / +91-9922-004-305",
                "availability": "12 PM to 8 PM daily",
                "description": "Emotional support and suicide prevention",
                "languages": ["Hindi", "English", "Marathi"]
            },
            {
                "name": "Aasra - Mumbai",
                "number": "+91-9820-466-726",
                "availability": "24/7",
                "description": "Helpline for distress and suicidal thoughts",
                "languages": ["Hindi", "English", "Marathi"]
            }
        ],
        "Tamil Nadu": [
            {
                "name": "Sneha - Chennai",
                "number": "+91-4424-640-050",
                "availability": "24/7",
                "description": "Suicide prevention center with emotional support",
                "languages": ["Tamil", "English", "Hindi"]
            }
        ],
        "West Bengal": [
            {
                "name": "Sumaitri - Kolkata",
                "number": "+91-3323-637-401 / +91-3323-637-432",
                "availability": "4 PM to 10 PM daily",
                "description": "Emotional support and crisis intervention",
                "languages": ["Bengali", "English", "Hindi"]
            }
        ],
        "Kerala": [
            {
                "name": "Maithri - Kochi",
                "number": "+91-4842-539-000",
                "availability": "24/7",
                "description": "Emotional support helpline",
                "languages": ["Malayalam", "English", "Hindi"]
            }
        ],
        "Delhi": [
            {
                "name": "Mental Health Helpline - Delhi Govt",
                "number": "+91-8800-144-353",
                "availability": "Mon-Sat, 9 AM to 5 PM",
                "description": "Delhi government mental health helpline",
                "languages": ["Hindi", "English"]
            }
        ]
    },
    
    "specialized_helplines": {
        "youth": [
            {
                "name": "Childline India",
                "number": "1098",
                "availability": "24/7",
                "description": "Emergency helpline for children and youth in distress",
                "languages": ["Hindi", "English", "Regional languages"]
            }
        ],
        "women": [
            {
                "name": "Women Power Helpline",
                "number": "181",
                "availability": "24/7",
                "description": "Support for women in distress",
                "languages": ["Hindi", "English", "Regional languages"]
            }
        ],
        "lgbtq": [
            {
                "name": "The Humsafar Trust Helpline",
                "number": "+91-8425-908-001",
                "availability": "Mon-Sat, 10 AM to 6 PM",
                "description": "LGBTQ+ mental health support",
                "languages": ["Hindi", "English", "Marathi"]
            }
        ]
    },
    
    "emergency_services": [
        {
            "name": "National Emergency Number",
            "number": "112",
            "availability": "24/7",
            "description": "Single emergency number for all emergencies including medical",
            "languages": ["All Indian languages"]
        },
        {
            "name": "Ambulance",
            "number": "102 / 108",
            "availability": "24/7",
            "description": "Emergency medical services",
            "languages": ["All Indian languages"]
        }
    ],
    
    "online_resources": [
        {
            "name": "NIMHANS - Tele MANAS",
            "description": "National Institute of Mental Health and Neuro Sciences tele-counseling",
            "website": "https://nimhans.ac.in",
            "contact": "080-4611-0007"
        },
        {
            "name": "MindPeers",
            "description": "Online mental health platform with therapists",
            "website": "https://mindpeers.co"
        },
        {
            "name": "YourDOST",
            "description": "Emotional wellness and counseling platform",
            "website": "https://yourdost.com"
        }
    ]
}


def format_crisis_resources(user_state=None):
    """
    Format crisis resources for display to user
    
    Args:
        user_state: Optional state name for state-specific resources
    
    Returns:
        Formatted string with crisis resources
    """
    resources_text = "\n" + "=" * 80 + "\n"
    resources_text += "🚨 **IMMEDIATE CRISIS RESOURCES - INDIA** 🚨\n"
    resources_text += "=" * 80 + "\n\n"
    
    resources_text += "**Please reach out to these helplines immediately:**\n\n"
    
    # National helplines
    resources_text += "**NATIONAL HELPLINES (Available across India):**\n"
    for helpline in INDIAN_CRISIS_RESOURCES["national_helplines"]:
        resources_text += f"\n📞 **{helpline['name']}**\n"
        resources_text += f"   Number: {helpline['number']}\n"
        resources_text += f"   Hours: {helpline['availability']}\n"
        resources_text += f"   Languages: {', '.join(helpline['languages'])}\n"
    
    # State-specific (if provided)
    if user_state and user_state in INDIAN_CRISIS_RESOURCES["state_specific"]:
        resources_text += f"\n\n**YOUR STATE ({user_state.upper()}) HELPLINES:**\n"
        for helpline in INDIAN_CRISIS_RESOURCES["state_specific"][user_state]:
            resources_text += f"\n📞 **{helpline['name']}**\n"
            resources_text += f"   Number: {helpline['number']}\n"
            resources_text += f"   Hours: {helpline['availability']}\n"
    
    # Emergency services
    resources_text += "\n\n**EMERGENCY SERVICES:**\n"
    for service in INDIAN_CRISIS_RESOURCES["emergency_services"]:
        resources_text += f"\n🚑 **{service['name']}**: {service['number']}\n"
        resources_text += f"   {service['description']}\n"
    
    resources_text += "\n" + "=" * 80 + "\n"
    resources_text += "**You are not alone. Help is available. Please reach out.**\n"
    resources_text += "=" * 80 + "\n\n"
    
    return resources_text


def get_all_crisis_numbers():
    """Get all crisis numbers as a simple list"""
    numbers = []
    
    # National helplines
    for helpline in INDIAN_CRISIS_RESOURCES["national_helplines"]:
        numbers.append(f"{helpline['name']}: {helpline['number']}")
    
    # Emergency services
    for service in INDIAN_CRISIS_RESOURCES["emergency_services"]:
        numbers.append(f"{service['name']}: {service['number']}")
    
    return numbers


# Crisis-specific templated responses
CRISIS_RESPONSE_TEMPLATES = [
    "I'm really concerned about what you're sharing. Your safety is the most important thing right now.",
    "What you're feeling is valid, and I want you to know that there is help available immediately.",
    "I can hear that you're in a lot of pain right now. Please reach out to these crisis helplines who can provide immediate support.",
    "You don't have to face this alone. There are trained professionals available 24/7 who want to help you.",
]


def build_crisis_response(user_message, crisis_confidence):
    """
    Build a compassionate crisis response with resources
    
    Args:
        user_message: User's message that triggered crisis detection
        crisis_confidence: Confidence score from crisis classifier
    
    Returns:
        Formatted crisis response with resources
    """
    import random
    
    response = random.choice(CRISIS_RESPONSE_TEMPLATES) + "\n\n"
    response += format_crisis_resources()
    
    # Add encouraging message
    response += "\n**Remember:**\n"
    response += "• You matter and your life has value\n"
    response += "• What you're feeling is temporary, even though it doesn't feel that way right now\n"
    response += "• Trained professionals can help you through this moment\n"
    response += "• Reaching out is a sign of strength, not weakness\n\n"
    
    response += "Would you like to talk about what's going on? I'm here to listen and support you."
    
    return response

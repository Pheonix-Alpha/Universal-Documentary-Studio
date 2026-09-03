"""
Production Bible Generator - The "Director's Brain"

Creates a comprehensive production bible that ensures consistency
across all scenes and workers.
"""

import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime

from app import model_manager
from app.scene_analyzer import analyze_story, _strip_code_fence


def _scene_people(scene: Dict) -> List[str]:
    """scene['people'] is a list of names when scenes come from the local
    LLM (see scene_analyzer._analyze_with_local_brain's prompt), but code elsewhere
    in this file used to assume it was a comma-separated string and called
    .split(',') on it -- which raises AttributeError on a real list. Accept
    either shape."""
    people = scene.get('people', [])
    if isinstance(people, str):
        return [p.strip() for p in people.split(',') if p.strip()]
    return [p.strip() for p in (people or []) if p and p.strip()]


@dataclass
class Character:
    """A character in the production"""
    name: str
    role: str  # protagonist, antagonist, supporting, etc.
    age: Optional[int] = None
    gender: Optional[str] = None
    appearance: str = ""
    personality_traits: List[str] = None
    voice_style: str = ""
    key_relationships: Dict[str, str] = None
    reference_image_prompt: str = ""  # For consistency
    
    def __post_init__(self):
        if self.personality_traits is None:
            self.personality_traits = []
        if self.key_relationships is None:
            self.key_relationships = {}


@dataclass
class Location:
    """A location in the production"""
    name: str
    description: str = ""
    atmosphere: str = ""
    key_visual_elements: List[str] = None
    time_of_day: str = ""  # morning, afternoon, night, etc.
    weather: str = ""
    
    def __post_init__(self):
        if self.key_visual_elements is None:
            self.key_visual_elements = []


@dataclass
class VisualStyle:
    """The visual style bible"""
    name: str = ""
    color_palette: List[str] = None
    cinematography_style: str = ""  # documentary, cinematic, handheld, etc.
    lighting: str = ""  # natural, dramatic, soft, etc.
    camera_style: str = ""  # static, tracking, steadycam, etc.
    film_grain: bool = False
    aspect_ratio: str = "16:9"
    video_style: str = ""  # realistic, animated, stylized, etc.
    
    def __post_init__(self):
        if self.color_palette is None:
            self.color_palette = []


@dataclass
class ProductionBible:
    """The complete production bible"""
    title: str = ""
    genre: str = ""
    era: str = ""
    logline: str = ""
    characters: Dict[str, Character] = None
    locations: Dict[str, Location] = None
    visual_style: VisualStyle = None
    scenes: List[Dict[str, Any]] = None  # Enhanced scene definitions
    global_seed: int = 42  # For consistency across generations
    created_at: str = ""
    version: int = 1
    
    def __post_init__(self):
        if self.characters is None:
            self.characters = {}
        if self.locations is None:
            self.locations = {}
        if self.visual_style is None:
            self.visual_style = VisualStyle()
        if self.scenes is None:
            self.scenes = []
        self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Convert to dict for JSON serialization"""
        result = {
            'title': self.title,
            'genre': self.genre,
            'era': self.era,
            'logline': self.logline,
            'characters': {k: asdict(v) for k, v in self.characters.items()},
            'locations': {k: asdict(v) for k, v in self.locations.items()},
            'visual_style': asdict(self.visual_style),
            'scenes': self.scenes,
            'global_seed': self.global_seed,
            'created_at': self.created_at,
            'version': self.version
        }
        return result


def generate_production_bible(story: str, use_local_brain: bool = True) -> ProductionBible:
    """
    Generate a complete production bible from a story.
    This is the main entry point for the "Director's Brain" -- which is now
    the local LLM running on the main Colab's own GPU, not the Anthropic API.
    """
    bible = ProductionBible()
    
    # Step 1: Analyze scenes first (we have this already)
    scenes = analyze_story(story)
    bible.scenes = scenes
    
    # Step 2: Extract characters, locations, and themes
    if use_local_brain:
        bible = _generate_bible_with_local_brain(story, bible)
    else:
        bible = _generate_bible_fallback(story, bible)
    
    # Step 3: Enhance scenes with bible context
    bible.scenes = _enrich_scenes_with_bible(bible.scenes, bible)
    
    return bible


def _generate_bible_with_local_brain(story: str, bible: ProductionBible) -> ProductionBible:
    """Use the local LLM (model_manager.generate_text) to generate a
    comprehensive bible. Falls back to the rule-based bible on any failure
    (e.g. no GPU available) rather than leaving the bible empty."""
    prompt = f"""
You are a master documentary film producer. Analyze this story and create a detailed production bible.

STORY:
{story}

OUTPUT AS JSON:
{{
    "title": "A compelling title for this documentary",
    "genre": "Documentary genre (historical, nature, biographical, etc.)",
    "era": "Time period (e.g., 1920s, modern day, Victorian era)",
    "logline": "A one-sentence summary of the story",
    "characters": {{
        "character_name": {{
            "role": "protagonist/antagonist/supporting",
            "age": number,
            "gender": "string",
            "appearance": "Detailed physical description",
            "personality_traits": ["trait1", "trait2"],
            "voice_style": "How they speak (formal, colloquial, etc.)",
            "key_relationships": {{"other_character": "relationship description"}},
            "reference_image_prompt": "A prompt that would generate this character in Stable Diffusion"
        }}
    }},
    "locations": {{
        "location_name": {{
            "description": "Detailed description",
            "atmosphere": "Mood/feeling of the location",
            "key_visual_elements": ["element1", "element2"],
            "time_of_day": "morning/afternoon/night",
            "weather": "sunny/rainy/etc"
        }}
    }},
    "visual_style": {{
        "color_palette": ["#hex1", "#hex2"],
        "cinematography_style": "documentary/cinematic/handheld",
        "lighting": "natural/dramatic/soft",
        "camera_style": "static/tracking/steadycam",
        "film_grain": true/false,
        "video_style": "realistic/animated/stylized"
    }}
}}

Extract ALL characters and locations mentioned. If not specified, infer from context.
    """
    
    try:
        content = model_manager.generate_text(
            user_prompt=prompt,
            system_prompt="You are a production bible generator. Output only valid JSON.",
            max_new_tokens=4096,
            temperature=0.3,
        )
        data = json.loads(_strip_code_fence(content))
        
        # Populate the bible
        bible.title = data.get('title', 'Untitled Documentary')
        bible.genre = data.get('genre', 'Documentary')
        bible.era = data.get('era', 'Present Day')
        bible.logline = data.get('logline', '')
        
        # Characters
        for name, char_data in data.get('characters', {}).items():
            bible.characters[name] = Character(
                name=name,
                role=char_data.get('role', 'supporting'),
                age=char_data.get('age'),
                gender=char_data.get('gender'),
                appearance=char_data.get('appearance', ''),
                personality_traits=char_data.get('personality_traits', []),
                voice_style=char_data.get('voice_style', ''),
                key_relationships=char_data.get('key_relationships', {}),
                reference_image_prompt=char_data.get('reference_image_prompt', '')
            )
        
        # Locations
        for name, loc_data in data.get('locations', {}).items():
            bible.locations[name] = Location(
                name=name,
                description=loc_data.get('description', ''),
                atmosphere=loc_data.get('atmosphere', ''),
                key_visual_elements=loc_data.get('key_visual_elements', []),
                time_of_day=loc_data.get('time_of_day', ''),
                weather=loc_data.get('weather', '')
            )
        
        # Visual Style
        style_data = data.get('visual_style', {})
        bible.visual_style = VisualStyle(
            name=style_data.get('name', 'Documentary Style'),
            color_palette=style_data.get('color_palette', []),
            cinematography_style=style_data.get('cinematography_style', 'documentary'),
            lighting=style_data.get('lighting', 'natural'),
            camera_style=style_data.get('camera_style', 'static'),
            film_grain=style_data.get('film_grain', False),
            video_style=style_data.get('video_style', 'realistic')
        )
        
    except Exception as e:
        print(f"[production_bible] Local brain bible generation failed, falling back to rule-based bible: {e}")
        return _generate_bible_fallback(story, bible)
    
    return bible


def _generate_bible_fallback(story: str, bible: ProductionBible) -> ProductionBible:
    """Rule-based fallback bible generation"""
    # Extract characters from scenes
    all_chars = set()
    for scene in bible.scenes:
        for name in _scene_people(scene):
            if name not in all_chars:
                all_chars.add(name)
                bible.characters[name] = Character(
                    name=name,
                    role='supporting'
                )
    
    # Extract locations
    for scene in bible.scenes:
        loc = scene.get('location', '')
        if loc and loc not in bible.locations:
            bible.locations[loc] = Location(name=loc)
    
    return bible


def _enrich_scenes_with_bible(scenes: List[Dict], bible: ProductionBible) -> List[Dict]:
    """Add bible context to each scene"""
    enriched = []
    
    for scene in scenes:
        enhanced = dict(scene)
        
        # Add character details
        people = _scene_people(scene)
        if people:
            char_descriptions = []
            for name in people:
                if name in bible.characters:
                    char = bible.characters[name]
                    char_descriptions.append(f"{name}: {char.appearance or char.role}")
            enhanced['character_details'] = char_descriptions
        
        # Add location details
        loc_name = scene.get('location', '')
        if loc_name in bible.locations:
            loc = bible.locations[loc_name]
            enhanced['location_details'] = {
                'description': loc.description,
                'atmosphere': loc.atmosphere,
                'time_of_day': loc.time_of_day,
                'weather': loc.weather
            }
        
        # Add style context
        enhanced['style_context'] = {
            'visual_style': bible.visual_style.video_style,
            'color_palette': bible.visual_style.color_palette,
            'cinematography': bible.visual_style.cinematography_style,
            'lighting': bible.visual_style.lighting,
            'camera': bible.visual_style.camera_style
        }
        
        # Include global seed
        enhanced['seed'] = bible.global_seed + scene.get('scene_id', 0)
        
        enriched.append(enhanced)
    
    return enriched
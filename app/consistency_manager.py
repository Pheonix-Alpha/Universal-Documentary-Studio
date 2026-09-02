"""
Consistency Manager - Ensures all workers generate consistent content
"""

from typing import Dict, Any, List, Optional
import json
import hashlib

from app.production_bible import ProductionBible, Character, Location
from app.scene_orchestrator import ProductionUnit


class ConsistencyManager:
    """
    Manages cross-worker consistency by sharing bible data and
    enforcing consistent seeds and reference contexts.
    """
    
    def __init__(self, bible: ProductionBible):
        self.bible = bible
        self.character_reference_cache: Dict[str, str] = {}  # character -> reference embedding/prompt
    
    def get_worker_context(self, unit: ProductionUnit) -> Dict[str, Any]:
        """
        Generate context for a worker to ensure consistency.
        This gets sent with every video generation request.
        """
        context = {
            'global_seed': self.bible.global_seed,
            'scene_seed': unit.seed,
            'style': {
                'visual_style': self.bible.visual_style.video_style,
                'color_palette': self.bible.visual_style.color_palette,
                'cinematography': self.bible.visual_style.cinematography_style,
                'lighting': self.bible.visual_style.lighting,
                'camera': self.bible.visual_style.camera_style,
                'film_grain': self.bible.visual_style.film_grain
            },
            'characters': {},
            'locations': {},
            'reference_prompts': {}
        }
        
        # Add character references
        for char_name in unit.characters:
            if char_name in self.bible.characters:
                char = self.bible.characters[char_name]
                context['characters'][char_name] = {
                    'appearance': char.appearance,
                    'personality': char.personality_traits,
                    'voice': char.voice_style,
                    'reference_prompt': char.reference_image_prompt
                }
        
        # Add location reference
        if unit.location in self.bible.locations:
            loc = self.bible.locations[unit.location]
            context['locations'][unit.location] = {
                'description': loc.description,
                'atmosphere': loc.atmosphere,
                'key_elements': loc.key_visual_elements,
                'time_of_day': loc.time_of_day,
                'weather': loc.weather
            }
        
        # Add character reference images (if we had an image generation step)
        # This is where you'd include IP-Adapter reference images
        
        return context
    
    def get_consistency_hash(self, unit: ProductionUnit) -> str:
        """
        Generate a hash that identifies this unit's consistency context.
        Workers can use this to cache context data.
        """
        context = self.get_worker_context(unit)
        context_str = json.dumps(context, sort_keys=True)
        return hashlib.md5(context_str.encode()).hexdigest()
    
    def share_character_reference(self, character_name: str, reference_data: str):
        """Share character reference data (embeddings or images) with all workers"""
        self.character_reference_cache[character_name] = reference_data
    
    def get_shared_character_reference(self, character_name: str) -> Optional[str]:
        """Retrieve shared character reference data"""
        return self.character_reference_cache.get(character_name)
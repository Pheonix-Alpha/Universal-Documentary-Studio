"""
Scene Orchestrator - Breaks bible into production units and assigns to workers
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import hashlib

from app.production_bible import ProductionBible, _scene_people


@dataclass
class ProductionUnit:
    """A single scene ready for video generation"""
    scene_id: int
    description: str
    characters: List[str]
    location: str
    visual_prompt: str  # Full prompt for video generation
    duration_seconds: int = 4  # Default clip length
    seed: int = 42
    style_context: Dict[str, Any] = None
    assigned_worker: Optional[str] = None
    status: str = "pending"  # pending, generating, completed, failed
    
    def to_dict(self) -> Dict:
        return asdict(self)


class SceneOrchestrator:
    """Orchestrates scene breakdown and worker assignment"""
    
    def __init__(self, bible: ProductionBible):
        self.bible = bible
        self.units: List[ProductionUnit] = []
        self._unit_map = {}
    
    def breakdown(self) -> List[ProductionUnit]:
        """Break the bible into production units (one per scene)"""
        self.units = []
        
        for scene in self.bible.scenes:
            unit = self._create_unit(scene)
            self.units.append(unit)
            self._unit_map[unit.scene_id] = unit
        
        return self.units
    
    def _create_unit(self, scene: Dict) -> ProductionUnit:
        """Create a production unit from a scene"""
        scene_id = scene.get('scene_id', 0)
        
        # Generate a comprehensive visual prompt
        visual_prompt = self._generate_visual_prompt(scene)
        
        # Extract characters. scene['people'] may be a list (Claude path) or
        # a comma-separated string (older fallback shape) -- _scene_people
        # handles both; calling .split(',') directly on a list here used to
        # raise AttributeError as soon as Claude-based scene analysis was on.
        characters = _scene_people(scene)
        
        unit = ProductionUnit(
            scene_id=scene_id,
            description=scene.get('description', ''),
            characters=characters,
            location=scene.get('location', ''),
            visual_prompt=visual_prompt,
            duration_seconds=4,
            seed=self.bible.global_seed + scene_id,
            style_context=scene.get('style_context', {})
        )
        
        return unit
    
    def _generate_visual_prompt(self, scene: Dict) -> str:
        """Generate a detailed visual prompt for video generation"""
        # Start with scene description
        prompt = scene.get('description', '')
        
        # Add style context
        style = scene.get('style_context', {})
        style_str = style.get('visual_style', 'realistic')
        cinematography = style.get('cinematography', 'documentary')
        lighting = style.get('lighting', 'natural')
        
        # Add character context
        char_details = scene.get('character_details', [])
        
        # Add location context
        loc_details = scene.get('location_details', {})
        atmosphere = loc_details.get('atmosphere', '')
        
        # Build a rich prompt
        full_prompt = (
            f"{prompt}. "
            f"Style: {style_str} documentary. "
            f"Lighting: {lighting}. "
            f"Atmosphere: {atmosphere}. "
            f"Characters: {', '.join(char_details)}. "
            f"Visual style: {cinematography} camerawork."
        )
        
        return full_prompt
    
    def get_pending_units(self) -> List[ProductionUnit]:
        """Get units not yet assigned or generated"""
        return [u for u in self.units if u.status in ['pending', 'failed']]
    
    def get_assigned_units(self, worker_id: str) -> List[ProductionUnit]:
        """Get units assigned to a specific worker"""
        return [u for u in self.units if u.assigned_worker == worker_id]
    
    def update_unit_status(self, scene_id: int, status: str, worker_id: str = None):
        """Update a unit's status"""
        if scene_id in self._unit_map:
            unit = self._unit_map[scene_id]
            unit.status = status
            if worker_id:
                unit.assigned_worker = worker_id
    
    def get_unit_by_id(self, scene_id: int) -> Optional[ProductionUnit]:
        return self._unit_map.get(scene_id)
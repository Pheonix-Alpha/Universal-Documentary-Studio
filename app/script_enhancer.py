"""
Script Enhancer - Improves dialogue, pacing, and narrative structure
"""

from typing import Dict, List, Optional
import json

from app import model_manager
from app.production_bible import ProductionBible


def enhance_script(story: str, bible: ProductionBible) -> str:
    """
    Enhance the script with better dialogue, pacing, and narrative flow.
    Uses the local "brain" LLM running on the main Colab's own GPU.
    """
    try:
        return _enhance_with_local_brain(story, bible)
    except Exception as e:
        print(f"[script_enhancer] Local brain enhancement failed, falling back: {e}")
        return _enhance_fallback(story, bible)


def _enhance_with_local_brain(story: str, bible: ProductionBible) -> str:
    # Build character context
    char_context = "\n".join([
        f"- {name}: {char.role} - {char.personality_traits}"
        for name, char in bible.characters.items()
    ])
    
    prompt = f"""
You are a documentary script editor. Improve this script for:
1. Better narrative pacing
2. More engaging dialogue
3. Clearer narrative structure
4. Stronger character voices
5. Better scene transitions

PRODUCTION CONTEXT:
Title: {bible.title}
Genre: {bible.genre}
Characters: {char_context}

ORIGINAL SCRIPT:
{story}

Return the enhanced script as plain text. Keep the same events and scenes.
"""
    
    text = model_manager.generate_text(
        user_prompt=prompt,
        system_prompt="You are a documentary script editor. Improve scripts.",
        max_new_tokens=4096,
        temperature=0.5,
    )
    return text or story


def _enhance_fallback(story: str, bible: ProductionBible) -> str:
    """Basic fallback enhancement"""
    # Simple improvements: add scene breaks, fix formatting
    lines = story.split('\n')
    enhanced = []
    
    for line in lines:
        line = line.strip()
        if line and not line.startswith('[') and not line.startswith('#'):
            # Add context where missing
            if 'scene' in line.lower() and not line.startswith('['):
                enhanced.append(f"[Scene: {line}]")
            else:
                enhanced.append(line)
        else:
            enhanced.append(line)
    
    return '\n'.join(enhanced)
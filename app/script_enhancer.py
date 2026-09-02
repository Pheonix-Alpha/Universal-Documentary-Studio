"""
Script Enhancer - Improves dialogue, pacing, and narrative structure
"""

from typing import Dict, List, Optional
import json

from app import config
from app.production_bible import ProductionBible


def enhance_script(story: str, bible: ProductionBible) -> str:
    """
    Enhance the script with better dialogue, pacing, and narrative flow.
    """
    if config.is_key_set('anthropic'):
        return _enhance_with_claude(story, bible)
    return _enhance_fallback(story, bible)


def _enhance_with_claude(story: str, bible: ProductionBible) -> str:
    import anthropic
    
    api_key = config.get_key('anthropic')
    if not api_key:
        return story
    
    client = anthropic.Anthropic(api_key=api_key)
    
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
    
    try:
        response = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=4096,
            temperature=0.5,
            system="You are a documentary script editor. Improve scripts.",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    except Exception as e:
        print(f"Script enhancement failed: {e}")
        return story


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
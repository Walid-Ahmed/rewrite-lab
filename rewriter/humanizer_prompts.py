"""Versioned training/inference prompts; old checkpoints retain their original policy."""
import hashlib
import json
from pathlib import Path

MODES=('academic',)
VIEWS=('Humanized','Polished')
LEGACY_PROMPT_POLICY='mode_view_conditioned_v1'
NATURAL_PROMPT_POLICY='natural_academic_v2'
PROMPT_POLICIES=(LEGACY_PROMPT_POLICY,NATURAL_PROMPT_POLICY)
NATURAL_PROMPT_FILE=Path(__file__).resolve().parent/'prompts/humanizer-natural-academic-v2.txt'


def prompt_metadata(policy):
    if policy not in PROMPT_POLICIES:raise ValueError('Unknown Humanizer prompt policy')
    result={'prompt_policy':policy}
    if policy==NATURAL_PROMPT_POLICY:
        text=NATURAL_PROMPT_FILE.read_text().strip()
        if not text:raise ValueError('Humanizer prompt file is empty')
        result.update(system_prompt=text,system_prompt_sha256=hashlib.sha256(text.encode()).hexdigest(),
                      system_prompt_file='prompts/humanizer-natural-academic-v2.txt')
    return result


def validated_prompt_text(metadata):
    """Use the export/checkpoint snapshot so later repo edits cannot change inference."""
    policy=metadata.get('prompt_policy',LEGACY_PROMPT_POLICY)
    if policy not in PROMPT_POLICIES:raise ValueError('Unknown Humanizer prompt policy')
    if policy==LEGACY_PROMPT_POLICY:return None
    text=metadata.get('system_prompt')
    if not isinstance(text,str) or not text.strip() or hashlib.sha256(text.encode()).hexdigest()!=metadata.get('system_prompt_sha256'):
        raise ValueError('Missing or changed Humanizer prompt snapshot/hash')
    return text


def humanizer_system_prompt(mode,view,policy=LEGACY_PROMPT_POLICY,prompt_text=None):
    if mode not in MODES or view not in VIEWS:raise ValueError('Choose a supported mode and output view')
    if policy==NATURAL_PROMPT_POLICY:
        return prompt_text if prompt_text is not None else prompt_metadata(policy)['system_prompt']
    if policy!=LEGACY_PROMPT_POLICY:raise ValueError('Unknown Humanizer prompt policy')
    return f"Rewrite the paragraph for {mode}, using Humanizer B's {view} output style. Preserve its meaning, facts, numbers and technical identifiers. Return only the rewritten paragraph."


def inference_system_prompt(adapter,mode=None,view='Humanized'):
    from train_qwen3_humanizer import SYSTEM
    config_file=Path(adapter)/'adapter_config.json'
    config=json.loads(config_file.read_text()) if config_file.exists() else {}
    metadata=config.get('dataset_provenance',{})
    policy=metadata.get('prompt_policy')
    if policy==NATURAL_PROMPT_POLICY:
        if mode is not None and mode not in MODES:raise ValueError('Choose a supported mode')
        if view!='Humanized':raise ValueError('This generic academic adapter is trained for Humanized output only')
        return validated_prompt_text(metadata)
    if policy is not None and policy!=LEGACY_PROMPT_POLICY:raise ValueError('Unknown Humanizer prompt policy')
    if policy==LEGACY_PROMPT_POLICY and not mode:
        raise ValueError('This adapter requires --mode; choose the desired target and --output-view')
    return humanizer_system_prompt(mode,view) if mode else SYSTEM

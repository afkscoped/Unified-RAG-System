"""
Persona Selector Component
"""
import streamlit as st
from src.enhancements.persona_engine import PersonaEngine

def render_persona_selector(default_id: str = 'scientist') -> str:
    """Renders buttons for persona selection"""
    st.markdown("### 🎭 AI Persona")
    
    # Initialize state
    if 'selected_persona' not in st.session_state:
        st.session_state.selected_persona = default_id
        
    current = st.session_state.selected_persona
    
    # Render loop
    cols = st.columns(len(PersonaEngine.PERSONAS))
    for col, (pid, persona) in zip(cols, PersonaEngine.PERSONAS.items()):
        with col:
            # Visual feedback for selection
            is_active = (pid == current)
            if st.button(
                f"{persona.icon}", 
                key=f"persona_btn_{pid}",
                help=f"{persona.name}: {persona.description}",
                use_container_width=True,
                type="primary" if is_active else "secondary"
            ):
                st.session_state.selected_persona = pid
                st.rerun()
                
    # Display active persona name below
    active_persona = PersonaEngine.PERSONAS.get(current)
    if active_persona:
        st.caption(f"**Current**: {active_persona.name} - {active_persona.description}")
        
    return current

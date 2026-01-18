"""
StoryWeaver Story Mode UI - Enhanced

Interactive story generation with side-by-side comparison of:
- Unified RAG
- Graph RAG  
- Hybrid Fusion

Enhanced Features:
- Three-column comparison view with metrics
- Multi-dimensional coherence analysis (radar chart)
- Entity and relationship display
- Plot suggestions with "Use This" buttons
- Interactive knowledge graph visualization
- Plot timeline visualization
- Consistency violation comparison
- Feedback system with analytics
- Story export
"""

import streamlit as st
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

st.set_page_config(
    page_title="StoryWeaver - Story Mode",
    page_icon="✍️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .comparison-header {
        text-align: center;
        padding: 0.5rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .unified-header { background: linear-gradient(90deg, #11998e, #38ef7d); color: white; }
    .graph-header { background: linear-gradient(90deg, #667eea, #764ba2); color: white; }
    .hybrid-header { background: linear-gradient(90deg, #f093fb, #f5576c); color: white; }
    .metric-box {
        background: #f0f2f6;
        padding: 0.5rem;
        border-radius: 8px;
        text-align: center;
    }
    .story-segment {
        background: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        min-height: 200px;
    }
    .entity-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        margin: 0.25rem 0;
    }
    .suggestion-card {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }
    .violation-card {
        background: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 0.75rem;
        margin: 0.25rem 0;
        border-radius: 0 8px 8px 0;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    defaults = {
        'comparison_engine': None,
        'story_history': [],
        'current_chapter': 1,
        'story_title': "Untitled Story",
        'initialized': False,
        'last_results': None,
        'active_tab': 'generation',
        'selected_prompt': None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def initialize_engine():
    """Initialize the comparison engine."""
    if st.session_state.comparison_engine is not None:
        return True
    
    try:
        from src.story.comparison_engine import StoryComparisonEngine
        from src.llm.llm_router import LLMRouter
        
        try:
            llm_router = LLMRouter()
            st.session_state.comparison_engine = StoryComparisonEngine(
                llm_router=llm_router
            )
            st.session_state.initialized = True
            return True
        except Exception as e:
            st.warning(f"LLM initialization warning: {e}")
            st.session_state.comparison_engine = StoryComparisonEngine()
            st.session_state.initialized = True
            return True
            
    except Exception as e:
        st.error(f"Failed to initialize story engine: {e}")
        return False


def render_sidebar():
    """Render sidebar with settings and navigation."""
    with st.sidebar:
        st.title("📖 StoryWeaver")
        
        st.session_state.story_title = st.text_input(
            "Story Title",
            st.session_state.story_title
        )
        
        st.divider()
        
        # Tab navigation
        st.subheader("📑 Views")
        tab_options = ["🎬 Generation", "📊 Visualization", "⭐ Feedback", "📈 Analytics"]
        selected = st.radio("Navigate to:", tab_options, label_visibility="collapsed")
        st.session_state.active_tab = {
            "🎬 Generation": "generation",
            "📊 Visualization": "visualization", 
            "⭐ Feedback": "feedback",
            "📈 Analytics": "analytics"
        }[selected]
        
        st.divider()
        
        st.subheader("⚙️ Settings")
        st.selectbox(
            "Generation Mode",
            ["Compare All Three", "Unified RAG Only", "Graph RAG Only", "Hybrid Fusion Only"],
            key="generation_mode"
        )
        
        # Recommended approach
        if st.session_state.comparison_engine:
            recommended = st.session_state.comparison_engine.get_recommended_approach()
            st.info(f"💡 Recommended: **{recommended.title()}** (based on feedback)")
        
        st.divider()
        
        # Statistics
        if st.session_state.comparison_engine:
            stats = st.session_state.comparison_engine.get_statistics()
            st.metric("Segments Generated", stats.get("total_segments", 0))
            st.metric("Characters Tracked", stats.get("characters_tracked", 0))
        
        st.divider()
        
        if st.button("🏠 Back to Landing", use_container_width=True):
            st.switch_page("landing_page.py")


def render_generation_tab():
    """Render the main story generation tab."""
    st.title(f"✍️ {st.session_state.story_title}")
    st.caption(f"Chapter {st.session_state.current_chapter} • Comparative Story Generation")
    
    # Initialize engine
    if not st.session_state.initialized:
        with st.spinner("Initializing story engine..."):
            if not initialize_engine():
                st.error("Failed to initialize. Please check your configuration.")
                st.stop()
    
    # Prompt input
    prompt = st.session_state.get('selected_prompt', '') or ""
    prompt = st.text_area(
        "Enter your story prompt:",
        value=prompt,
        placeholder="Example: Elena discovers a hidden chamber beneath the castle...",
        height=100,
        key="story_prompt_input"
    )
    
    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
    with col1:
        generate_btn = st.button("🎬 Generate", type="primary", use_container_width=True)
    with col2:
        continue_btn = st.button("➡️ Next Chapter", use_container_width=True)
    with col3:
        suggestions_btn = st.button("💡 Suggestions", use_container_width=True)
    with col4:
        export_btn = st.button("📥 Export", use_container_width=True)
    
    if continue_btn:
        st.session_state.current_chapter += 1
        st.rerun()
    
    if export_btn:
        export_story()
    
    # Show plot suggestions
    if suggestions_btn:
        render_plot_suggestions()
    
    # Generate
    if generate_btn:
        if not prompt or prompt.strip() == "":
            st.warning("Please enter a story prompt first!")
        else:
            with st.spinner("Generating with all three approaches... (10-30 seconds)"):
                try:
                    results = st.session_state.comparison_engine.generate_comparative(
                        prompt, st.session_state.current_chapter
                    )
                    st.session_state.last_results = results
                    st.session_state.story_history.append({
                        "chapter": st.session_state.current_chapter,
                        "prompt": prompt,
                        "results": results
                    })
                    st.session_state.selected_prompt = None
                    st.success("✅ Generation complete!")
                except Exception as e:
                    st.error(f"Generation failed: {e}")
                    import traceback
                    st.code(traceback.format_exc())
    
    # Display results
    if st.session_state.last_results:
        render_results(st.session_state.last_results)


def render_results(results: Dict):
    """Render generation results."""
    st.divider()
    
    mode = st.session_state.get('generation_mode', 'Compare All Three')
    
    if mode == "Compare All Three":
        render_comparison_results(results)
        st.divider()
        render_coherence_breakdown(results)
        st.divider()
        render_consistency_comparison(results)
    else:
        mode_key = {
            "Unified RAG Only": "unified",
            "Graph RAG Only": "graph",
            "Hybrid Fusion Only": "hybrid"
        }.get(mode, "hybrid")
        render_single_result(results[mode_key], mode)
    
    # Entity and relationship display for Graph RAG
    if "graph" in results:
        st.divider()
        render_entities_and_relationships(results["graph"])


def render_comparison_results(results: Dict):
    """Render three-column comparison."""
    st.subheader("📝 Generated Stories")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="comparison-header unified-header"><b>🔍 Unified RAG</b></div>', unsafe_allow_html=True)
        unified = results.get("unified", {})
        st.write(unified.get("text", "No text generated")[:2000])
        metrics = unified.get("metrics")
        if metrics:
            render_metrics_row(metrics)
    
    with col2:
        st.markdown('<div class="comparison-header graph-header"><b>🕸️ Graph RAG</b></div>', unsafe_allow_html=True)
        graph = results.get("graph", {})
        st.write(graph.get("text", "No text generated")[:2000])
        metrics = graph.get("metrics")
        if metrics:
            render_metrics_row(metrics)
    
    with col3:
        st.markdown('<div class="comparison-header hybrid-header"><b>⚡ Hybrid Fusion</b></div>', unsafe_allow_html=True)
        hybrid = results.get("hybrid", {})
        st.write(hybrid.get("text", "No text generated")[:2000])
        metrics = hybrid.get("metrics")
        if metrics:
            render_metrics_row(metrics)


def render_metrics_row(metrics):
    """Render metrics for a result."""
    if hasattr(metrics, 'to_dict'):
        m = metrics.to_dict()
    else:
        m = metrics
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("⏱️ Time", f"{m.get('response_time', 0):.1f}s")
    with col2:
        st.metric("📊 Coherence", f"{m.get('coherence_score', 0):.0%}")
    with col3:
        st.metric("✅ Consistency", f"{m.get('consistency_score', 0):.0%}")


def render_coherence_breakdown(results: Dict):
    """Render coherence breakdown with radar chart."""
    st.subheader("🎯 Coherence Analysis")
    
    try:
        import plotly.graph_objects as go
        
        # Get coherence analysis for hybrid (best)
        hybrid_text = results.get("hybrid", {}).get("text", "")
        if hybrid_text and st.session_state.comparison_engine:
            coherence = st.session_state.comparison_engine.analyze_coherence(
                hybrid_text, "", st.session_state.current_chapter
            )
            
            # Radar chart
            categories = ['Semantic', 'Lexical', 'Discourse', 'Temporal']
            values = [
                coherence.get("semantic", 0),
                coherence.get("lexical", 0),
                coherence.get("discourse", 0),
                coherence.get("temporal", 0)
            ]
            values.append(values[0])  # Close the radar
            
            fig = go.Figure(data=go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],
                fill='toself',
                line_color='#667eea'
            ))
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 1])
                ),
                showlegend=False,
                height=300,
                margin=dict(l=50, r=50, t=30, b=30)
            )
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                st.markdown("**Dimension Scores:**")
                for cat, val in zip(categories, values[:-1]):
                    st.progress(val, text=f"{cat}: {val:.0%}")
                st.metric("Composite Score", f"{coherence.get('composite', 0):.0%}")
    except Exception as e:
        st.warning(f"Could not render coherence analysis: {e}")


def render_consistency_comparison(results: Dict):
    """Render consistency comparison showing Graph RAG advantage."""
    st.subheader("🛡️ Consistency Check - Graph RAG vs Unified RAG")
    
    try:
        engine = st.session_state.comparison_engine
        if not engine:
            return
        
        unified_text = results.get("unified", {}).get("text", "")
        graph_text = results.get("graph", {}).get("text", "")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Unified RAG Issues:**")
            unified_report = engine.check_consistency(unified_text, "unified")
            violations = unified_report.get("violations", [])
            if violations:
                for v in violations[:3]:
                    st.markdown(f'<div class="violation-card">⚠️ <b>{v.get("type", "Unknown")}</b>: {v.get("description", "")}</div>', unsafe_allow_html=True)
            else:
                st.success("No violations detected")
            st.metric("Consistency Score", f"{unified_report.get('score', 0):.0%}")
        
        with col2:
            st.markdown("**Graph RAG (Prevented Issues):**")
            graph_report = engine.check_consistency(graph_text, "graph")
            prevented = graph_report.get("prevented_by_graph", [])
            if prevented:
                for p in prevented[:3]:
                    st.success(f"✅ {p}")
            else:
                st.info("Using graph structure to maintain consistency")
            st.metric("Consistency Score", f"{graph_report.get('score', 0):.0%}")
    
    except Exception as e:
        st.warning(f"Consistency check unavailable: {e}")


def render_entities_and_relationships(graph_result: Dict):
    """Render extracted entities and relationships."""
    st.subheader("🧩 Extracted Entities & Relationships")
    
    entities = graph_result.get("entities", [])
    relationships = graph_result.get("relationships", [])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Characters & Entities:**")
        if entities:
            for entity in entities[:10]:
                name = entity.get("name", "Unknown")
                etype = entity.get("type", "ENTITY")
                conf = entity.get("confidence", 0.5)
                icon = {"CHARACTER": "👤", "LOCATION": "📍", "EVENT": "⚡", "ARTIFACT": "🔮"}.get(etype, "📌")
                st.markdown(f'{icon} **{name}** ({etype}) - {conf:.0%} confidence')
        else:
            st.info("No entities extracted yet")
    
    with col2:
        st.markdown("**Relationships:**")
        if relationships:
            for rel in relationships[:10]:
                if isinstance(rel, dict):
                    src = rel.get("source", "")
                    tgt = rel.get("target", "")
                    rtype = rel.get("relation_type", "RELATED")
                    st.markdown(f"🔗 {src} → **{rtype}** → {tgt}")
        else:
            st.info("No relationships extracted yet")


def render_plot_suggestions():
    """Render plot suggestions with Use This buttons."""
    st.subheader("💡 Plot Suggestions")
    
    engine = st.session_state.comparison_engine
    if not engine:
        return
    
    try:
        suggestions = engine.get_plot_suggestions(limit=5)
        
        if not suggestions:
            st.info("Generate more story content to receive plot suggestions!")
            return
        
        for i, sug in enumerate(suggestions):
            with st.expander(f"**{sug.get('title', 'Suggestion')}** ({sug.get('priority', 'medium')})"):
                st.write(sug.get("description", ""))
                st.code(sug.get("actionable_prompt", ""), language=None)
                
                if st.button(f"Use This Prompt", key=f"use_sug_{i}"):
                    st.session_state.selected_prompt = sug.get("actionable_prompt", "")
                    st.rerun()
    
    except Exception as e:
        st.warning(f"Could not load suggestions: {e}")


def render_visualization_tab():
    """Render the visualization tab with graphs and timeline."""
    st.title("📊 Story Visualization")
    
    engine = st.session_state.comparison_engine
    if not engine:
        st.warning("Initialize the engine first by generating a story.")
        return
    
    viz_type = st.selectbox(
        "Select Visualization",
        ["Knowledge Graph", "Character Network", "Event Chain", "Location Map", "Plot Timeline"]
    )
    
    try:
        if viz_type == "Knowledge Graph":
            fig = engine.create_graph_visualization("full")
        elif viz_type == "Character Network":
            fig = engine.create_graph_visualization("characters")
        elif viz_type == "Event Chain":
            fig = engine.create_graph_visualization("events")
        elif viz_type == "Location Map":
            fig = engine.create_graph_visualization("locations")
        elif viz_type == "Plot Timeline":
            fig = engine.create_timeline_visualization()
        
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available for this visualization yet. Generate more story content!")
    
    except Exception as e:
        st.error(f"Visualization error: {e}")
    
    # Timeline analysis
    if viz_type == "Plot Timeline":
        st.divider()
        st.subheader("📈 Story Structure Analysis")
        try:
            analysis = engine.get_timeline_analysis()
            
            col1, col2 = st.columns(2)
            with col1:
                st.json(analysis.get("structure", {}))
            with col2:
                holes = analysis.get("plot_holes", [])
                if holes:
                    st.warning(f"⚠️ {len(holes)} potential plot holes detected")
                    for hole in holes[:3]:
                        st.markdown(f"- {hole.get('description', '')}")
                else:
                    st.success("No plot holes detected!")
        except Exception as e:
            st.warning(f"Analysis unavailable: {e}")


def render_feedback_tab():
    """Render the feedback submission tab."""
    st.title("⭐ Rate Generation Quality")
    
    if not st.session_state.last_results:
        st.info("Generate a story first, then provide feedback!")
        return
    
    st.markdown("Rate the quality of each approach to help the system learn your preferences.")
    
    approach = st.radio("Select approach to rate:", ["Unified RAG", "Graph RAG", "Hybrid Fusion"])
    approach_key = {"Unified RAG": "unified", "Graph RAG": "graph", "Hybrid Fusion": "hybrid"}[approach]
    
    col1, col2 = st.columns(2)
    
    with col1:
        overall = st.slider("Overall Quality", 1, 5, 3, key="fb_overall")
        consistency = st.slider("Narrative Consistency", 1, 5, 3, key="fb_consistency")
        creativity = st.slider("Creativity", 1, 5, 3, key="fb_creativity")
    
    with col2:
        char_auth = st.slider("Character Authenticity", 1, 5, 3, key="fb_char")
        plot_coh = st.slider("Plot Coherence", 1, 5, 3, key="fb_plot")
        is_best = st.checkbox("This was the best approach", key="fb_best")
    
    if st.button("Submit Feedback", type="primary"):
        try:
            engine = st.session_state.comparison_engine
            feedback = engine.submit_feedback(
                approach=approach_key,
                prompt=st.session_state.story_history[-1].get("prompt", "") if st.session_state.story_history else "",
                overall=overall,
                consistency=consistency,
                creativity=creativity,
                character_authenticity=char_auth,
                plot_coherence=plot_coh,
                selected_best=is_best
            )
            st.success("✅ Feedback submitted! The system will learn from your preferences.")
        except Exception as e:
            st.error(f"Failed to submit feedback: {e}")


def render_analytics_tab():
    """Render the feedback analytics tab."""
    st.title("📈 Performance Analytics")
    
    engine = st.session_state.comparison_engine
    if not engine:
        st.warning("Initialize the engine first.")
        return
    
    try:
        analytics = engine.get_feedback_analytics()
        
        # Performance by approach
        st.subheader("🏆 Approach Performance")
        perf = analytics.get("performance_by_approach", {})
        
        if any(p.get("count", 0) > 0 for p in perf.values()):
            col1, col2, col3 = st.columns(3)
            with col1:
                u = perf.get("unified", {})
                st.metric("Unified RAG", f"{u.get('average_rating', 0):.1f}/5", f"{u.get('count', 0)} ratings")
            with col2:
                g = perf.get("graph", {})
                st.metric("Graph RAG", f"{g.get('average_rating', 0):.1f}/5", f"{g.get('count', 0)} ratings")
            with col3:
                h = perf.get("hybrid", {})
                st.metric("Hybrid Fusion", f"{h.get('average_rating', 0):.1f}/5", f"{h.get('count', 0)} ratings")
        else:
            st.info("Submit feedback to see analytics!")
        
        st.divider()
        
        # Current weights
        st.subheader("⚖️ Adaptive Weights")
        weights = analytics.get("weights", {})
        col1, col2, col3 = st.columns(3)
        with col1:
            st.progress(weights.get("unified", 0.33), text=f"Unified: {weights.get('unified', 0.33):.0%}")
        with col2:
            st.progress(weights.get("graph", 0.33), text=f"Graph: {weights.get('graph', 0.33):.0%}")
        with col3:
            st.progress(weights.get("hybrid", 0.34), text=f"Hybrid: {weights.get('hybrid', 0.34):.0%}")
        
        # Recommendations
        recs = analytics.get("recommendations", [])
        if recs:
            st.divider()
            st.subheader("💡 Recommendations")
            for rec in recs:
                st.info(f"**{rec.get('type', '')}**: {rec.get('suggestion', rec.get('reason', ''))}")
    
    except Exception as e:
        st.warning(f"Analytics unavailable: {e}")


def render_single_result(result: Dict, mode: str):
    """Render a single generation result."""
    st.subheader(f"📝 {mode}")
    st.write(result.get("text", "No text generated"))
    
    metrics = result.get("metrics")
    if metrics:
        render_metrics_row(metrics)


def export_story():
    """Export story to markdown."""
    if not st.session_state.story_history:
        st.warning("No story to export!")
        return
    
    content = f"# {st.session_state.story_title}\n\n"
    for entry in st.session_state.story_history:
        chapter = entry.get("chapter", 1)
        content += f"## Chapter {chapter}\n\n"
        content += f"*Prompt: {entry.get('prompt', '')}*\n\n"
        hybrid = entry.get("results", {}).get("hybrid", {})
        content += hybrid.get("text", "") + "\n\n"
    
    st.download_button(
        "📥 Download Story",
        content,
        file_name=f"{st.session_state.story_title.replace(' ', '_')}.md",
        mime="text/markdown"
    )


def main():
    init_session_state()
    render_sidebar()
    
    # Route to appropriate tab
    tab = st.session_state.active_tab
    
    if tab == "generation":
        render_generation_tab()
    elif tab == "visualization":
        render_visualization_tab()
    elif tab == "feedback":
        render_feedback_tab()
    elif tab == "analytics":
        render_analytics_tab()
    
    # Story history (always shown at bottom)
    if st.session_state.story_history and tab == "generation":
        st.divider()
        st.header("📜 Story Timeline")
        for entry in reversed(st.session_state.story_history[-5:]):
            with st.expander(f"Chapter {entry.get('chapter', '?')}: {entry.get('prompt', '')[:50]}..."):
                st.write(entry.get("results", {}).get("hybrid", {}).get("text", "")[:500])


if __name__ == "__main__":
    main()

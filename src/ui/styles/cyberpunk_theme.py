"""
Cyberpunk / Glitch Design System Theme Loader
"""
import streamlit as st

def load_cyberpunk_theme():
    """
    Injects the Cyberpunk/Glitch CSS design system into the Streamlit app.
    """
    
    # CSS Definition
    css = """
    <style>
        /* 1. IMPORTS & FONTS */
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&display=swap');

        /* 2. DESIGN TOKENS */
        :root {
            /* Colors */
            --bg-deep: #0a0a0f;
            --bg-card: #12121a;
            --bg-muted: #1c1c2e;
            
            --text-primary: #e0e0e0;
            --text-secondary: #6b7280;
            
            --neon-green: #00ff88;
            --neon-purple: #ff00ff;
            --neon-cyan: #00d4ff;
            --danger: #ff3366;
            
            --border-dim: #2a2a3a;
            
            /* Shadows / Glows */
            --glow-green: 0 0 5px #00ff88, 0 0 10px #00ff8840;
            --glow-green-lg: 0 0 10px #00ff88, 0 0 20px #00ff8860, 0 0 40px #00ff8830;
            
            /* Typography */
            --font-headings: 'Orbitron', monospace;
            --font-body: 'JetBrains Mono', monospace;
            --font-ui: 'Share Tech Mono', monospace;
        }

        /* 3. GLOBAL RESTYLING */
        
        /* Main Application Background */
        .stApp {
            background-color: var(--bg-deep);
            background-image: 
                linear-gradient(rgba(0, 255, 136, 0.03) 1px, transparent 1px),
                linear-gradient(90deg, rgba(0, 255, 136, 0.03) 1px, transparent 1px);
            background-size: 50px 50px;
            font-family: var(--font-body);
            color: var(--text-primary);
        }

        /* Scanline Overlay via pseudo-element on the main container */
        .stApp::before {
            content: " ";
            display: block;
            position: fixed;
            top: 0;
            left: 0;
            bottom: 0;
            right: 0;
            background: repeating-linear-gradient(
                0deg,
                transparent,
                transparent 2px,
                rgba(0, 0, 0, 0.1) 2px,
                rgba(0, 0, 0, 0.1) 4px
            );
            z-index: 9999;
            pointer-events: none;
            opacity: 0.6;
        }

        /* Titles & Headings */
        h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            font-family: var(--font-headings) !important;
            text-transform: uppercase;
            letter-spacing: 2px;
            color: white !important;
        }

        h1 {
            text-shadow: var(--glow-green);
        }

        /* 4. COMPONENT OVERRIDES */

        /* Buttons (stButton) */
        .stButton > button {
            background: transparent;
            border: 1px solid var(--neon-green);
            color: var(--neon-green);
            font-family: var(--font-ui);
            text-transform: uppercase;
            letter-spacing: 1px;
            border-radius: 0; /* Chamfered look requires clip-path, but 0 radius helps */
            position: relative;
            transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
            
            /* Chamfer effect */
            clip-path: polygon(
                0 10px, 10px 0, 
                100% 0, 100% calc(100% - 10px), 
                calc(100% - 10px) 100%, 0 100%
            );
        }

        .stButton > button:hover {
            background-color: var(--neon-green);
            color: var(--bg-deep);
            box-shadow: var(--glow-green-lg);
            border-color: var(--neon-green);
            transform: translateY(-2px);
        }
        
        /* Secondary Buttons (mapped to secondary type if possible, or use custom CSS wrappers) */
        
        /* Inputs (TextInput, NumberInput, etc.) */
        .stTextInput > div > div > input, .stNumberInput > div > div > input, .stTextArea > div > div > textarea {
            background-color: var(--bg-card);
            color: var(--neon-green);
            border: 1px solid var(--border-dim);
            font-family: var(--font-body);
            border-radius: 0;
        }
        
        .stTextInput > div > div > input:focus, .stTextArea > div > div > textarea:focus {
            border-color: var(--neon-green);
            box-shadow: var(--glow-green);
            color: white;
        }

        /* Streamlit Expander */
        .streamlit-expanderHeader {
            background-color: var(--bg-card);
            border: 1px solid var(--border-dim);
            font-family: var(--font-ui);
            color: var(--text-primary);
        }
        
        /* Streamlit Metrics */
        [data-testid="stMetricValue"] {
            font-family: var(--font-headings);
            color: var(--neon-cyan);
            text-shadow: 0 0 10px rgba(0, 212, 255, 0.5);
        }

        /* 5. CUSTOM UTILITY CLASSES (Use with st.markdown specific containers) */
        
        /* Cyber Card Container */
        .cyber-card {
            background: var(--bg-card);
            border: 1px solid var(--border-dim);
            padding: 1.5rem;
            margin-bottom: 1rem;
            position: relative;
            clip-path: polygon(
                0 10px, 10px 0, 
                100% 0, 100% calc(100% - 10px), 
                calc(100% - 10px) 100%, 0 100%
            );
            transition: transform 0.3s ease, border-color 0.3s ease;
        }
        
        .cyber-card:hover {
            border-color: var(--neon-green);
            box-shadow: var(--glow-green);
            transform: translateY(-2px);
        }
        
        .cyber-header {
            font-family: var(--font-headings);
            color: var(--neon-green);
            margin-bottom: 0.5rem;
            font-size: 1.2rem;
            display: flex;
            align-items: center;
        }
        
        .cyber-header::before {
            content: ">";
            margin-right: 8px;
            color: var(--neon-purple);
            animation: blink 1s step-end infinite;
        }
        
        /* Glitch Animation Keyframes */
        @keyframes blink {
            50% { opacity: 0; }
        }
        
        @keyframes glitch {
            0% { transform: translate(0); }
            20% { transform: translate(-2px, 2px); }
            40% { transform: translate(-2px, -2px); }
            60% { transform: translate(2px, 2px); }
            80% { transform: translate(2px, -2px); }
            100% { transform: translate(0); }
        }
        
        .glitch-hover:hover {
            animation: glitch 0.3s cubic-bezier(.25, .46, .45, .94) both infinite;
        }

    </style>
    """
    
    st.markdown(css, unsafe_allow_html=True)

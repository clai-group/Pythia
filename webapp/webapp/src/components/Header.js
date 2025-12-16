import React from 'react';
import { useState, useEffect } from 'react';

const Header = () => {
  const REPO_LINK = 'https://github.com/clai-group/Pythia';
  const contactEmail = 'pythia@clai.group'; // Replace with your contact email

  const [stars, setStars] = useState(3);

  useEffect(() => {
    fetch('https://api.github.com/repos/clai-group/Pythia')
      .then(response => {
        if (!response.ok) {
          return;
        }
        return response.json();
      })
      .then(data => {
        if (data && data.stargazers_count !== undefined) {
          setStars(data.stargazers_count);
        }
      })
      .catch(error => {
        console.error('Error fetching GitHub stars:', error);
      });
  }, []);

  const scrollToSection = (sectionId) => {
    const element = document.getElementById(sectionId);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  };

  const scrollToTop = () => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  return (
    <header>
      <nav className="container">
        <div className="logo" style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <button
            onClick={scrollToTop}
            style={{
              background: 'none',
              border: 'none',
              fontSize: '28px',
              fontWeight: '600',
              letterSpacing: '-0.5px',
              fontFamily: "'Courier New', monospace",
              cursor: 'pointer',
              textDecoration: 'none',
              padding: 0,
              display: 'flex',
              alignItems: 'center',
              gap: '12px'
            }}
          >
            <div className="logo-mark">🧠</div>
            <div style={{ textAlign: 'left' }}>
              <div style={{ color: '#0f172a' }}>Pythia</div>
              <div style={{ fontSize: '12px', color: '#64748b', letterSpacing: 0 }}>Agentic prompt refinement</div>
            </div>
          </button>
        </div>
        <ul className="nav-links">
          <li><button onClick={() => scrollToSection('paper')}>Paper</button></li>
          <li><button onClick={() => scrollToSection('demos')}>Workflow</button></li>
      <li><button onClick={() => scrollToSection('installation')}>Run Pythia</button></li>
    </ul>
    <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
      <a
        href={`mailto:${contactEmail}`}
        style={{
          background: '#0052ff',
          color: '#fff',
          textDecoration: 'none',
          fontWeight: 600,
          padding: '10px 14px',
          borderRadius: '8px',
          display: 'inline-flex',
          alignItems: 'center',
          gap: '8px',
          boxShadow: '0 6px 16px rgba(0, 82, 255, 0.25)'
        }}
      >
        <span>✉️</span>
        <span>{contactEmail}</span>
      </a>
      <a href={REPO_LINK} target="_blank" rel="noopener noreferrer" className="btn-github">
        <span className="star-count">{stars.toLocaleString()} ⭐</span> Star on GitHub
      </a>
    </div>
  </nav>
    </header>
  );
};

export default Header;

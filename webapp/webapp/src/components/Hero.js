import React, { useState } from 'react';

const Hero = () => {
  const [isCopied, setIsCopied] = useState(false);
  const paperLink = 'https://example.com/pythia-paper'; // Replace with final paper link

  const handleScroll = (id) => {
    const element = document.getElementById(id);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
    }
  };

  const copyToClipboard = () => {
    navigator.clipboard.writeText('pip install pythia-tool')
      .then(() => {
        setIsCopied(true);
        setTimeout(() => setIsCopied(false), 2000);
      })
      .catch(err => {
        console.error('Failed to copy: ', err);
      });
  };

  const handlePaperClick = () => {
    if (paperLink) {
      window.open(paperLink, '_blank', 'noopener,noreferrer');
    } else {
      handleScroll('paper');
    }
  };

  return (
    <section className="hero">
      <div className="container">
        <div className="hero-content">
          <div className="hero-text">
            <h1>Meet Pythia</h1>
            <p className="subtitle">An iterative tool that helps your LLM improve its own prompts using the datasets and targets you provide.</p>
            <p className="description">Pythia runs a multi-agent loop to test prompts, learn from false positives and false negatives, and deliver stronger, reusable prompts without manual tweaking.</p>
            <div className="hero-cta-group">
              <button onClick={handlePaperClick} className="cta-button">
                <span>📄</span> Read Paper
              </button>
              <button onClick={copyToClipboard} className="cta-button-secondary">
                <span role="img" aria-label={isCopied ? 'check mark' : 'laptop'}>
                  {isCopied ? '✅' : '💻'}
                </span>
                {isCopied ? 'Copied!' : 'Copy pip command'}
              </button>
            </div>
          </div>
          <div className="hero-visual">
            <div className="laptop-mockup">
              <div className="laptop-frame">
                <div className="laptop-screen">
                  <div className="screen-content">
                    <div className="app-header">
                      <div className="traffic-lights">
                        <div className="traffic-light red"></div>
                        <div className="traffic-light yellow"></div>
                        <div className="traffic-light green"></div>
                      </div>
                      <div className="app-title">Pythia Agentic Loop</div>
                    </div>
                    <div className="dashboard-image-container">
                      <div className="asset-placeholder">
                        <div>Hero visual placeholder</div>
                        <div style={{ fontSize: '12px', color: '#94a3b8' }}>Drop hero image at public/assets/hero-visual.png</div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Hero;

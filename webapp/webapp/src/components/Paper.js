import React from 'react';

const Paper = () => {
  const paperLink = 'https://example.com/pythia-paper.pdf'; // Replace with final PDF/arXiv link

  return (
    <section className="paper-section" id="paper">
      <div className="container">
        <div className="section-header fade-in">
          <h2>Research Paper</h2>
          <p>Agentic prompt refinement against labeled data and validation with XYZ (depending on ArXiv paper).</p>
        </div>

        <div className="paper-preview fade-in" onClick={() => window.open(paperLink, '_blank', 'noopener,noreferrer')}>
          <div className="paper-header">
            <div className="arxiv-badge">Preprint</div>
            <div className="paper-id">Link pending</div>
          </div>
          <h3 className="paper-title">Coming soon...</h3>
          <div className="paper-authors">Coming soon...</div>
          <div className="paper-abstract">
            Coming soon...          </div>
          <div className="paper-stats">
            <div className="paper-stat">
              <span>📊</span>
              <span>Results box 1</span>
            </div>
            <div className="paper-stat">
              <span>✅</span>
              <span>Results box 2</span>
            </div>
            <div className="paper-stat">
              <span>🤖</span>
              <span>Results box 3</span>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Paper;

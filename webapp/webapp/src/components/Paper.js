import React from 'react';

const Paper = () => {
  const paperLink = 'https://example.com/pythia-paper.pdf'; // Replace with final PDF/arXiv link

  return (
    <section className="paper-section" id="paper">
      <div className="container">
        <div className="section-header fade-in">
          <h2>Research Paper</h2>
          <p>Agentic prompt refinement for cognitive concern detection in clinical notes.</p>
        </div>

        <div className="paper-preview fade-in" onClick={() => window.open(paperLink, '_blank', 'noopener,noreferrer')}>
          <div className="paper-header">
            <div className="arxiv-badge">Preprint</div>
            <div className="paper-id">Link pending</div>
          </div>
          <h3 className="paper-title">Autonomous Agentic LLM Workflow for Cognitive Concern Detection</h3>
          <div className="paper-authors">Author list placeholder — update with final authorship</div>
          <div className="paper-abstract">
            ABSTRACT PLACEHOLDER          </div>
          <div className="paper-stats">
            <div className="paper-stat">
              <span>📊</span>
              <span>Refinement: 2,228 notes (50% positive); Validation: 1,110 notes (33% positive)</span>
            </div>
            <div className="paper-stat">
              <span>✅</span>
              <span>Expert-guided (XP3) validation: F1 0.81, Sens 0.82, Spec 0.93 (post re-adjudication)</span>
            </div>
            <div className="paper-stat">
              <span>🤖</span>
              <span>Agentic (AP3) validation: F1 0.74, Sens 0.62, Spec 0.98 with transparent agent logs</span>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Paper;

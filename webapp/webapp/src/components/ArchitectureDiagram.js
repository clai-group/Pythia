import React from 'react';

const ArchitectureDiagram = () => {
  return (
    <section className="architecture-section">
      <div className="container">
        <div className="section-header fade-in">
          <h2>Architecture Overview</h2>
          <p>How the agentic workflow coordinates sensitivity, specificity, and reasoning agents.</p>
        </div>

        <div className="architecture-diagram fade-in">
          <div
            className="asset-placeholder"
            style={{
              width: '80%',
              maxWidth: '800px',
              height: '280px',
              display: 'block',
              margin: '0 auto'
            }}
          >
            <div>Architecture diagram placeholder</div>
            <div style={{ fontSize: '12px', color: '#94a3b8' }}>Place final diagram at public/assets/architecture.png</div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default ArchitectureDiagram;
